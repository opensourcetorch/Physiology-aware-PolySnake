# _*_ coding: utf-8 _*_

""" define train and test functions here """

import matplotlib as mpl
mpl.use('Agg')

import imageio
import sys
sys.path.append("..")
import copy
import numpy as np
np.set_printoptions(precision=4)
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os.path as osp
import os
import pickle

from metric import cal_f_score, cal_f_score_slicewise, volumewise_hd95, volumewise_asd
from vision import plot_metrics, plot_class_f1, sample_results
from snake import probmap2bound
from utils import innerouterbound2mask, AverageMeter, mask2innerouterbound

import warnings
warnings.filterwarnings('ignore', module='imageio')

def train_model(model, criterion, optimizer, scheduler, args):
    """ train the model
    Args:
        model: model inheriting from nn.Module class
        criterion: criterion class, loss function used
        optimizer: optimizer, optimization strategy
        scheduler: lr scheduler
        args: parser arguments
    """

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1.0e9
    loss_up_keep = 0 # check how many times the val loss has de
    # test': []} # class-wise F1 score

    # for hard mining
    metric_prev_epoch = None
    phases_prev_epoch = None

    # start training
    for epoch in range(args.num_train_epochs):
        print("{}/{}".format(epoch+1, args.num_train_epochs))
        if epoch != 0 and epoch % args.n_epoch_hardmining == 0:
            is_hard_mining = True
        else:
            is_hard_mining = False

        if args.model_type == '2d':
            from image.dataloader import read_train_data
            dataloaders = read_train_data(args.data_dir, args.compose, 'train', metric_prev_epoch, phases_prev_epoch, True,
                                  is_hard_mining, args.num_workers, args.batch_size, args.percentile, args.multi_view,
                                  args.config)

        else:  # parameters of dataloader for 2.5D and 3D is the same
            if args.model_type == '3d':
                from volume.dataloader import read_train_data
            elif args.model_type == '2.5d':
                from hybrid.dataloader import read_train_data

            dataloaders = read_train_data(args.data_dir, metric_prev_epoch, phases_prev_epoch, args.compose, 'train',
                                          is_hard_mining, args.percentile, args.multi_view, args.interval, args.down_sample,
                                          args.batch_size, args.num_workers, True, args.config)

        # during hard mining, if # of training samples is lower than threshold, stop training
        if len(dataloaders['train'].dataset.phases) <= 20:
            break

        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
                slicewise_metric_epoch = [] # for hard mining

            else:
                model.eval()  # Set model to evaluate mode

            loss_meter, correct_meter = AverageMeter(), AverageMeter()
            predhdf_meter, reghdf_meter, asd_meter =  AverageMeter(), AverageMeter(), AverageMeter()
            fscores_meter = [AverageMeter() for _ in range(args.output_channel)]

            dl_pbar = tqdm(dataloaders[phase])
            for sample_inx, sample in enumerate(dl_pbar):
                dl_pbar.update(100)
                inputs, labels = sample
                num_imgs_batch = len(inputs)

                # wrap them in Variable
                if args.use_gpu:
                    inputs = Variable(inputs.cuda()).float()
                    labels = Variable(labels.cuda()).long()

                else:
                    inputs = Variable(inputs).float()
                    labels = Variable(labels).long()

                optimizer.zero_grad()
                outputs = model(inputs)

                if args.criterion == 'nll':
                    loss = criterion(F.log_softmax(outputs, dim=1), labels)
                elif args.criterion == 'whddb':
                    loss = criterion(F.softmax(outputs, dim=1), labels)
                else: # dice, ce, wce et. al.
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs.data, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # various metrics
                loss_meter.update(loss.data.item() * num_imgs_batch, num_imgs_batch)
                correct_meter.update(float(torch.sum(preds == labels.data)) / preds[0].numel(), num_imgs_batch)

                # obtain prediction and label for segment and bound respectively
                if args.bound_out: # bound output
                    regs = probmap2bound(F.softmax(outputs, 1), n_workers=args.num_workers, thres=0.7, kernel_size=9)
                    preds_bound_np, labels_bound_np = preds.cpu().numpy(), labels.data.cpu().numpy()
                    regs_bound_np = regs.data.cpu().numpy()

                    preds_seg_np = np.stack([innerouterbound2mask(r, args.output_channel) for r in regs_bound_np])
                    labels_seg_np = np.stack([innerouterbound2mask(label, args.output_channel) for label in labels_bound_np])

                else: # segment output
                    preds_seg_np, labels_seg_np = preds.cpu().numpy(), labels.data.cpu().numpy()
                    preds_bound_np = np.stack([mask2innerouterbound(pred) for pred in preds_seg_np])
                    labels_bound_np = np.stack([mask2innerouterbound(label) for label in labels_seg_np])
                    regs_bound_np = preds_bound_np

                # calculate hd95 and asd
                mean_predhdf, batch_hdf = volumewise_hd95(preds_bound_np, labels_bound_np, return_slicewise_hdf=True)
                mean_reghdf = volumewise_hd95(regs_bound_np, labels_bound_np, return_slicewise_hdf=False)
                mean_asd = volumewise_asd(regs_bound_np, labels_bound_np, n_classes=3)
                predhdf_meter.update(mean_predhdf * num_imgs_batch, num_imgs_batch)
                reghdf_meter.update(mean_reghdf * num_imgs_batch, num_imgs_batch)
                asd_meter.update(mean_asd * num_imgs_batch, num_imgs_batch)

                # calculate class-wise F1
                cal_f1 = cal_f_score_slicewise if args.model_type == '3d' else cal_f_score
                _, f_scores, n_effect_samples = cal_f1(preds_seg_np, labels_seg_np, n_class=args.output_channel,
                                                       return_slice_f1=False, return_class_f1=True)
                for i in range(args.output_channel):
                    fscores_meter[i].update(f_scores[i], n_effect_samples[i])

                if phase == 'train':
                    slicewise_metric_epoch += batch_hdf

            dl_pbar.close()
            print()

            epoch_loss[phase].append(loss_meter.avg)
            epoch_acc[phase].append(correct_meter.avg)
            epoch_predhdf[phase].append(predhdf_meter.avg)
            epoch_reghdf[phase].append(reghdf_meter.avg)
            epoch_asd[phase].append(asd_meter.avg)

            running_f1_class = np.array([m.avg for m in fscores_meter])
            epoch_f1_score_class[phase].append(running_f1_class)  # f1 score for each class
            epoch_f1_score[phase].append(running_f1_class.mean())

            # print metrics
            print("[{:5s}({} samples)] Loss: {:.4f} Acc: {:.4f} Ave_F1: {:.4f} class-wise F1: {} Ave_predhdf: {:.4f} "
                  "Ave_reghdf: {:.4f} Ave_ASD: {:.4f}".format(phase, len(dataloaders[phase].dataset.phases),
            loss_meter.avg, correct_meter.avg, epoch_f1_score[phase][-1], running_f1_class,
            predhdf_meter.avg, reghdf_meter.avg, asd_meter.avg))

            # for hard mining, update metric_prev_epoch and phases_prev_epoch
            if phase == 'train':
                metric_prev_epoch = np.array(slicewise_metric_epoch)
                phases_prev_epoch = dataloaders['train'].dataset.phases

            # save the learnt best model evaluated on validation data
            if phase == 'val':
                val_loss_bf = sum(epoch_loss['val'][-5:]) / len(epoch_loss['val'][-5:])
                if val_loss_bf <= best_loss:
                    best_loss = val_loss_bf

                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model, args.model_save_name)

                if val_loss_bf > epoch_loss_prev:
                    loss_up_keep += 1
                else:
                    loss_up_keep = 0

                epoch_loss_prev = val_loss_bf

        # plot temporal loss, acc, f1_score for train, val and test respectively.
        if (epoch+1) % 5 == 0 and phase == 'test':
            metrics = [epoch_loss, epoch_acc, epoch_f1_score, epoch_asd, epoch_predhdf, epoch_reghdf]
            labels = ['total_loss', 'pixel_acc', 'F1_score', 'asd', 'hd95_pred', 'hd95_reg']
            plot_metrics(metrics, labels, fig_dir=args.fig_dir)
            plot_class_f1(epoch_f1_score_class, args.fig_dir)

        if loss_up_keep == 10:  # if val loss continuously increase for 10+ times --> over-fitting --> break
            break

    metrics = [epoch_loss, epoch_acc, epoch_f1_score, epoch_asd, epoch_predhdf, epoch_reghdf]
    labels = ['total_loss', 'pixel_acc', 'F1_score', 'asd', 'hd95_pred', 'hd95_reg']
    plot_metrics(metrics, labels, fig_dir=args.fig_dir)
    plot_class_f1(epoch_f1_score_class, args.fig_dir)

    # final results
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    torch.save(model, args.model_save_name)


def model_reference(args):
    """ model reference and plot the segmentation results
        for model reference, several epochs are used to balance the risks and other metrics
        for segmentation results plotting, only one epoch is used without data augmentation
    Args:
        args: parser arguments
        sample_stack_rows: int, how many slices to plot per image
    """

    #############################################################################################
    # Part 1: model reference and metric evaluations
    #############################################################################################
    model = torch.load(args.model_save_name, map_location=lambda storage, loc: storage)
    if args.use_gpu:
        model = model.cuda()

    correct_meter = AverageMeter()
    predhdf_meter, reghdf_meter, asd_meter = AverageMeter(), AverageMeter(), AverageMeter()
    fscores_meter = [AverageMeter() for _ in range(args.output_channel)]

    if args.model_type == '2d':
        from image.dataloader import read_train_data
        dataloaders = read_train_data(args.data_dir, args.compose, 'test', None, None, True,
                                      False, args.num_workers, args.batch_size, args.percentile,
                                      args.multi_view, args.config)
    else:  # parameters of dataloader for 2.5D and 3D is the same
        if args.model_type == '3d':
            from volume.dataloader import read_train_data
        elif args.model_type == '2.5d':
            from hybrid.dataloader import read_train_data

        dataloaders = read_train_data(args.data_dir, None, None, args.compose, 'test',
                                      False, args.percentile, args.multi_view, args.interval, args.down_sample,
                                      args.batch_size, args.num_workers, True, args.config)

    for samp_inx, sample in enumerate(dataloaders['test']):
        inputs, labels  = sample
        num_imgs_batch = len(inputs)

        # wrap them in Variable
        if args.use_gpu:
            inputs = Variable(inputs.cuda()).float()
            labels = Variable(labels.cuda()).long()
        else:
            inputs = Variable(inputs).float()
            labels = Variable(labels).long()

        outputs = model(inputs) # outputs can be tensor, tuple or list based on model we choose

        _, preds = torch.max(outputs.data, 1)

        # various metrics
        correct_meter.update(float(torch.sum(preds == labels.data)) / preds[0].numel(), num_imgs_batch)

        # obtain prediction and label for segment and bound respectively
        if args.bound_out:  # bound output
            regs = probmap2bound(F.softmax(outputs, 1), n_workers=args.num_workers, thres=0.7, kernel_size=7)
            preds_bound_np, labels_bound_np = preds.cpu().numpy(), labels.data.cpu().numpy()
            regs_bound_np = regs.data.cpu().numpy()

            preds_seg_np = np.stack([innerouterbound2mask(r, args.output_channel) for r in regs_bound_np])
            labels_seg_np = np.stack([innerouterbound2mask(label, args.output_channel) for label in labels_bound_np])

        else:  # segment output
            preds_seg_np, labels_seg_np = preds.cpu().numpy(), labels.data.cpu().numpy()
            preds_bound_np = np.stack([mask2innerouterbound(pred) for pred in preds_seg_np])
            labels_bound_np = np.stack([mask2innerouterbound(label) for label in labels_seg_np])
            regs_bound_np = preds_bound_np

        # calculate hd95 and asd
        mean_predhdf, batch_hdf = volumewise_hd95(preds_bound_np, labels_bound_np, return_slicewise_hdf=True)
        mean_reghdf = volumewise_hd95(regs_bound_np, labels_bound_np, return_slicewise_hdf=False)
        mean_asd = volumewise_asd(regs_bound_np, labels_bound_np, n_classes=3)
        predhdf_meter.update(mean_predhdf * num_imgs_batch, num_imgs_batch)
        reghdf_meter.update(mean_reghdf * num_imgs_batch, num_imgs_batch)
        asd_meter.update(mean_asd * num_imgs_batch, num_imgs_batch)

        # calculate class-wise F1
        cal_f1 = cal_f_score_slicewise if args.model_type == '3d' else cal_f_score
        _, f_scores, n_effect_samples = cal_f1(preds_seg_np, labels_seg_np, n_class=args.output_channel,
                                               return_slice_f1=False, return_class_f1=True)
        for i in range(args.output_channel):
            fscores_meter[i].update(f_scores[i], n_effect_samples[i])

        running_f1_class = np.array([m.avg for m in fscores_meter])

    # print various metrics
    print("Acc: {:.4f} Ave_F1: {:.4f} Ave_predhdf: {:.4f}, Ave_reghdf: {:.4f}. Ave_ASD: {:.4f}".format(
        correct_meter.avg, running_f1_class.mean(), predhdf_meter.avg, reghdf_meter.avg, asd_meter.avg))

    for c_inx, each_f1 in enumerate(running_f1_class):
        print("Class-{}: F1-{:.4f}".format(c_inx, each_f1))

    ##########################################################################################
    # plot the prediction results
    ##########################################################################################
    if args.do_plot:
        plot_data = args.plot_data
        args.compose[plot_data] = args.compose['test']

        if args.model_type == '2d':
            from image.dataloader import read_plot_data
            dataloaders = read_plot_data(args.data_dir, args.compose, plot_data, False, args.num_workers,
                                         args.batch_size, args.multi_view, args.config)
        else:  # parameters of dataloader for 2.5D and 3D is the same
            if args.model_type == '3d':
                from volume.dataloader import read_plot_data
            elif args.model_type == '2.5d':
                from hybrid.dataloader import read_plot_data

            dataloaders = read_plot_data(args.data_dir, args.compose, plot_data, args.multi_view, args.interval,
                                         args.down_sample, args.num_workers, False, args.config)

        for samp_inx, sample in enumerate(dataloaders[plot_data]):
            inputs_batch, labels, sample_name, start = sample
            sample_name, start = sample_name[0], start.item()
            inputs_batch = torch.squeeze(inputs_batch, dim=0)  # [N, 1, T, H, W]
            labels = torch.squeeze(labels, dim=0)  # [N, T, H, W]
            num_imgs_batch = len(inputs_batch)

            # process each mini-batch
            for mb_inx in range(0, num_imgs_batch, args.batch_size):
                end = min(mb_inx + args.batch_size, num_imgs_batch)
                inputs =inputs_batch[mb_inx:end]

                # wrap them in Variable
                if args.use_gpu:
                    inputs = Variable(inputs.cuda()).float()
                else:
                    inputs = Variable(inputs).float()

                outputs = model(inputs)
                outputs_mb_np = F.softmax(outputs, dim=1).data.cpu().numpy() # don't forget the softmax here

                _, preds = torch.max(outputs.data, 1)
                preds_mb_np = preds.cpu().numpy()

                if mb_inx == 0:
                    preds_np = np.zeros((num_imgs_batch, *(preds_mb_np[0].shape)), dtype=preds_mb_np.dtype)
                    # outputs_np shape: [N * C * T * H * W] or [N * C * H * W]
                    outputs_np = np.zeros((num_imgs_batch, *(outputs_mb_np[0].shape)), dtype=outputs_mb_np.dtype)

                preds_np[mb_inx:end], outputs_np[mb_inx:end] = preds_mb_np, outputs_mb_np

            # convert into numpy
            labels_np = labels.cpu().numpy()
            if inputs_batch.size(1) == 1:  # only one channel
                inputs_np = torch.squeeze(inputs_batch, dim=1).cpu().numpy() # [N, T, H, W]
            else: # if 3 channels, only select the first channel
                inputs_np = inputs_batch[:, 0].cpu().numpy()

            if args.model_type == '2.5d':
                n_slices = inputs_np.shape[1]
                inputs_np = inputs_np[:, n_slices//2]

            # for 2D images, we can directly use it for plot, for 3D volume, transform is necessary
            if args.model_type == '3d':
                inputs_np, labels_np, preds_np, outputs_np = rearrange_volume(
                    inputs_np, labels_np, preds_np, outputs_np, args)

            if args.model_type == '2.5d': # shift start index if 2.5D model
                start += (args.interval // 2) * args.down_sample
            # save predictions into pickle and plot the results
            plot_save_result(args.bound_out, labels_np, inputs_np, preds_np, outputs_np, start, sample_name, args.fig_dir,
                               args.sample_stack_rows, args.output_channel)


def rearrange_volume(inputs, labels, preds, outputs, args):
    """ rearrange volumes into the correct order
    :param inputs: list of ndarrays (N, D, H, W)
    :param labels: ndarray (N, D, H, W)
    :param preds: ndarray (N, D, H, W)
    :param outputs: ndarray (N, C, D, H, W)
    :return:
    """
    inputs = np.reshape(inputs, (-1, *(inputs.shape[2:])))
    labels = np.reshape(labels, (-1, *(labels.shape[2:])))
    preds = np.reshape(preds, (-1, *(preds.shape[2:])))
    outputs = outputs.transpose(0, 2, 1, 3, 4) # [N, C, T, H, W] -- > [N, T, C, H, W]
    outputs = np.reshape(outputs, (-1, *(outputs.shape[2:])))

    num_slices = len(inputs)
    indexes = []
    args.stride = args.down_sample * args.interval
    for s_inx in range(0, num_slices, args.stride):
        for i in range(args.interval):
            for j in range(args.down_sample):
                inx = s_inx + i + j * args.interval
                if inx < num_slices:
                    indexes.append(inx)

    inputs, labels, preds, outputs = inputs[indexes], labels[indexes], preds[indexes], outputs[indexes]

    return (inputs, labels, preds, outputs)


def plot_save_result(is_bound_out, labels, inputs, preds, outputs, start, samp_art_name, root_fig_dir, sample_stack_rows, n_class):
    """ save seg/bound results into pickle and plot the prediction """

    fig_dir = root_fig_dir + '/' + samp_art_name
    if not osp.exists(fig_dir):
        os.makedirs(fig_dir)

    data = {'input': inputs, 'label': labels, 'pred': preds, 'output': outputs,
            'sample_name': samp_art_name, 'start': start, 'n_class': n_class}

    with open(osp.join(fig_dir, 'data.pkl'), 'wb') as writer:
        pickle.dump(data, writer, protocol=pickle.HIGHEST_PROTOCOL)

    # plot the inputs, ground truth, outputs and F1 scores with sample_stack2
    for inx in range(0, len(inputs), sample_stack_rows):
        over = min(inx + sample_stack_rows, len(inputs))
        label_plot, input_plot, pred_plot, output_plot = labels[inx:over], inputs[inx:over], \
                                        preds[inx:over], outputs[inx:over]

        data_list = [{"input": input, "GT": label, "pred": pred, "output": output}
                     for (input, label, pred, output) in zip(input_plot, label_plot, pred_plot, output_plot)]

        file_name = "{}/{:03d}".format(fig_dir, inx + start)
        sample_results(data_list, is_bound_out, rows=over - inx, start_with=0, show_every=1, fig_name=file_name,
                             start_inx=inx + start, n_class=n_class)