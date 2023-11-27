from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
from lib.visualizers import make_visualizer
import torch.multiprocessing
import numpy as np
import ipdb
import torch

def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    import sys
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import matplotlib.mlab as mlab
    from scipy.stats import norm

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    id=0
    # for batch in tqdm.tqdm(data_loader):
    for batch in data_loader:
        # print("batch['inp']:",batch['inp'].shape)
        img=batch['inp'].numpy().squeeze(0)
        img = img[7]
        # print(batch["path"])
        path= batch["path"][0].split('/')[5]+'+'+batch["path"][0].split('/')[6]+'+'+batch["path"][0].split('/')[9]
        print(path)
        # if batch["path"][0].split('/')[5] != "S218928036_S1ff973cbbf73b8_20181024":
        #     continue
        # sys.exit()
        # print("img_shape:",img.shape)
        # cv2.imwrite("test_inp/{}.png".format(path),img)
        for k in batch:

            if k != 'meta' and k!= 'poly' and k!= 'path':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)

        id=id+1
        print("Test_%d:in length of %d"%(id,len(data_loader)))
        # print("Ok")


        haus,ct_cls,amount,ave_hdf_0,ave_hdf_1,hdf_cal_0,hdf_cal_1,hdf_noncal_0,hdf_noncal_1 = visualizer.visualize(output, batch,id)
    return ave_hdf_0,ave_hdf_1,hdf_cal_0,hdf_cal_1,hdf_noncal_0,hdf_noncal_1
def train(cfg, network):
    epoch = 0
    #
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network,epoch)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir,cfg.pretrain_dir, resume=cfg.resume)
    # set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    hdf0_best = 100
    hdf1_best = 100
    #for general score
    hdf_health_0_best = 100
    hdf_health_1_best = 100
    hdf_cal_0_best = 100
    hdf_cal_1_best = 100
    hdf_noncal_0_best = 100
    hdf_noncal_1_best = 100
    epoch_general_best = 0
    epoch_macro_best =0
    # if epoch_general_best == 0:
    #     hdf0,hdf1,hdf_cal_0,hdf_cal_1,hdf_noncal_0,hdf_noncal_1 =run_visualize()
    #     if hdf0 + hdf1 < hdf0_best + hdf1_best:
    #         hdf0_best = hdf0
    #         hdf1_best = hdf1
    #         epoch_general_best = epoch
    #     if hdf0+hdf1+hdf_cal_0+hdf_cal_1+hdf_noncal_0+hdf_noncal_1 < hdf_health_0_best+hdf_health_1_best+hdf_cal_0_best+hdf_cal_1_best+hdf_noncal_0_best+hdf_noncal_1_best:
    #         hdf_health_0_best = hdf0
    #         hdf_health_1_best = hdf1
    #         hdf_cal_0_best = hdf_cal_0
    #         hdf_cal_1_best = hdf_cal_1
    #         hdf_noncal_0_best = hdf_noncal_0
    #         hdf_noncal_1_best = hdf_noncal_1
    #         epoch_macro_best = epoch
    for epoch in range(begin_epoch, cfg.train.epoch):

        # hdf0,hdf1 =run_visualize()
        recorder.epoch = epoch
        optimizer = make_optimizer(cfg, network, epoch)
        # trainer.val(epoch, val_loader, evaluator, recorder)

        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()


        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

        if (epoch + 1) % 25== 0:
            hdf0,hdf1,hdf_cal_0,hdf_cal_1,hdf_noncal_0,hdf_noncal_1 =run_visualize()
            if hdf0 + hdf1 < hdf0_best + hdf1_best:
                hdf0_best = hdf0
                hdf1_best = hdf1
                epoch_general_best = epoch
            if hdf0+hdf1+hdf_cal_0+hdf_cal_1+hdf_noncal_0+hdf_noncal_1 < hdf_health_0_best+hdf_health_1_best+hdf_cal_0_best+hdf_cal_1_best+hdf_noncal_0_best+hdf_noncal_1_best:
                hdf_health_0_best = hdf0
                hdf_health_1_best = hdf1
                hdf_cal_0_best = hdf_cal_0
                hdf_cal_1_best = hdf_cal_1
                hdf_noncal_0_best = hdf_noncal_0
                hdf_noncal_1_best = hdf_noncal_1
                epoch_macro_best = epoch




        print("hdf0:",hdf0_best)
        print("hdf1:",hdf1_best)
        print("epoch_best:",epoch_general_best)
        print("hdf_health_0_best:",hdf_health_0_best)
        print("hdf_health_1_best:",hdf_health_1_best)
        print("hdf_cal_0_best:",hdf_cal_0_best)
        print("hdf_cal_1_best:",hdf_cal_1_best)
        print("hdf_noncal_0_best:",hdf_noncal_0_best)
        print("hdf_noncal_1_best:",hdf_noncal_1_best)
        print("epoch_macro_best:",epoch_macro_best)


        # if (epoch + 1) % cfg.eval_ep == 0:
        #     trainer.val(epoch, val_loader, evaluator, recorder)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def main():
    # torch.set_default_dtype(torch.double)
    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)


if __name__ == "__main__":
    main()
