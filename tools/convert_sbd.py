
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import pickle
from tqdm import tqdm
import math
from scipy.spatial import distance
from skimage import measure
import scipy.io as scio
import pycocotools.mask as mask_util
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os.path as osp
from os import listdir
from lib.config import cfg, args
from torchvision import transforms
from volume.transforms import CentralCrop, Rescale, Gray2Mask, ToTensor, Identical, HU2Gray, RandomFlip, Gray2Triple
from volume.transforms import RandomTranslate, RandomCentralCrop, AddNoise, RandomRotation, Gray2InnerOuterBound
import tifffile


import sys
from termcolor import colored
class Annotation():
    """ dataloader of train and validation dataset. """

    def __init__(self,txt_dir, config_num, metric_prev_epoch = None, phases_prev_epoch = None, transform = None, mode = 'train',
                is_hard_mining = False, percentile = 85, multi_view = False, interval=32, down_sample=1, config='config',split='snake'):
        """ read images from img_dir and save them into a list
        Args:
            data_dir: string, from where to read image
            transform: transform, what transforms to operate on input images
            interval: int, interval of sub-volume
            n_samples_art: int, how many samples to extract per artery
            hard_mining: bool, whether use bad mining or not
            metric_prev_epoch: numpy ndarray, metric obtained from the previous epoch
            phases_prev_epoch: list, phases of the previous epoch
            multi_view: whether use multi_view input or not
            interval: int, how many slices in one batch volume
            down_sample: int, down sampling rate (every how many slices)
        """
        self.data_dir = "/data/ugui0/ruan/CPR_multiview_interp2_huang_full"
        # self.data_dir = "/data/ugui0/antonio-t/BOUND/cpr_data/"
        # self.txt_dir = "configs/config"
        self.txt_dir=txt_dir
        self.err_length = 0
        self.config_num=config_num
        # self.label_dir = os.path.join(self.data_dir, "annotations")
        self.label_dir = os.path.join("/data/ugui0/ruan/CPR_multiview_interp2_huang_full", "annotations")
        self.interval = interval
        self.mode = mode
        self.split=split
        self.transform = transform
        self.down_sample = down_sample
        self.is_hard_mining = is_hard_mining
        self.percentile = percentile
        self.multi_view = multi_view  # whether to use multi-view inputs or not
        self.slice_range = self.interval * self.down_sample
        self.config = config
        self.categories = [
    {'supercategory': 'none', 'id': 1, 'name': 'innerbound'},
    {'supercategory': 'none', 'id': 2, 'name': 'outerbound'},
]

        # initialize phases for different modes
        if self.mode == 'train':
            self.phases = self.update_phases(metric_prev_epoch, phases_prev_epoch)
        else:
            self.phases = self.get_phases()

    def is_tiff_valid(self,filename):
        try:
            with tifffile.TiffFile(filename) as tif:
                return True
        except tifffile.TiffFileError:
            return False

    def update_phases(self, metric_prev_epoch, phases_prev_epoch):
        """ update the phases by mining the bad samples
        :return: phases: refined phases after mining the bad samples
        """
        if phases_prev_epoch is None:
            phases = self.get_phases()

        else:
            if self.is_hard_mining:
                thres = np.percentile(metric_prev_epoch, self.percentile)
                phases = [phase for phase, metric in zip(phases_prev_epoch, metric_prev_epoch) if metric <= thres]
            else:
                phases = phases_prev_epoch

        return phases

    def get_phases(self):
        phases = []
        with open(osp.join(self.txt_dir, self.mode + '.txt'), 'r') as reader:
            samples = [line.strip('\n') for line in reader.readlines()]

        for sample in samples:
            sample_path = osp.join(self.data_dir, sample)
            # print("sample_path:",sample_path)
            # if not os.path.isdir(sample_path):
            #     print(colored("Not Exist:{}".format(sample_path),"red"))
            #     continue


            for artery in sorted(listdir(sample_path)):
                # artery_path = osp.join(sample_path, artery)
                # print("artery:",artery)

                image_path = osp.join(sample_path, artery, 'ordinate', 'image')
                mask_path = osp.join(sample_path, artery, 'ordinate', 'mask')
                # if not os.path.isdir(image_path):
                #     print(colored("sample:{}".format(sample), "red"))
                #     print(colored("artery:{}".format(artery), "red"))
                #     sys.exit()
                # extract slice files
                slice_files = sorted(
                    [file for file in listdir(image_path) if file.endswith('.tiff') and not file.startswith('.')])
                if not len(slice_files):
                    # print(colored("Broken:{}".format(sample_path),"red"))

                    continue


                start_file, end_file = slice_files[0], slice_files[-1]
                start, end = int(start_file.split('.')[0]), int(end_file.split('.')[0])
                for s_inx in range(start, end + 1 - self.slice_range + self.down_sample):
                    phases.append((image_path, mask_path, s_inx))
            # print("Ok for ",sample_path)

        print("{} : {} samples".format(self.mode, len(phases)))
        # sys.exit()

        return phases

    def __getitem__(self, inx):
        sample = self.phases[inx]
        image_path, mask_path, rand_inx  = sample

        if self.multi_view:
            axis_names = ['applicate', 'abscissa', 'ordinate']
        else:
            axis_names = ['applicate']

        for a_inx, axis_name in enumerate(axis_names):
            image_path_axis = image_path.replace('ordinate', axis_name)
            mask_path_axis = mask_path.replace('ordinate', axis_name)

            slice_files_axis = [osp.join(image_path_axis, "{:03d}.tiff".format(i))
                           for i in range(rand_inx, rand_inx + self.slice_range, self.down_sample)]

            label_files_axis = [osp.join(mask_path_axis, "{:03d}.tiff".format(i))
                           for i in range(rand_inx, rand_inx + self.slice_range, self.down_sample)]

            image_axis = np.stack([io.imread(slice_file) for slice_file in slice_files_axis])
            mask_axis = np.stack([io.imread(label_file) for label_file in label_files_axis])

            if axis_name == 'applicate':
                new_d, new_h, new_w = image_axis.shape
                image = np.zeros((*image_axis.shape, len(axis_names)), dtype=np.int16)
                image[:, :, :, a_inx] = image_axis
                mask = mask_axis

            else:
                # if slice size doesn't match with each other, resize them into the same as applicate slice
                for s_inx in range(new_d):
                    slice_axis = image_axis[s_inx]
                    if slice_axis.shape != (new_h, new_w):
                        slice_axis = transform.resize(slice_axis, (new_h, new_w), mode='reflect',
                                                  preserve_range=True).astype(np.int16)
                    image[s_inx, :, :, a_inx] = slice_axis

        # transform 3D image and mask
        # here we have to define volume transform for image and 2D transform for mask
        sample_img, sample_mask = self.transform((image, mask))
        sample_central_mask = sample_mask[self.interval//2]


        return sample_img, sample_central_mask




    def read_txt(self):
        txt_path = os.path.join(self.txt_dir, "{}.txt".format(self.mode))
        with open(txt_path) as f:
            patients  = [line.strip('\n') for line in f.readlines()]

        return patients

    def convert_labels(self,patients):
        images = []
        annotations = []
        label_save_dir = self.label_dir
        count = 0
        countimg=0
        phases=[]
        for patient in patients:
            patient_path = osp.join(self.data_dir, patient)
            for artery in sorted(listdir(patient_path)):
                image_path = osp.join(patient_path, artery, 'ordinate', 'image')
                mask_path = osp.join(patient_path, artery, 'ordinate', 'mask')
                slice_files = sorted(
                    [file for file in listdir(image_path) if file.endswith('.tiff') and not file.startswith('.')])
                if not len(slice_files):
                    # print(colored("Broken:{}".format(sample_path),"red"))

                    continue

                start_file, end_file = slice_files[0], slice_files[-1]
                start, end = int(start_file.split('.')[0]), int(end_file.split('.')[0])
                for s_inx in range(start, end + 1 - self.slice_range + self.down_sample):
                    phases.append((image_path, mask_path, s_inx))

        # self.generate_anno(phases, images, annotations, count, countimg)
        images, annotations, count = self.generate_anno(phases, images, annotations, count,countimg)
        voc_instance = {'images': images, 'annotations': annotations, 'categories': self.categories}
        self.save_annotations(voc_instance, label_save_dir)

    def generate_anno(self,phases, images_info, annotations, count,countimg, N=128):

        for (image_path, mask_path, rand_inx) in phases:
            if self.multi_view:
                axis_names = ['applicate', 'abscissa', 'ordinate']
            else:
                axis_names = ['applicate']
            countimg +=1

            for a_inx, axis_name in enumerate(axis_names):
                image_path_axis = image_path.replace('ordinate', axis_name)
                mask_path_axis = mask_path.replace('ordinate', axis_name)
                slice_files_axis = [osp.join(image_path_axis, "{:03d}.tiff".format(i))
                                    for i in range(rand_inx, rand_inx + self.slice_range, self.down_sample)if  self.is_tiff_valid(os.path.join(image_path_axis, "{:03d}.tiff".format(i)))]

                label_files_axis = [osp.join(mask_path_axis, "{:03d}.tiff".format(i))
                                    for i in range(rand_inx, rand_inx + self.slice_range, self.down_sample)if  self.is_tiff_valid(os.path.join(mask_path_axis, "{:03d}.tiff".format(i)))]
                if len(slice_files_axis) != self.slice_range//self.down_sample or len(label_files_axis) != self.slice_range//self.down_sample:
                    self.err_length += 1
                    print("err_length:{}".format(self.err_length))
                    continue


                image_axis = np.stack([io.imread(slice_file) for slice_file in slice_files_axis])
                mask_axis = np.stack([io.imread(label_file) for label_file in label_files_axis])

                if axis_name == 'applicate':
                    new_d, new_h, new_w = mask_axis.shape
                    image = np.zeros((*image_axis.shape, len(axis_names)), dtype=np.int16)
                    image[:, :, :, a_inx] = image_axis
                    mask = mask_axis


                else:
                    # if slice size doesn't match with each other, resize them into the same as applicate slice
                    for s_inx in range(new_d):
                        slice_axis = image_axis[s_inx]
                        if slice_axis.shape != (new_h, new_w):
                            slice_axis = transform.resize(slice_axis, (new_h, new_w), mode='reflect',
                                                          preserve_range=True).astype(np.int16)
                        image[s_inx, :, :, a_inx] = slice_axis

                # transform 3D image and mask
                # here we have to define volume transform for image and 2D transform for mask
            sample_img, sample_mask,mask_seg ,_= self.transform((image, mask,mask,mask))
            sample_central_mask = sample_mask[self.interval // 2]
            # seg_cls_name = os.path.join(self.data_dir, 'cls', patient_path.split('/')[-1])
            # seg_cls_mat = scio.loadmat(seg_cls_name)
            # semantic_mask = seg_cls_mat['GTcls']['Segmentation'][0][0]
            #
            # seg_obj_mat = scio.loadmat(patient_path)
            instance_mask = sample_central_mask
            # np.savetxt("insmask.txt", instance_mask,fmt='%d ')

            instance_ids = np.unique(instance_mask)
            mask_seg_ids = np.unique(mask_seg[self.interval // 2])
            # if 4 not in mask_seg_ids:
            #     print("images_info_len:",len(images_info))
            #     continue
            # print("Exist 4")


            # img_name = image_path.split('/')[-4] + '/' + image_path.split('/')[-3] + "/{:03d}".format(rand_inx)
            img_name = image_path+ "/{:03d}".format(rand_inx)

            imw = instance_mask.shape[1]
            imh = instance_mask.shape[0]
            # has_object = False
            for instance_id in instance_ids:
                if instance_id == 0 or instance_id == 3 or instance_id == 4:  # background or edge, pass
                    continue
                # extract instance

                # temp = np.ones(instance_mask.shape)
                #   # semantic category of this instance
                #
                # temp[instance_mask == instance_id] = 0
                # self.fill(temp,(0,0),0)
                # instance = temp

                temp = np.zeros(instance_mask.shape)
                # semantic category of this instance

                temp[instance_mask == instance_id] = 1
                self.fill(temp, (0, 0), 2)
                temp_f = np.ones(instance_mask.shape)
                temp_f[temp == 2] = 0
                # cv2.imwrite("temp_f.png",temp_f*255)
                instance = temp_f
                # print(instance_id)
                # np.savetxt("mask_fill.txt",instance,fmt='%d ')
                # print("finish save")
                # with open('mask.txt', 'w') as outfile:
                #     # I'm writing a header here just for the sake of readability
                #     # Any line starting with "#" will be ignored by numpy.loadtxt
                #     outfile.write('# Array shape: {0}\n'.format(instance.shape))
                #
                #     # Iterating througaskh a ndimensional array produces slices along
                #     # the last axis. This is equivalent to data[i,:,:] in this case
                #     for data_slice in instance:
                #         # The formatting string indicates that I'm writing out
                #         # the values in left-justified columns 7 characters in width
                #         # with 2 decimal places.
                #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
                #
                #         # Writing out a break to indicate different slices...
                #         outfile.write('# New slice\n')
                # print("finish mask")
                instance_temp = instance.copy()  # findContours will change instance, so copy first
                # if self.mode == 'mask':
                #     rle = mask_util.encode(np.array(instance, order='F'))
                #     rle['counts'] = rle['counts'].decode('utf-8')
                #
                #     x, y, w, h = cv.boundingRect(instance_temp.astype(np.uint8))
                #     # area = int(np.sum(tempMask))
                #     area = w*h
                #     has_object = True
                #     count += 1
                #     anno = {'segmentation': rle, 'area': area,
                #             'image_id': rand_inx, 'bbox': [x, y, w, h],
                #             'iscrowd': 0, 'category_id': int(instance_id), 'id': count}
                # else:
                area = int(np.sum(instance))

                polys = self.binary_mask_to_polygon(instance)
                if len(polys) == 0:
                    continue

                if len(polys)>1:
                    print("num:",len(polys))
                    print(polys)
                    np.savetxt("check/checksbd.txt",instance_temp,fmt='%d')
                    np.savetxt("check/checkmask.txt", sample_central_mask, fmt='%d')

                x, y, w, h = cv.boundingRect(instance_temp.astype(np.uint8))
                # area = int(np.sum(tempMask))
                area1 = w * h
                print("area:",area,"area1:",area1)
                has_object = True
                count += 1
                print("polys_dim:",len(polys))
                print("instance_id:", instance_id)
                anno = {'segmentation': polys, 'area': area,
                        'image_id': countimg, "interval":self.interval,'bbox': [x, y, w, h],
                        'iscrowd': 0, 'category_id': int(instance_id), 'id': count}

                annotations.append(anno)
            # sample_img=sample_img.tolist()
            # sample_central_mask=sample_central_mask.tolist()

            info = {'file_name': img_name,
                    'height': imh, 'width': imw, 'id': countimg,"interval":self.interval}
            print("count_img:", countimg,"file_name:",img_name)
            images_info.append(info)
            print("images_info_len:",len(images_info))
            print("err_len:",self.err_length)
            # return images_info, annotations, count



        return images_info, annotations, count

    def convert_sbd(self):
        # ids_train_noval = read_txt(txt_dir, 'train_noval')
        patients = self.read_txt()
        # ids_val = read_txt(txt_dir, 'val')
        # ids_val5732 = []

        # for id in ids_train+ids_val:
        #     if id not in ids_train_noval:
        #         ids_val5732.append(id)

        self.convert_labels(patients)
        # convert_labels(ids_val5732, 'trainval', 'snake')
        # convert_labels(ids_val5732, 'val', 'mask')def convert_labels(ids, split, mode):

    def close_contour(self,contour):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour
    # def find_start_coords(self,mask,bound_value):



    def fill(self,data, start_coords, fill_value):
        """
        Flood fill algorithm

        Parameters
        ----------
        data : (M, N) ndarray of uint8 type
            Image with flood to be filled. Modified inplace.
        start_coords : tuple
            Length-2 tuple of ints defining (row, col) start coordinates.
        fill_value : int
            Value the flooded area will take after the fill.

        Returns
        -------
        None, ``data`` is modified inplace.
        """
        xsize, ysize = data.shape
        orig_value = data[start_coords[0], start_coords[1]]

        stack = set(((start_coords[0], start_coords[1]),))
        if fill_value == orig_value:
            raise ValueError("Filling region with same value "
                         "already present is unsupported. "
                         "Did you already fill this region?")
        while stack:
            x, y = stack.pop()
            if data[x, y] == orig_value:
                data[x, y] = fill_value
                if x > 0:
                    stack.add((x - 1, y))
                if x < (xsize - 1):
                    stack.add((x + 1, y))
                if y > 0:
                    stack.add((x, y - 1))
                if y < (ysize - 1):
                    stack.add((x, y + 1))

    def binary_mask_to_polygon(self,binary_mask, tolerance=0):
        polygons = []
        # pad mask to close contours of shapes which start and end at an edge
        # print("binary_mask.shape:",binary_mask.shape)
        padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        # contours = measure.find_contours(binary_mask, 0.5)
        # print(contours)
        # with open('contour.txt', 'w') as outfile:
        #     # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     for data_slice in contours:
        #         # The formatting string indicates that I'm writing out
        #         # the values in left-justified columns 7 characters in width
        #         # with 2 decimal places.
        #         np.savetxt(outfile, data_slice, fmt='%s')
        #
        #         # Writing out a break to indicate different slices...
        #         outfile.write('# New slice\n')
        # print("finish countours")

        contours = np.subtract(contours, 1)
        for contour in contours:
            contour = self.close_contour(contour)
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            segmentation = [0 if i < 0 else i for i in segmentation]
            polygons.append(segmentation)
        return polygons

    def save_annotations(self,ann, path):
        os.system('mkdir -p {}'.format(path))
        instance_path = os.path.join(path, "medical_{}_instance_{}.json".format("val" if self.mode == 'test' else self.mode,self.config_num))
        with open(instance_path, 'w') as f:
            json.dump(ann, f)


if __name__ == '__main__':
    metric_prev_epoch = None
    phases_prev_epoch = None
    transform = None
    mode = 'train'
    is_hard_mining = False
    split = 'snake'

    compose = {'train': transforms.Compose([HU2Gray(),
                                                RandomRotation() if args.rotation else Identical(),
                                                RandomFlip() if args.flip else Identical(),
                                                RandomCentralCrop() if args.r_central_crop else CentralCrop(
                                                    args.central_crop),
                                                Rescale((args.rescale)),
                                                RandomTranslate() if args.random_trans else Identical(),
                                                AddNoise() if args.noise else Identical(),
                                                Gray2Mask(),
                                                Gray2InnerOuterBound()
                                                # ToTensor(norm=True)
                                                ]),

                   'test': transforms.Compose([HU2Gray(),
                                               CentralCrop(args.central_crop),
                                               Rescale(args.rescale),
                                               Gray2Mask(),
                                               Gray2InnerOuterBound()
                                               # ToTensor(norm=True)
                                               ])}

    # text_dir = "configs/configs_all/config_all"
    #
    # mode = 'train'
    # print("start config_all")
    # ann = Annotation(text_dir, 1, metric_prev_epoch, phases_prev_epoch, compose[mode], mode,
    #                  is_hard_mining, cfg.percentile, cfg.multi_view, cfg.interval, cfg.down_sample,
    #                  args.config, split)
    # ann.convert_sbd()
    # text_dir = "configs/configs_all/config_all1"
    # mode = 'train'
    # print("start config_all1")
    # ann = Annotation(text_dir, 5, metric_prev_epoch, phases_prev_epoch, compose[mode], mode,
    #                  is_hard_mining, cfg.percentile, cfg.multi_view, cfg.interval, cfg.down_sample,
    #                  args.config, split)
    # ann.convert_sbd()
    # print("finish train")
    # mode = "test"
    # ann = Annotation(text_dir, 5, metric_prev_epoch, phases_prev_epoch, compose[mode], mode,
    #                  is_hard_mining, cfg.percentile, cfg.multi_view, cfg.interval, cfg.down_sample,
    #                  args.config, split)
    # ann.convert_sbd()
    # print("finish test config_all1")
    # sys.exit()

    # text_dir = "configs/configs_all/config_demo"
    # # mode = 'train'
    # # print("start config_all")
    # # ann = Annotation(text_dir,3, metric_prev_epoch, phases_prev_epoch, compose[mode], mode,
    # #                  is_hard_mining, cfg.percentile, cfg.multi_view, cfg.interval, cfg.down_sample,
    # #                  args.config, split)
    # # ann.convert_sbd()
    # # print("finish train")
    # mode = "test"
    # ann = Annotation(text_dir,3, metric_prev_epoch, phases_prev_epoch, compose[mode], mode,
    #                  is_hard_mining, cfg.percentile, cfg.multi_view, cfg.interval, cfg.down_sample,
    #                  args.config, split)
    # ann.convert_sbd()
    # print("finish test config_all")


    # text_dir="configs/configs_all/config_all1"
    #
    # mode = 'train'
    # print("start config_all1")
    # ann=Annotation(text_dir,1, metric_prev_epoch, phases_prev_epoch, compose[mode], mode,
    #                                       is_hard_mining, cfg.percentile, cfg.multi_view, cfg.interval, cfg.down_sample,
    #                                        args.config,split)
    # ann.convert_sbd()
    # print("finish train")
    # mode="test"
    # ann = Annotation(text_dir,1,metric_prev_epoch, phases_prev_epoch, compose[mode], mode,
    #                  is_hard_mining, cfg.percentile, cfg.multi_view, cfg.interval, cfg.down_sample,
    #                  args.config, split)
    # ann.convert_sbd()
    # print("finish test config_all1")
    #
    #
    #
    text_dir = "configs/configs_all/config_all6"
    mode = 'train'
    print("start config_all6")
    ann = Annotation(text_dir,6, metric_prev_epoch, phases_prev_epoch, compose[mode], mode,
                     is_hard_mining, cfg.percentile, cfg.multi_view, cfg.interval, cfg.down_sample,
                     args.config, split)
    ann.convert_sbd()
    print("finish train")
    mode = "test"
    ann = Annotation(text_dir,6, metric_prev_epoch, phases_prev_epoch, compose[mode], mode,
                     is_hard_mining, cfg.percentile, cfg.multi_view, cfg.interval, cfg.down_sample,
                     args.config, split)
    ann.convert_sbd()
    print("finish test config_all6")
    #
    #
    #
    #
    #
    #
    # text_dir = "configs/configs_all/config_all4"
    # mode = 'train'
    # print("start config_all4")
    # ann = Annotation(text_dir,4, metric_prev_epoch, phases_prev_epoch, compose[mode], mode,
    #                  is_hard_mining, cfg.percentile, cfg.multi_view, cfg.interval, cfg.down_sample,
    #                  args.config, split)
    # ann.convert_sbd()
    # print("finish train")
    # mode = "test"
    # ann = Annotation(text_dir,4, metric_prev_epoch, phases_prev_epoch, compose[mode], mode,
    #                  is_hard_mining, cfg.percentile, cfg.multi_view, cfg.interval, cfg.down_sample,
    #                  args.config, split)
    # ann.convert_sbd()
    # print("finish test config_all4")

