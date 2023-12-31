# _*_ coding: utf-8 _*_

""" commonly used functions for data-processing and visualization """

import matplotlib as mpl
mpl.use('Agg')

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')

import numpy as np
from sklearn.preprocessing import label_binarize
from scipy import ndimage
import cv2
from copy import deepcopy

def count_parameters(model):
    """ count number of parameters in a model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rgb2gray(rgb):
    """ convert rgb image to grayscale one """
    if rgb.ndim == 2:
        img_gray = rgb
    elif rgb.ndim == 3:
        img_gray = np.dot(rgb, [0.299, 0.587, 0.114])

    return img_gray.astype(np.uint8)


def denormalize(image, v=182.7666473388672, m=-4.676876544952393):
    """ de-normalize image into original HU range
    where m and v are the mean and std of image with HU value """
    return (image * v + m).astype(np.int16)


def rgb2mask(rgb):
    """ convert rgb image into mask
        red - (255, 0, 0) : low-density plaque --> 4
        black - (0, 0, 0) : background --> 0
        orange - (255, 128, 0) : calcification --> 3
        white - (255, 255, 255) : Border of the artery (small in healthy patients) --> 2
        blue - (0, 0, 255) : inside of the artery --> 1
    """
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[np.all(rgb == [255, 0, 0], axis=2)] = 4
    mask[np.all(rgb == [255, 128, 0], axis=2)] = 3
    mask[np.all(rgb == [255, 255, 255], axis=2)] = 2
    mask[np.all(rgb == [0, 0, 255], axis=2)] = 1

    return mask


def gray2rgb(gray):
    """ convert grayscale rgb for some discrete values """
    h, w = gray.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[gray == 29] = [0, 0, 255]
    rgb[gray == 255] = [255, 255, 255]
    rgb[gray == 151] = [255, 128, 0]
    rgb[gray == 76] = [255, 0, 0]
    rgb[gray == 226] = [255, 255, 0]
    rgb[gray == 150] = [0, 255, 0]

    return rgb


def mask2rgb(mask):
    """ convert mask into RGB image """
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask == 1] = [0, 0, 255]
    rgb[mask == 2] = [255, 255, 255]
    rgb[mask == 3] = [255, 128, 0]
    rgb[mask == 4] = [255, 0, 0]

    return rgb


def mask2gray(mask):
    """ convert mask to gray """
    h, w = mask.shape[:2]
    gray = np.zeros((h, w), dtype=np.uint8)
    gray[mask == 1] = 29
    gray[mask == 2] = 255
    gray[mask == 3] = 151
    gray[mask == 4] = 76

    return gray


def gray2mask(gray):
    """ convert gray-scale image to 2D mask
        red - 76 : low-density plaque --> 4
        black - 0 : background --> 0
        orange - 151 : calcification --> 3
        white - 255 : Border of the artery (small in healthy patients) --> 2
        blue - 29 : inside of the artery --> 1
    """
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[gray == 76] = 4
    mask[gray == 151] = 3
    mask[gray == 255] = 2
    mask[gray == 29] = 1

    return mask


def central_crop(image, patch_size):
    """ centre crop the given image
    Args:
        im: numpy ndarray, input image
        new_size: tuple, new image size
    """
    assert isinstance(patch_size, (int, tuple)), "size must be int or tuple"
    if isinstance(patch_size, int):
        size = (patch_size, patch_size)
    else:
        size = patch_size

    h, w = image.shape[:2]
    assert (h - size[0]) % 2 == 0 and (w - size[1]) % 2 == 0, \
        "new image size must match with the input image size"
    h_low, w_low = (h - size[0]) // 2, (w - size[1]) // 2
    h_high, w_high = (h + size[0]) // 2, (w + size[1]) // 2

    new_image = image[h_low:h_high, w_low:w_high]

    return new_image


def dcm2hu(dcm):
    """ convert dicom image into Hounsfield (HU) value """

    image = dcm.pixel_array
    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
    image = slope * image + intercept
    return np.array(image, dtype=np.int16)


def hu2gray(image, hu_max=1640.0, hu_min=-1024.0):
    scale = float(255) / (hu_max - hu_min)
    image = (image - hu_min) * scale

    return image

def gray2n1p1range(image):
    """ convert grayscale to -1~1 """
    return 2.0 * image / 255.0 - 1.0


def hu2lut(data, window, level):
    lut = np.piecewise(data, [data <= (level - 0.5 - (window - 1) / 2),
                              data > (level - 0.5 + (window - 1) / 2)],
                       [0, 255, lambda data: ((data - (level - 0.5)) / (window - 1) + 0.5) * (255 - 0)])

    return lut.astype(np.float32)


def hu2norm(image, hu_max=1440.0, hu_min=-1024.0):
    """ scale image into (0.0 1.0) """
    scale = 1.0 / (hu_max - hu_min)
    image = (image - hu_min) * scale

    return image

def shuffle_backward(l, order):
    """ shuffle back to original """
    l_out = np.zeros_like(l)
    for i, j in enumerate(order):
        l_out[j] = l[i]
    return l_out


def gray2innerouterbound(gray, width=1):
    """ convert mask annotation into inner and outer boundaries
        where inner and outer boundaries are treated as different classes
    """
    h, w = gray.shape[:2]
    gray_cp = deepcopy(gray)
    gray_cp[gray == 76] = 255
    gray_cp[gray == 151] = 255
    bound = np.zeros_like(gray, dtype=np.uint8)
    label = gray2mask(gray_cp)
    label_binary = label_binarize(label.flatten(), classes=range(0, 3))
    label_binary = np.reshape(label_binary, (h, w, -1))
    bound_binary = np.zeros_like(label_binary)

    for i in range(2):  # number of classes before edge detection
        tmp = ndimage.distance_transform_cdt(label_binary[:, :, i], 'taxicab')
        cdt = np.logical_and(tmp >= 1, tmp <= width)
        bound_binary[:, :, i] = cdt

    bound[bound_binary[:, :, 0] != 0] = 2  # outer bound marked as 2
    bound[bound_binary[:, :, 1] != 0] = 1  # inner bound marked as 1

    return bound

def mask2innerouterbound(mask, width=1):
    """ convert mask annotation into inner and outer boundaries
        where inner and outer boundaries are treated as different classes
    """
    h, w = mask.shape[:2]
    if type(mask) is np.ndarray:
        mask_np = deepcopy(mask)
    else:
        mask_np = mask.clone()

    mask_np[mask == 3] = 2
    mask_np[mask == 4] = 2

    bound = np.zeros_like(mask_np, dtype=np.uint8)
    label_binary = label_binarize(mask_np.flatten(), classes=range(0, 3))
    label_binary = np.reshape(label_binary, (h, w, -1))
    bound_binary = np.zeros_like(label_binary)

    for i in range(3):  # number of classes before edge detection
        tmp = ndimage.distance_transform_cdt(label_binary[:, :, i], 'taxicab')
        cdt = np.logical_and(tmp >= 1, tmp <= width)
        bound_binary[:, :, i] = cdt

    bound[bound_binary[:, :, 0] != 0] = 2  # outer bound marked as 2
    bound[bound_binary[:, :, 1] != 0] = 1  # inner bound marked as 1

    return bound


def innerouterbound2mask(innerouter, n_classes=3):
    """ transform innerouter bound to mask segmentation, cv2.drawContours is used for transformation
    :param innerouter: ndarray of size [H, W], 1 - inner, 2 - outer
    :param n_classes: int, number of classes
    :return: mask: ndarray of size [H, W]
    """
    # only apply to situation with n_classes = 3
    ls = np.zeros(innerouter.shape[:2], np.uint8)
    for c_inx in range(n_classes - 1, 0, -1):
        points = np.array(np.where(innerouter == c_inx)).transpose()  # [N, 2]
        points = np.expand_dims(np.flip(points, axis=1), axis=1)  # [N, 2] --> [N, 1, 2]
        if len(points) > 0:
            ls = cv2.drawContours(ls, [points], -1, c_inx, thickness=cv2.FILLED)

    return ls


def ls2bound(ls, width=1):
    """ convert morphological snake result into boundary """
    tmp = ndimage.distance_transform_cdt(ls, 'taxicab')
    bound = np.logical_and(tmp >= 1, tmp <= width)

    return bound


def lslist2bound(ls_list):
    """ convert ls list to boundary """

    h, w = ls_list[0].shape
    bound = np.zeros((h, w), dtype=np.uint8)
    for inx, ls in enumerate(ls_list):
        b_inx = ls2bound(ls, width=1)
        if b_inx.any():
            bound[b_inx] = inx + 1

    return bound


class AverageMeter(object):
    """ Compute and store the average and current value. """
    def __init__(self):
        self.val = None
        self.avg = None
        self.val_sum = None
        self.cnt_sum = None
        self.reset()  # Reset the values.

    def reset(self):
        self.val = 0
        self.avg = 0
        self.val_sum = 0
        self.cnt_sum = 0


    def update(self, val, cnt):
        self.val = val
        self.val_sum += val
        self.cnt_sum += cnt
        self.avg = float(self.val_sum) / self.cnt_sum

if __name__ == "__main__":
    pass
