# _*_ coding: utf-8 _*_

""" transforms for 3D volume """

import torch
from skimage import transform
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import cv2
from utils import gray2innerouterbound, hu2lut

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


class Gray2InnerOuterBound(object):
    """ convert mask with grayscale value to inner and outer boundaries
        where inner and outer boundaries are treated as different classes
    """
    def __init__(self, width=1):
        self.width = width  # boundary width

    def __call__(self, sample):
        image, gray,gray_seg,mask_dis = sample
        bounds = np.zeros_like(gray)
        for l_inx, label in enumerate(gray):
            bounds[l_inx] = gray2innerouterbound(label, self.width)

        return image, bounds,gray_seg,mask_dis
        # image, gray, mask_seg = sample
        # bounds = np.zeros_like(gray)
        # for l_inx, label in enumerate(gray):
        #     bounds[l_inx] = gray2innerouterbound(label, self.width)
        #
        # return image, bounds, mask_seg


class RandomRotation(object):
    """ random rotation (angle is randomly set as a multiplier of given angle) """
    def __init__(self, angle=90, prob=0.8):
        self.angle = angle
        self.prob = prob

    def __call__(self, sample):
        image, mask,mask_seg,mask_dis = sample
        # print(mask_dis)
        # print("mask_dis_shape:",mask_dis.shape)
        rand_angle = random.randrange(0, 360, self.angle)
        x = random.uniform(0, 1)
        if x <= self.prob:
            for inx,label in enumerate(mask_dis):
                mask_dis[inx] = transform.rotate(label, rand_angle, mode='reflect', preserve_range=True)
            for inx,label in enumerate(mask_seg):
                mask_seg[inx] = transform.rotate(label, rand_angle, mode='reflect', preserve_range=True, order=0)
            for s_inx, (slice, label) in enumerate(zip(image, mask)):
                image[s_inx] = transform.rotate(slice, rand_angle, mode='reflect', preserve_range=True)
                mask[s_inx] = transform.rotate(label, rand_angle, mode='reflect', preserve_range=True, order=0)

        return (image, mask,mask_seg,mask_dis)

class RandomFlip(object):
    """ random horizontal or vertical flip """
    def __init__(self, prob=0.8):
        self.prob = prob

    def __call__(self, sample):
        image, mask,mask_seg,mask_dis = sample
        x = random.uniform(0, 1)
        if x < self.prob:
            phase = random.randint(0, 1)
            for inx,label in enumerate(mask_dis):
                mask_dis[inx] = np.flip(label, phase)
            for inx,label in enumerate(mask_seg):
                mask_seg[inx] = np.flip(label, phase)
            for s_inx, (slice, label) in enumerate(zip(image, mask)):
                image[s_inx] = np.flip(slice, phase)
                mask[s_inx] = np.flip(label, phase)

        return (image, mask,mask_seg,mask_dis)

class Rescale(object):
    """Rescale the image in a sample to a given size with range preserved

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2, "output_size should be a 2-dimensional tuple"
            self.output_size = output_size

    def __call__(self, sample):
        image, mask,mask_seg,mask_dis = sample
        h, w = image.shape[1:3]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        if image.ndim == 3:
            new_image = np.zeros((len(image), new_h, new_w), dtype=image.dtype)
        elif image.ndim == 4:
            new_image = np.zeros((len(image), new_h, new_w, image.shape[3]), dtype=image.dtype)

        new_mask = np.zeros((len(mask), new_h, new_w), dtype=mask.dtype)
        new_mask_seg = np.zeros((len(mask_seg), new_h, new_w), dtype=mask_seg.dtype)
        new_mask_dis = np.zeros((len(mask_dis), new_h, new_w), dtype=mask_dis.dtype)
        for i,label in enumerate(mask_dis):
            # print("i:",i)
            # print(label)
            new_mask_dis[i] = transform.resize(label, (new_h, new_w), mode='edge', preserve_range=True,
                                           anti_aliasing=False)
        for i,label in enumerate(mask_seg):
            # print("i:",i)
            # print(label)
            new_mask_seg[i] = transform.resize(label, (new_h, new_w), mode='edge', preserve_range=True,
                                           anti_aliasing=False)

        for i, (slice, label) in enumerate(zip(image, mask)):
            # new_image[i] = cv2.resize(np.unsqueeze(slice, 2), (new_h, new_w), interpolation=cv2.INTER_NEAREST)
            # new_mask[i] = cv2.resize(np.unsqueeze(label, 2), (new_h, new_w), interpolation=cv2.INTER_NEAREST)
            new_image[i] = transform.resize(slice, (new_h, new_w), mode= 'reflect', preserve_range=True)
            new_mask[i] = transform.resize(label, (new_h, new_w), mode= 'edge', preserve_range=True, order=0, anti_aliasing=False)


        return new_image, new_mask,new_mask_seg,new_mask_dis

class RandomCrop(object):
    """ Randomly crop the image
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample

        h, w = image.shape[1:3]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # print("random crop position: {} {}".format(top, left))
        image = image[:, top:top + new_h, left:left + new_w]
        mask = mask[:, top:top + new_h, left:left + new_w]

        return image, mask

class RGB2Mask(object):
    """ convert 3D rgb annotation to 3D mask
        red - (255, 0, 0) : low-density plaque --> 4
        black - (0, 0, 0) : background --> 0
        orange - (255, 128, 0) : calcification --> 3
        white - (255, 255, 255) : Border of the artery (small in healthy patients) --> 2
        blue - (0, 0, 255) : inside of the artery --> 1
    """
    def __call__(self, sample):
        image, rgb = sample

        d, h, w = rgb.shape[:3]
        mask = np.zeros((d, h, w), dtype=np.uint8)
        mask[np.all(rgb == [255, 0, 0], axis=3)] = 4
        mask[np.all(rgb == [255, 128, 0], axis=3)] = 3
        mask[np.all(rgb == [255, 255, 255], axis=3)] = 2
        mask[np.all(rgb == [0, 0, 255], axis=3)] = 1

        return image, mask


class Gray2Triple(object):
    """ convert gray-scale image to mask which only marks central part, outline and background """
    def __call__(self, sample):
        image, gray = sample
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[gray == 76] = 2
        mask[gray == 151] = 2
        mask[gray == 255] = 2
        mask[gray == 29] = 1

        return image, mask


class Gray2Mask(object):
    """ convert gray-scale image to 2D mask
        red - 76 : low-density plaque --> 4
        black - 0 : background --> 0
        orange - 151 : calcification --> 3
        white - 255 : Border of the artery (small in healthy patients) --> 2
        blue - 29 : inside of the artery --> 1
    """
    def __call__(self, sample):
        # image, gray, mask_dis = sample
        #
        # mask = np.zeros_like(gray, dtype=np.uint8)
        # mask[gray == 76] = 4
        # mask[gray == 151] = 3
        # mask[gray == 255] = 2
        # mask[gray == 29] = 1
        #
        # return image, mask,mask_dis
        image, gray, gray_seg ,mask_dis= sample

        mask = np.zeros_like(gray_seg, dtype=np.uint8)
        mask[gray == 76] = 4
        mask[gray == 151] = 3
        mask[gray == 255] = 2
        mask[gray == 29] = 1

        return image, gray, mask,mask_dis


class Gray2Binary(object):
    """ convert gray-scale image to Binary label mask """
    def __call__(self, sample):
        image, gray = sample
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[gray != 0] = 1

        return image, mask

class HU2Gray(object):
    def __init__(self, hu_max=1440.0, hu_min=-1024.0):
        self.hu_max = hu_max
        self.hu_min = hu_min
        self.scale = float(255) / (self.hu_max - self.hu_min)

    def __call__(self, sample):
        """ convert HU value to gray scale [0, 255]
        hu: numpy ndarray, Image of HU value, [H, W]

        """
        # print("sample:",sample)
        # print("sample_len:", len(sample))
        image, mask, mask_seg, mask_dis = sample
        # image, mask,mask_seg,mask_dis= sample
        # print(mask_dis)
        # print("mask_dis_shape:", mask_dis.shape)
        image =  (image - self.hu_min) * self.scale

        return image, mask,mask_seg,mask_dis

class HU2LUT(object):
    def __init__(self, window, level):
        self.window = window
        self.level = level

    def __call__(self, sample):
        data, mask = sample

        lut = np.piecewise(data, [data <= (self.level - 0.5 - (self.window - 1) / 2),
                           data > (self.level - 0.5 + (self.window - 1) / 2)],
                        [0, 255, lambda data: ((data - (self.level - 0.5)) / (self.window - 1) + 0.5) * (255 - 0)])

        return lut.astype(np.uint8), mask



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, norm=True):
        self.norm = norm

    def __call__(self, sample):
        image, mask = sample
        # swap color axis because (1) numpy image: H x W x C (2) torch image: C X H X W
        if image.ndim == 4:
            image = image.transpose((3, 0, 1, 2))
        elif image.ndim == 3:
            image = image[np.newaxis, :, :, :]

        image = torch.from_numpy(image).float()
        if self.norm:
            image = image / 255.0

        mask = torch.from_numpy(mask).long()

        return image, mask

class HU2GrayMultiStreamToTensor(object):
    """ convert HU value to grayscale for different windows + ToTensor """
    def __init__(self, w_widths = [500.0, 100.0], w_centers = [250.0, 50.0], norm=True):
        self.w_widths = w_widths
        self.w_centers = w_centers
        self.norm = norm

    def __call__(self, sample):
        image, mask = sample
        if image.ndim == 4:
            image = image.transpose((3, 0, 1, 2))
        elif image.ndim == 3:
            image = image[np.newaxis, :, :, :]
        sample_img = []
        for w_w, w_c in zip(self.w_widths, self.w_centers):
            stream = hu2lut(image, w_w, w_c)
            stream = torch.from_numpy(stream).float()
            if self.norm:
                stream = stream / 255.0
            sample_img.append(stream)

        sample_mask = torch.from_numpy(mask).long()

        return sample_img, sample_mask


class Identical(object):

    def __call__(self, sample):
        return sample

class CentralCrop(object):

    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        """ centre crop the given image
        Args:
            sample : (image, mask)
            size: tuple, new image size
        """
        image, mask,mask_seg,mask_dis = sample
        # print("mask_dis_shape_for old_centralcrop:",mask_dis.shape)

        h, w = image.shape[1:3]
        assert (h - self.size[0]) % 2 == 0 and (w - self.size[1]) % 2 == 0, \
            "new image size must match with the input image size"
        h_low, w_low = (h - self.size[0]) // 2, (w - self.size[1]) // 2
        h_high, w_high = (h + self.size[0]) // 2, (w + self.size[1]) // 2

        # print("h_low, w_low :",h_low,"and", w_low )
        # print("h_high, w_high :", h_high,  "and", w_high)

        new_image = image[:, h_low:h_high, w_low:w_high]
        new_mask = mask[:, h_low:h_high, w_low:w_high]
        new_mask_seg = mask_seg[:, h_low:h_high, w_low:w_high]
        # new_mask_dis = mask_dis[:, h_low:h_high, w_low:w_high]
        # print(new_mask_dis)
        # print("mask_dis_shapeincentralcrop:", new_mask_dis.shape)

        return new_image, new_mask,new_mask_seg,mask_dis

class GaussianCrop(object):
    """ crop patches with central pixel position (x, y) obeying Guassian distribution """
    def __init__(self, size, sigma=0.1):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.sigma = sigma

    def __call__(self, sample):
        """ centre crop the given image
        Args:
            sample : (image, mask)
        """
        image, mask = sample
        h, w = image.shape[1:3]
        p_h, p_w = self.size

        c_h = int(random.normalvariate(0.5, self.sigma) * h)
        if c_h < p_h // 2:
            c_h = p_h // 2
        elif c_h > h - p_h // 2:
            c_h = h - p_h // 2

        c_w = int(random.normalvariate(0.5, self.sigma) * w)
        if c_w < p_w // 2 :
            c_w = p_w // 2
        elif c_w > w - p_w // 2:
            c_w = w - p_w // 2

        new_image = image[:, c_h - p_h // 2:c_h + p_h // 2, c_w - p_w // 2:c_w + p_w // 2]
        new_mask = mask[:, c_h - p_h // 2:c_h + p_h // 2, c_w - p_w // 2:c_w + p_w // 2]

        return new_image, new_mask


class RandomCentralCrop(object):
    """ randomly central crop with given size options """
    def __init__(self, lower_size=160, upper_size=256, step=4):
        assert lower_size%2 ==0 and upper_size%2==0, "both lower and upper size should be even number"
        self.lower_size = lower_size
        self.upper_size = upper_size
        self.step = step

    def __call__(self, sample):
        x = random.randrange(self.lower_size, self.upper_size, self.step)
        return CentralCrop(x)(sample)

class AddNoise(object):
    """ add Gaussian noise to given sample """
    def __init__(self, loc=0.0, scale=1.0, prob=0.5):
        self.loc = loc
        self.scale = scale
        self.prob = prob

    def __call__(self, sample):
        image, mask ,mask_seg,mask_dis= sample
        x = random.uniform(0, 1)
        if x <= self.prob:
            noise = np.random.normal(self.loc, self.scale, image.shape)
            image += noise

        return image, mask,mask_seg,mask_dis

class RandomTranslate(object):
    """ random translate the given image """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):

        image, mask,mask_seg,mask_dis = sample
        [H, W] = image.shape[1:3]

        x = random.uniform(0, 1)
        if x <= self.prob:
            right = random.randint(int(-W/4), int(W/4))
            down = random.randint(int(H/4), int(H/4))
            M = np.float32([[1, 0, right], [0, 1, down]])
            for i, label in enumerate(mask_dis):
                mask_dis[i]=cv2.warpAffine(label, M, (W, H))
            for i, label in enumerate(mask_seg):
                mask_seg[i]=cv2.warpAffine(label, M, (W, H))

            for i, (slice, label) in enumerate(zip(image, mask)):
                image[i] = cv2.warpAffine(slice, M, (W, H))
                mask[i] = cv2.warpAffine(label, M, (W, H))

        return image, mask,mask_seg,mask_dis