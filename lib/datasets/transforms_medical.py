from torchvision import transforms
from volume.transforms import CentralCrop, Rescale, Gray2Mask, ToTensor, Identical, HU2Gray, RandomFlip, Gray2Triple
from volume.transforms import RandomTranslate, RandomCentralCrop, AddNoise, RandomRotation, Gray2InnerOuterBound
from lib.config import cfg, args
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpts=None):
        for t in self.transforms:
            img, kpts = t(img, kpts)
        if kpts is None:
            return img
        else:
            return img, kpts

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

#
# class ToTensor(object):
#     def __call__(self, img, kpts):
#         return img / 255., kpts


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, kpts):
        img -= self.mean
        img /= self.std
        return img, kpts


def make_transforms(args, is_train):
    if is_train is True:

        transform = transforms.Compose([HU2Gray(),
                                        RandomRotation() if args.rotation else Identical(),
                                        RandomFlip() if args.flip else Identical(),
                                        RandomCentralCrop() if args.r_central_crop else CentralCrop(
                                            args.central_crop),
                                        Rescale((args.rescale)),
                                        RandomTranslate() if args.random_trans else Identical(),
                                        AddNoise() if args.noise else Identical(),
                                        Gray2Mask(),
                                        Gray2InnerOuterBound()
                                        ]
        )
    else:

        transform = transforms.Compose([HU2Gray(),
                                        CentralCrop(args.central_crop),
                                        Rescale(args.rescale),
                                        Gray2Mask(),
                                        Gray2InnerOuterBound()
                                        ]
)


    return transform
