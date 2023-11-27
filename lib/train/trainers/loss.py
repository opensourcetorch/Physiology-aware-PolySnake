# _*_ coding: utf-8 _*_

""" define custom loss functions """
import sys

import matplotlib as mpl
mpl.use('Agg')

from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch
from torch.autograd import Function
torch.set_default_dtype(torch.float32)
import math
import ipdb

torch.set_default_dtype(torch.float32)

def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"


class DiceLoss(Function):
    """ Normal Dice Loss for multi-class segmentation """
    def __init__(self, weight=None, ignore_index=None, weight_type=None, reduce=True, cal_zerogt=False):
        """
        :param weight: tensor vector, custom weight for each class
        :param ignore_index: int, index of class to ignore
        :param weight_type: str, in which way to assign class weight if 'weight' is not given
        :param reduce: bool, if true, calculate mean loss within given minibatch
        :param cal_zerogt: bool, if true, calculate Dice for case of all GT pixels are zero
        """
        self.weight = weight
        self.ignore_index = ignore_index
        self.weight_type = weight_type
        self.reduce = reduce
        self.cal_zerogt = cal_zerogt

    def __call__(self, output, target):
        # output : N x C x *,  Variable of float tensor (* means any dimensions)
        # target : N x *, Variable of long tensor (* means any dimensions)
        # weights : C, float tensor
        # ignore_index : int, class index to ignore from loss (0 for background)
        smooth = 1.0e-9
        output = F.softmax(output, dim=1)
        n_classes = output.size(1)

        if output.size() != target.size():
            target = target.data
            encoded_target = output.data.clone().zero_()  # make output size array and initialize with zeros

            if self.ignore_index is not None:
                mask = (target == self.ignore_index)
                target = target.clone()
                target[mask] = 0
                encoded_target.scatter_(1, target.unsqueeze(1), 1)
                mask = mask.unsqueeze(1).expand_as(encoded_target)
                encoded_target[mask] = 0
            else:
                unseq = target.long()
                encoded_target.scatter_(1, unseq.unsqueeze(1), 1)

            encoded_target = Variable(encoded_target, requires_grad = False)

        else:
            encoded_target = target

        # calculate gt, t and p from perspective of 1
        intersection = output * encoded_target
        numerator = 2 * torch.sum(intersection.view(*intersection.size()[:2], -1), 2) + smooth
        denominator1 = torch.sum(output.view(*output.size()[:2], -1), 2)
        denominator2 = torch.sum(encoded_target.view(*encoded_target.size()[:2], -1), 2)
        denominator = denominator1 + denominator2 + smooth
        mask_gt = (denominator2 == 0)

        if self.weight is None:
            if self.weight_type is None:
                weight =  Variable(torch.ones(n_classes).cuda(), requires_grad=False)
            else:
                tmp = denominator2.sum(0)
                tmp =  tmp / tmp.sum()

                if self.weight_type == 'nlf':
                    weight = -1.0 * torch.log(tmp + smooth)

                elif self.weight_type == 'mfb':
                    weight = torch.median(tmp) / (tmp + smooth)

            weight = weight.detach()

        else: # prior weight is setting manually
            weight = self.weight

        loss_per_channel = weight * (1.0 - (numerator / denominator))

        # calculate Dice for special case of all GT pixels are zero
        if self.cal_zerogt:
            output_com = 1.0 - output
            encoded_target_com = 1 - encoded_target
            intersection_com = output_com * encoded_target_com
            numerator_com = 2 * torch.sum(intersection_com.view(*intersection_com.size()[:2], -1), 2) + smooth
            denominator1_com = torch.sum(output_com.view(*output_com.size()[:2], -1), 2)
            denominator2_com = torch.sum(encoded_target_com.view(*encoded_target_com.size()[:2], -1), 2)
            denominator_com = denominator1_com + denominator2_com + smooth
            loss_per_channel_com = weight * (1.0 - (numerator_com / denominator_com))

        loss_per_channel = loss_per_channel.clone()
        # if all GT pixels are zero, use the complementary pixels for calculation
        if self.cal_zerogt:
            loss_per_channel[mask_gt] = loss_per_channel_com[mask_gt]
        else:
            loss_per_channel[mask_gt] = 0

        if self.reduce:
            if self.cal_zerogt:
                dice_loss = loss_per_channel.mean()
            else:
                loss_ave_class = loss_per_channel.sum() / (mask_gt == 0).sum(1).float()
                dice_loss = loss_ave_class.mean()
        else:
            dice_loss = loss_per_channel  # [N, C]

        return dice_loss


class WeightedKLDivLoss(Function):
    """ weighted KL-divergence loss for batch with imbalanced class distribution
        correctness has already been checked
    """

    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        self.size_average = size_average

    def __call__(self, output, target):
        """ forward propagation
        :param output: Variable of output [N x C x *]
        :param target: Variable of GT prob [N x C x *]
        """
        smooth = 1.0e-9
        output = F.log_softmax(output, 1)
        kl_div = target * (torch.log(target + smooth) - output)
        kl_div = kl_div.permute(0, *range(2, len(kl_div.size())), 1)

        if self.weight is not None:
            kl_div = self.weight * kl_div

        if self.size_average:
            loss = kl_div.mean()
        else:
            loss = kl_div.sum()

        return loss


class GeneralizedDiceLoss(Function):
    """ generalized dice score for multi-class segmentation defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
            loss function for highly unbalanced segmentations. DLMIA 2017
        weight for each class is calculated from the distribution of Ground
        Truth pixel/voxel belonging to each class
    """

    def __init__(self, weight=None, ignore_index=None, weight_type='inv_square',
                 alpha=0.5):
        """
        :param weight_type: str, in which way to calculate class weight
            here we consider 3 strategies for deciding proper class weight
            (1) inv_square class weight is set as inverse of summing up square of
                ground truth for each class
            (2) others_one_pred: class weight is set as others_over_one ratio of predicted
                probabilities for each class
            (3) others_one_gt: class weight is set as others_over_one ratio of ground truth
                for each class
        :param alpha: float, ratio of false positive
        :param beta: float, ratio of false negative
            increase alpha if you care more about false positive and beta otherwise
        """
        self.ignore_index = ignore_index
        self.weight = weight
        self.weight_type = weight_type
        self.alpha= alpha
        self.beta = 1- self.alpha

    def __call__(self, output, target):
        # output : N x C x *,  Variable of float tensor (* means any dimensions)
        # target : N x *, Variable of long tensor (* means any dimensions)
        # weights : C, float tensor
        # ignore_index : int, class index to ignore from loss (0 for background)
        # back propagation is checked to be correct

        smooth = 1.0e-9
        # print("diceloss_shape:",output.shape)
        # output = F.softmax(output, 1)
        n_pixels = output[:, 0].numel()
        n_classes = output.size(1)
        # print("output.size():",output.size())
        # print("target.size():",target.size())

        if output.size() != target.size():
            # for normal input
            target = target.data
            encoded_target = output.data.clone().zero_()  # make output size array and initialize with zeros

            if self.ignore_index is not None:
                mask = (target == self.ignore_index)
                target = target.clone()
                target[mask] = 0
                encoded_target.scatter_(1, target.unsqueeze(1), 1)
                mask = mask.unsqueeze(1).expand_as(encoded_target)
                encoded_target[mask] = 0
            else:
                unseq = target.long()
                encoded_target.scatter_(1, unseq.unsqueeze(1), 1)

            encoded_target = Variable(encoded_target, requires_grad=False)


        else:
            # for BC learning input
            encoded_target = target

        tp = output * encoded_target
        fp = output * (1-encoded_target)
        fn = (1.0 - output) * encoded_target
        # print("encoded_target.shape:",encoded_target.shape)

        # add along all dimensions except the first (n_batch) and the second (n_class) dim

        gt_sum = torch.sum(encoded_target.view(*encoded_target.size()[:2], -1), 2).sum(0)
        # print("gt_sum.shape:",gt_sum.shape)

        mask_gt = (gt_sum == 0)
        tp_sum = torch.sum(tp.view(*tp.size()[:2], -1), 2).sum(0)
        fp_sum = torch.sum(fp.view(*fp.size()[:2], -1), 2).sum(0)
        fn_sum = torch.sum(fn.view(*fn.size()[:2], -1), 2).sum(0)

        numerator = tp_sum
        denominator = tp_sum + self.alpha * fp_sum + self.beta * fn_sum
        # print("encoded_target.shape:",encoded_target.shape)

        if self.weight is None:
            if self.weight_type is None:
                weight =  Variable(torch.ones(n_classes).cuda(), requires_grad=False)
            else:
                if self.weight_type == 'inv_square':
                    weight = 1.0 / (gt_sum.pow(2) + smooth)
                    weight[gt_sum==0] = 0.0
                elif self.weight_type == 'others_one_pred':
                    prob_sum_per_class = torch.sum(output.view(*output.size()[:2], -1), 2).sum(0)
                    weight = (n_pixels - prob_sum_per_class) / prob_sum_per_class
                elif self.weight_type == 'others_one_gt':
                    weight = (n_pixels - gt_sum) / (gt_sum + smooth)
                    weight[gt_sum==0] = 0.0

            weight = weight.detach()

        else:
            weight = self.weight
        # print("gt_sum:",gt_sum)
        # print("weight:",weight)
        # print("numerator:",numerator)
        # print("denominator:",denominator)
        # print("(weight * numerator).sum():",(weight * numerator).sum())
        # print("(weight * denominator).sum():",(weight * denominator).sum())
        # print("(weight * numerator).sum() / (weight * denominator).sum()",(weight * numerator).sum() / (weight * denominator).sum())


        loss = 1.0 - (weight * numerator).sum() / (weight * denominator).sum()
        # print("loss:",loss)

        return loss

class WeightedCrossEntropy(CrossEntropyLoss):
    """ weighted cross entropy for multi-class semantic segmentation defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    """
    def __init__(self, weight=None, ignore_index=-100, reduction="mean", weight_type='log_inv_freq'):
        super(WeightedCrossEntropy, self).__init__(weight, ignore_index=ignore_index, reduction=reduction)
        self.weight_type = weight_type


    def forward(self, output, target):
        """ weighted cross entropy where weight is calculated from input data
        :param output: Variable, N x C x *, probabilities for each class
        :param target: Variable, N x *, GT labels
        """
        if self.weight is None:
            output_prob = F.softmax(output, 1)
            prob_per_class = torch.sum(output_prob.view(*output_prob.size()[:2], -1), 2).sum(0)
            prob_sum = prob_per_class.sum()

            if self.weight_type == 'others_over_one':
                weight = (prob_sum - prob_per_class)/prob_per_class
            elif self.weight_type == 'log_inv_freq':
                weight = torch.log(prob_sum/prob_per_class)

            weight = weight.detach()
        else:
            weight = self.weight

        return F.cross_entropy(output, target, weight=weight, ignore_index=self.ignore_index, reduction=self.reduction)


class FocalLoss(Function):
    """ focal loss for multi-class object detection defined in
    Tsung-Yi Lin et. al. Focal Loss for Dense Object Detection. CVPR 2017
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, output, target):
        """ weighted cross entropy where weight is calculated from input data
        :param output: Variable, N x C x *, probabilities for each class
        :param target: Variable, N x *, GT labels
        """
        encoded_target = output.data.clone().zero_()
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        encoded_target = Variable(encoded_target, requires_grad=False)

        prob = output.sigmoid()
        pt = prob*encoded_target + (1-prob)*(1-encoded_target)         # pt = p if t > 0 else 1-p
        # w is given for every element
        weight = self.alpha*encoded_target + (1-self.alpha)*(1-encoded_target)  # w = alpha if t > 0 else 1-alpha
        weight = weight * (1-pt).pow(self.gamma)

        # since loss decay very fast, sum is returned instead of average
        return F.binary_cross_entropy_with_logits(output, encoded_target, weight, size_average=False)


############### weighted Hausdorff distance loss ###############
#   probability of each class is included so that the loss can be back-propagated

def cdist(x, y):
    ''' :param x: Tensor of size Nxd
    :param y: Tensor of size Mxd
    :return dist: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:]
                  i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances
class WeightedHausdorffDistanceLoss(Function):
    """ weighted HausdorffDistanceLoss defined in "Weighted Hausdorff Distance: A Loss Function For Object
        Localization". Javier Ribera. (for single class boundary)
        for more information, please refer to the original paper.
    """
    def __init__(self, return_2_terms=False, alpha=4, beta=2):
        """
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param resized_height: int, height after resize
        :param resized_width: int, width after resize
        :param alpha: int, decay factor
        """
        self.return_2_terms = return_2_terms
        self.alpha = alpha
        self.beta = beta

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x H x W) or (B x D x H x W), Tensor of the probability map of the estimation.
        :param gt: (B x H x W) or (B x D x H x W), Tensor of the GT annotation
        """

        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)
        # assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        if prob_map.dim() == 4 and gt.dim() == 4:
            prob_map = prob_map.contiguous().view(prob_map.size(0) * prob_map.size(1), *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(gt.size(0) * gt.size(1), *gt.size()[2:])                         # [B*D, H, W]

        batch_size, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same size'

        max_dist = math.sqrt(height ** 2 + width ** 2)
        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(), torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        terms_1 = []
        terms_2 = []
        for b in range(batch_size):
            prob_map_b, gt_b = prob_map[b], gt[b]
            gt_pts = torch.nonzero(gt_b).float() # N, 2
            n_gt_pts = gt_pts.size()[0]

            if n_gt_pts > 0:
                d_matrix = cdist(all_img_locations, gt_pts)
                p = prob_map_b.view(prob_map_b.numel())

                n_est_pts = (p**beta).sum()
                p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                # term 1
                term_1 = (1 / (n_est_pts + eps)) * torch.sum(p**beta * torch.min(d_matrix, 1)[0])
                d_div_p = torch.min((d_matrix + eps) /
                                    (p_replicated**alpha + eps / max_dist), 0)[0]
                d_div_p = torch.clamp(d_div_p, 0, max_dist)
                term_2 = torch.mean(d_div_p)

                terms_1.append(term_1)
                terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        # check both terms_1 and terms_2 are not None
        res =[]
        if terms_1 is not None and terms_2 is not None:
            if self.return_2_terms:
                res = terms_1.mean(), terms_2.mean()
            else:
                res = terms_1.mean() + terms_2.mean()

        return res


class WeightedHausdorffDistanceLoss(Function):
    """ weighted HausdorffDistanceLoss defined in "Weighted Hausdorff Distance: A Loss Function For Object
        Localization". Javier Ribera. (for single class boundary)
        for more information, please refer to the original paper.
    """
    def __init__(self, return_2_terms=False, alpha=4, beta=2):
        """
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param resized_height: int, height after resize
        :param resized_width: int, width after resize
        :param alpha: int, decay factor
        """
        self.return_2_terms = return_2_terms
        self.alpha = alpha
        self.beta = beta

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x H x W) or (B x D x H x W), Tensor of the probability map of the estimation.
        :param gt: (B x H x W) or (B x D x H x W), Tensor of the GT annotation
        """

        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)
        # assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        if prob_map.dim() == 4 and gt.dim() == 4:
            prob_map = prob_map.contiguous().view(prob_map.size(0) * prob_map.size(1), *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(gt.size(0) * gt.size(1), *gt.size()[2:])                         # [B*D, H, W]

        batch_size, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same size'

        max_dist = math.sqrt(height ** 2 + width ** 2)
        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(), torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        terms_1 = []
        terms_2 = []
        for b in range(batch_size):
            prob_map_b, gt_b = prob_map[b], gt[b]
            gt_pts = torch.nonzero(gt_b).float() # N, 2
            n_gt_pts = gt_pts.size()[0]

            if n_gt_pts > 0:
                d_matrix = cdist(all_img_locations, gt_pts)
                p = prob_map_b.view(prob_map_b.numel())

                n_est_pts = (p**beta).sum()
                p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                # term 1
                term_1 = (1 / (n_est_pts + eps)) * torch.sum(p**beta * torch.min(d_matrix, 1)[0])
                d_div_p = torch.min((d_matrix + eps) /
                                    (p_replicated**alpha + eps / max_dist), 0)[0]
                d_div_p = torch.clamp(d_div_p, 0, max_dist)
                term_2 = torch.mean(d_div_p)

                terms_1.append(term_1)
                terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        # check both terms_1 and terms_2 are not None
        res =[]
        if terms_1 is not None and terms_2 is not None:
            if self.return_2_terms:
                res = terms_1.mean(), terms_2.mean()
            else:
                res = terms_1.mean() + terms_2.mean()

        return res





# class WeightedHausdorffDistanceDoubleBoundLoss(Function):
#     def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5):
#         """ whd loss for inner and outer bound separately
#             inner bound -- 1, outer bound -- 2
#         :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
#         :param alpha: int, decay factor
#         :param ratio: float, ratio of inner bound, default is 0.5
#         """
#         self.return_boundwise_loss = return_boundwise_loss
#         self.alpha = alpha
#         self.beta = beta
#         self.ratio = ratio
#
#     def __call__(self, prob_map, gt):
#         """ Compute the Weighted Hausdorff Distance function between the estimated probability map
#         and ground truth points. The output is the WHD averaged through all the batch.
#         :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
#         :param gt: (B x H x W) Tensor of the GT annotation
#         """
#         eps = 1e-6
#         alpha = self.alpha
#         beta = self.beta
#         _assert_no_grad(gt)
#
#         # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
#         # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
#         if prob_map.dim() == 5: # 3D volume
#             prob_map = prob_map.permute(0, 2, 1, 3, 4)
#             prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
#             gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims
#
#         batch_size, n_channel, height, width = prob_map.size()
#         assert batch_size == len(gt), 'prob map and GT must have the same length'
#
#         max_dist = math.sqrt(height ** 2 + width ** 2)
#
#         n_pixels = height * width
#         all_img_locations = torch.meshgrid([torch.arange(height).cuda(),
#                                             torch.arange(width).cuda()])
#         all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2
#
#         # print(height, width, all_img_locations.size())
#         # here we consider inner bound and outer bound respectively
#         res_bounds_lst = [[] for _ in range(0, n_channel)]
#
#         for b in range(batch_size):
#
#             prob_map_b, gt_b = prob_map[b], gt[b]
#
#             for bound_inx in range(1, n_channel):
#                 # gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
#                 gt_bb = gt_b[bound_inx]
#                 gt_pts = torch.nonzero(gt_bb).float()  # N, 2
#                 n_gt_pts = gt_pts.size()[0]
#                 prob_map_bb = prob_map_b[bound_inx]
#                 if n_gt_pts > 0:
#                     d_matrix = cdist(all_img_locations, gt_pts)
#                     p = prob_map_bb.view(prob_map_bb.numel())
#
#                     n_est_pts = (p**beta).sum()
#                     p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)
#
#                     term_1 = (1 / (n_est_pts + eps)) * torch.sum(p**beta * torch.min(d_matrix, 1)[0])
#                     d_div_p = torch.min((d_matrix + eps) /
#                                         (p_replicated**alpha + eps / max_dist), 0)[0]
#                     d_div_p = torch.clamp(d_div_p, 0, max_dist)
#                     term_2 = torch.mean(d_div_p)
#                     # set different ratio for inner and outer bound
#                     res_bounds_lst[bound_inx].append(term_1 + term_2)
#
#         res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel)]
#         res_bounds_mean = [res_bound.mean() for res_bound in res_bounds]
#         res_boundwise = torch.stack(res_bounds_mean)  # convert list into torch array
#         # ratio: inner bound ratio
#         res = res_boundwise[0] * self.ratio + res_boundwise[1] * (1.0 - self.ratio)
#
#         if self.return_boundwise_loss:  # return inner bound loss and outer bound loss respectively
#             return res, res_boundwise
#         else:
#             return res
class WeightedHausdorffDistanceDoubleBoundLoss(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5):
        """ whd loss for inner and outer bound separately
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5
        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)

        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same length'

        max_dist = math.sqrt(height ** 2 + width ** 2)

        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(),
                                            torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # print(height, width, all_img_locations.size())
        # here we consider inner bound and outer bound respectively
        res_bounds_lst = [[] for _ in range(0, n_channel)]
        term1_lst = [[] for _ in range(0, n_channel)]
        term2_lst = [[] for _ in range(0, n_channel)]

        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())

                    n_est_pts = (p**beta).sum()
                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)
                    # ipdb.set_trace()

                    term_1 = (1 / (n_est_pts + eps)) * torch.sum(p**beta * torch.min(d_matrix, 1)[0])
                    d_div_p = torch.min((d_matrix + eps) /(p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)
                    term_2 = torch.mean(d_div_p)
                    # set different ratio for inner and outer bound
                    res_bounds_lst[bound_inx].append(term_1 + term_2)
                    term1_lst[bound_inx].append(term_1)
                    term2_lst[bound_inx].append(term_2)

        res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel) if len(res_bounds_lst[i]) > 0]
        res_bounds_mean = [res_bound.mean() for res_bound in res_bounds if len(res_bound) > 0]
        term1 = [torch.stack(term1_lst[i]) for i in range(1, n_channel) if len(term1_lst[i]) > 0]
        term1_mean = [term.mean() for term in term1 if len(term) > 0]
        term2 = [torch.stack(term2_lst[i]) for i in range(1, n_channel) if len(term2_lst[i]) > 0]
        term2_mean = [term.mean() for term in term2 if len(term) > 0]
        if len(res_bounds_mean) == 0:
            return torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        else:
            res_boundwise = torch.stack(res_bounds_mean)
            res = res_boundwise.mean()
            term1 = torch.stack(term1_mean).mean()
            term2 = torch.stack(term2_mean).mean()
            # ipdb.set_trace()
            return res,term1,term2

class ShapeFocalWeightedHausdorffDistanceDoubleBoundLoss_tanh(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5,sigma_1=2,sigma_2=2):
        """ whd loss for inner and outer bound separately
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5
        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio
        self.sigma = torch.stack([torch.tensor(sigma_1), torch.tensor(sigma_2)]).cuda().to(torch.float32)

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)

        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same length'

        max_dist = math.sqrt(height ** 2 + width ** 2)

        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(),
                                            torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # print(height, width, all_img_locations.size())
        # here we consider inner bound and outer bound respectively
        res_bounds_lst = [[] for _ in range(0, n_channel)]
        term1_lst = [[] for _ in range(0, n_channel)]
        term2_lst = [[] for _ in range(0, n_channel)]

        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())

                    n_est_pts = (p**beta).sum()
                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)
                    # ipdb.set_trace()

                    term_1_old = (1 / (n_est_pts + eps)) * torch.sum(p**beta * torch.min(d_matrix, 1)[0])
                    term_1 = term_1_old
                    d_div_p = torch.min((d_matrix + eps) /(p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)
                    term_2 = torch.mean(d_div_p)/torch.tanh(term_1_old)**self.sigma[bound_inx - 1]
                    # set different ratio for inner and outer bound
                    res_bounds_lst[bound_inx].append(term_1 + term_2)
                    term1_lst[bound_inx].append(term_1)
                    term2_lst[bound_inx].append(term_2)

        res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel) if len(res_bounds_lst[i]) > 0]
        res_bounds_mean = [res_bound.mean() for res_bound in res_bounds if len(res_bound) > 0]
        term1 = [torch.stack(term1_lst[i]) for i in range(1, n_channel) if len(term1_lst[i]) > 0]
        term1_mean = [term.mean() for term in term1 if len(term) > 0]
        term2 = [torch.stack(term2_lst[i]) for i in range(1, n_channel) if len(term2_lst[i]) > 0]
        term2_mean = [term.mean() for term in term2 if len(term) > 0]
        if len(res_bounds_mean) == 0:
            return torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        else:
            res_boundwise = torch.stack(res_bounds_mean)
            res = res_boundwise.mean()
            term1 = torch.stack(term1_mean).mean()
            term2 = torch.stack(term2_mean).mean()
            # ipdb.set_trace()
            return res,term1,term2

class FocalWeightedHausdorffDistanceDoubleBoundLoss_tanh_v3(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5,sigma_1=2,sigma_2=2,gamma_1 = 2,gamma_2 = 1e-2):
        """ whd loss for inner and outer bound separately
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5
        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio
        self.sigma = [sigma_1,sigma_2]
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        sigma =torch.stack([torch.tensor(self.sigma[0]).cuda().to(torch.float32),torch.tensor(self.sigma[1]).cuda().to(torch.float32)])
        _assert_no_grad(gt)

        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same length'

        max_dist = math.sqrt(height ** 2 + width ** 2)

        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(),
                                            torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # print(height, width, all_img_locations.size())
        # here we consider inner bound and outer bound respectively
        res_bounds_lst = [[] for _ in range(0, n_channel)]
        focal_res_bounds_lst = [[] for _ in range(0, n_channel)]
        term1_lst = [[] for _ in range(0, n_channel)]
        term2_lst = [[] for _ in range(0, n_channel)]

        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())
                    # rp_term_1 = (torch.tanh((p**beta)* torch.min(d_matrix, 1)[0]**2/self.gamma_1))**sigma[bound_inx-1]
                    # rp_term_1 = (torch.tanh((p**beta)* torch.min(d_matrix, 1)[0]**2/4))

                    # n_est_pts = (rp_term_1*p**beta).sum()
                    n_est_pts = (p**beta).sum()
                    # focal_n_est_pts = (rp_term_1*p**beta).sum()
                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)
                    # ipdb.set_trace()

                    # term_1 = (1 / (n_est_pts + eps)) * torch.sum(rp_term_1*p**beta * torch.min(d_matrix, 1)[0])
                    term_1 = (1 / (n_est_pts + eps)) * torch.sum(p**beta * torch.min(d_matrix, 1)[0])
                    # focal_term_1 = (1 / (focal_n_est_pts + eps)) * torch.sum(rp_term_1*p**beta * torch.min(d_matrix, 1)[0])
                    d_div_p = torch.min((d_matrix + eps) /(p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)
                    # rp_term_2 = (torch.tanh(d_div_p/self.gamma_2))**sigma[bound_inx-1]
                    # rp_term_2 = (torch.tanh(d_div_p/self.gamma_2))**0.0
                    # term_2 = (rp_term_2*d_div_p).sum()/rp_term_2.sum()
                    term_2 =torch.mean(d_div_p)
                    # focal_term_2 = (rp_term_2*d_div_p).sum()/rp_term_2.sum()
                    # set different ratio for inner and outer bound
                    res_bounds_lst[bound_inx].append(term_1 + term_2)
                    # focal_res_bounds_lst[bound_inx].append(focal_term_1 + focal_term_2)
                    term1_lst[bound_inx].append(term_1)
                    term2_lst[bound_inx].append(term_2)
                    # ipdb.set_trace()


        term1 = [torch.stack(term1_lst[i]) for i in range(1, n_channel) if len(term1_lst[i]) > 0]
        r_term1 = [torch.tanh(2*term1[i])**1 for i in range(len(term1))]
        term1_mean = [(r_term1[i]*term1[i]).sum()/r_term1[i].sum() for i in range(len(term1))]
        term2 = [torch.stack(term2_lst[i]) for i in range(1, n_channel) if len(term2_lst[i]) > 0]
        # r_term2 = [torch.tanh((torch.log10(term2[i])+6)/3)*2 for i in range(len(term2))]
        r_term2 = [torch.tanh(term2[i]*20000)**1 for i in range(len(term2))]
        term2_mean = [(r_term2[i]*term2[i]).sum()/r_term2[i].sum()  for i in range(len(term2))]
        res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel) if len(res_bounds_lst[i]) > 0]
        # focal_res_bounds = [torch.stack(focal_res_bounds_lst[i]) for i in range(1, n_channel) if len(focal_res_bounds_lst[i]) > 0]
        # r_res_bounds_div = [(1-torch.sigmoid((term1[i]-0.6)/0.001)) for i in range(len(res_bounds))]
        # r_res_bounds_div = [torch.clamp(r_res_bounds_div[i],1e-3, 1).detach() for i in range(len(res_bounds))]
        #
        r_res_bounds = [torch.tanh(res_bounds[i])**2 for i in range(len(res_bounds))]
        # #
        res_bounds_mean = [(r_res_bounds[i]*res_bounds[i]).mean() for i in range(len(res_bounds))]
        # res_bounds_mean = [(term1_mean[i]+term2_mean[i]) for i in range(len(res_bounds))]
        # ipdb.set_trace()
        if len(res_bounds_mean) == 0:
            return torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        else:
            res_boundwise = torch.stack(res_bounds_mean)
            res = res_boundwise.mean()
            term1_f = torch.stack(term1_mean).mean()
            term2_f = torch.stack(term2_mean).mean()
            # ipdb.set_trace()
            return term1_f+term2_f,term1_f,term2_f


class FocalWeightedHausdorffDistanceDoubleBoundLoss(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5,sigma_1=1.5,sigma_2=1.8):
        """ whd loss for inner and outer bound separately
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5
        :param sigma_1: float, sigma for inner bound
        :param sigma_2: float, sigma for outer bound
        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio
        self.sigma = torch.stack([torch.tensor(sigma_1), torch.tensor(sigma_2)]).cuda().to(torch.float32)

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)

        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same length'

        max_dist = math.sqrt(height ** 2 + width ** 2)

        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(),
                                            torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # print(height, width, all_img_locations.size())
        # here we consider inner bound and outer bound respectively
        res_bounds_lst = [[] for _ in range(0, n_channel)]
        term1_lst = [[] for _ in range(0, n_channel)]
        term2_lst = [[] for _ in range(0, n_channel)]
        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())

                    # n_est_pts = (p**beta).sum()
                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)


                    term_1_max =  torch.max((p**beta) * torch.min(d_matrix, 1)[0])
                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    term_2_max = torch.max(torch.clamp(d_div_p, 0, max_dist))


                    # set different ratio for inner and outer bound
                    term1_lst[bound_inx].append(term_1_max)
                    term2_lst[bound_inx].append(term_2_max)
        term_1s_list = [torch.stack(term1_lst[i]) for i in range(1, n_channel) if len(term1_lst[i]) > 0]
        term_2s_list = [torch.stack(term2_lst[i]) for i in range(1, n_channel) if len(term2_lst[i]) > 0]

        term_1s_max = torch.stack([torch.max(term_1s_list[i]) for i in range(len(term_1s_list))]).detach()
        term_2s_max = torch.stack([torch.max(term_2s_list[i]) for i in range(len(term_2s_list))]).detach()


        assert len(term_1s_max) == len(term_2s_max) == n_channel - 1
        # ipdb.set_trace()

        n_est_pts = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        n_est_gt_pts = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        term_1 = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        term_2 = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())


                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                    rp_term_1 = ((p**beta)* torch.min(d_matrix, 1)[0]/term_1s_max[bound_inx-1])**self.sigma[bound_inx-1]
                    # ipdb.set_trace()

                    # ratio matrix for term1 with probability map size
                    n_est_pts[bound_inx - 1] += (rp_term_1 * p**beta).sum()
                    term_1[bound_inx - 1] += torch.sum(rp_term_1*p**beta * torch.min(d_matrix, 1)[0])
                    # ipdb.set_trace()



                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)

                    rp_term_2 = (d_div_p/term_2s_max[bound_inx-1])**self.sigma[bound_inx-1]
                    # ratio matrix for term2 with probability map size

                    n_est_gt_pts[bound_inx - 1] += rp_term_2.sum()



                    term_2[bound_inx - 1] += (rp_term_2 * d_div_p).sum()
                    # set different ratio for inner and outer bound
                    # res_bounds_lst[bound_inx].append(term_1 + term_2)
                    # ipdb.set_trace()
        # ipdb.set_trace()
        term_1_f = torch.mean(term_1/n_est_pts)
        term_2_f = torch.mean(term_2/n_est_gt_pts)

        return term_1_f + term_2_f






class FocalWeightedHausdorffDistanceDoubleBoundLoss_tanh_v2(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5,sigma_1=2,sigma_2=2,gamma_1 = 0.5,gamma_2 = 1e-3):
        """ whd loss for inner and outer bound separately
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5
        :param sigma_1: float, sigma for inner bound
        :param sigma_2: float, sigma for outer bound
        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio
        self.sigma = torch.stack([torch.tensor(sigma_1), torch.tensor(sigma_2)]).cuda().to(torch.float32)
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)

        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same length'

        max_dist = math.sqrt(height ** 2 + width ** 2)

        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(),
                                            torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # print(height, width, all_img_locations.size())
        # here we consider inner bound and outer bound respectively
        res_bounds_lst = [[] for _ in range(0, n_channel)]

        # ipdb.set_trace()

        n_est_pts = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        n_est_gt_pts = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        term_1 = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        term_2 = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())


                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                    rp_term_1 = (torch.tanh((p**beta)* torch.min(d_matrix, 1)[0]/self.gamma_1))**self.sigma[bound_inx-1]
                    # ipdb.set_trace()

                    # ratio matrix for term1 with probability map size
                    n_est_pts[bound_inx - 1] += (rp_term_1 * p**beta).sum()
                    term_1[bound_inx - 1] += torch.sum(rp_term_1*p**beta * torch.min(d_matrix, 1)[0])
                    # ipdb.set_trace()



                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)

                    rp_term_2 = (torch.tanh(d_div_p/self.gamma_2))**self.sigma[bound_inx-1]
                    # ratio matrix for term2 with probability map size

                    n_est_gt_pts[bound_inx - 1] += rp_term_2.sum()



                    term_2[bound_inx - 1] += (rp_term_2*d_div_p).sum()
                    # set different ratio for inner and outer bound
                    # res_bounds_lst[bound_inx].append(term_1 + term_2)
                    # ipdb.set_trace()
        # ipdb.set_trace()
        term_1_f = torch.mean(term_1/n_est_pts)
        term_2_f = torch.mean(term_2/n_est_gt_pts)

        return term_1_f + term_2_f, term_1_f, term_2_f

class FocalSmoothL1Loss(Function):
    def __init__(self, beta=1.0, alpha=0.2,gamma=2.0, reduction='mean'):
        """focal smooth l1 loss for poly prediction
        :param beta: float, beta for smooth l1 loss
        :param alpha: float, alpha for smooth l1 loss
        :param reduction: str, reduction method, default is 'mean'

        """
        super(FocalSmoothL1Loss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def __call__(self, pred, target):
        smthl1_loss = F.smooth_l1_loss(pred, target, reduction='none', beta=self.beta)
        focal_smtl1_loss = torch.tanh(smthl1_loss / self.alpha)**self.gamma * smthl1_loss
        # ipdb.set_trace()
        if self.reduction == 'mean':
            return focal_smtl1_loss.mean()
        elif self.reduction == 'sum':
            return focal_smtl1_loss.sum()
        else:
            return focal_smtl1_loss

class FocalWeightedHausdorffDistanceDoubleBoundLoss_1b2s(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5,sigma_1=1.3,sigma_2=1.7):
        """ whd loss for inner and outer bound separately
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5
        :param sigma_1: float, sigma for inner bound
        :param sigma_2: float, sigma for outer bound
        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio
        self.sigma = torch.stack([torch.tensor(sigma_1), torch.tensor(sigma_2)]).cuda().to(torch.float32)

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)

        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same length'

        max_dist = math.sqrt(height ** 2 + width ** 2)

        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(),
                                            torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # print(height, width, all_img_locations.size())
        # here we consider inner bound and outer bound respectively
        res_bounds_lst = [[] for _ in range(0, n_channel)]
        term1_lst = [[] for _ in range(0, n_channel)]
        term2_lst = [[] for _ in range(0, n_channel)]
        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())

                    # n_est_pts = (p**beta).sum()
                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)


                    term_1_max =  torch.max((p**beta) * torch.min(d_matrix, 1)[0])
                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    term_2_max = torch.max(torch.clamp(d_div_p, 0, max_dist))

                    # set different ratio for inner and outer bound
                    term1_lst[bound_inx].append(term_1_max)
                    term2_lst[bound_inx].append(term_2_max)
        term_1s_list = [torch.stack(term1_lst[i]) for i in range(1, n_channel) if len(term1_lst[i]) > 0]
        term_2s_list = [torch.stack(term2_lst[i]) for i in range(1, n_channel) if len(term2_lst[i]) > 0]

        term_1s_max = torch.stack([torch.max(term_1s_list[i]) for i in range(len(term_1s_list))]).detach()
        term_2s_max = torch.stack([torch.max(term_2s_list[i]) for i in range(len(term_2s_list))]).detach()


        assert len(term_1s_max) == len(term_2s_max) == n_channel - 1
        # ipdb.set_trace()

        n_est_pts = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        n_est_gt_pts = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        term_1 = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        term_2 = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())


                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                    rp_term_1 = ((p**beta)* torch.min(d_matrix, 1)[0]/term_1s_max[bound_inx-1])**self.sigma[bound_inx-1]
                    # ipdb.set_trace()

                    # ratio matrix for term1 with probability map size
                    n_est_pts[bound_inx - 1] += (rp_term_1 * p**beta).sum()
                    term_1[bound_inx - 1] += torch.sum(rp_term_1*p**beta * torch.min(d_matrix, 1)[0])
                    # ipdb.set_trace()



                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)

                    rp_term_2 = (d_div_p/term_2s_max[bound_inx-1])**self.sigma[bound_inx-1]
                    # ratio matrix for term2 with probability map size

                    n_est_gt_pts[bound_inx - 1] += rp_term_2.sum()



                    term_2[bound_inx - 1] += (rp_term_2 * d_div_p).sum()
                    # set different ratio for inner and outer bound
                    # res_bounds_lst[bound_inx].append(term_1 + term_2)
                    # ipdb.set_trace()
        # ipdb.set_trace()
        term_1_f = torch.mean(term_1/n_est_pts)
        term_2_f = torch.mean(term_2/n_est_gt_pts)

        return term_1_f + term_2_f

class FocalWeightedHausdorffDistanceDoubleBoundLoss_tanh(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5,sigma_1=2,sigma_2=2,gamma =0.4):
        """ whd loss for inner and outer bound separately
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5
        :param sigma_1: float, sigma for inner bound
        :param sigma_2: float, sigma for outer bound
        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio
        self.sigma = torch.stack([torch.tensor(sigma_1), torch.tensor(sigma_2)]).cuda().to(torch.float32)
        self.gamma = gamma

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)

        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same length'

        max_dist = math.sqrt(height ** 2 + width ** 2)

        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(),
                                            torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # print(height, width, all_img_locations.size())
        # here we consider inner bound and outer bound respectively
        res_bounds_lst = [[] for _ in range(0, n_channel)]
        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())


                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                    rp_term_1 =  (torch.tanh((p**beta)* torch.min(d_matrix, 1)[0]/self.gamma))**self.sigma[bound_inx-1]
                    # ipdb.set_trace()

                    # ratio matrix for term1 with probability map size
                    n_est_pts= (p**beta).sum()
                    term_1= (1 / (n_est_pts + eps))*torch.sum(rp_term_1*p**beta * torch.min(d_matrix, 1)[0])
                    # ipdb.set_trace()



                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)

                    rp_term_2 = (torch.tanh(d_div_p/self.gamma))**self.sigma[bound_inx-1]
                    # ratio matrix for term2 with probability map size
                    term_2= torch.mean(rp_term_2 * d_div_p)
                    res_bounds_lst[bound_inx].append(term_1 + term_2)
                    # set different ratio for inner and outer bound
                    # res_bounds_lst[bound_inx].append(term_1 + term_2)
                    ipdb.set_trace()
        # ipdb.set_trace()
        res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel) if len(res_bounds_lst[i]) > 0]
        res_bounds_mean = [res_bound.mean() for res_bound in res_bounds if len(res_bound) > 0]
        if len(res_bounds_mean) == 0:
            return torch.tensor(0.0).cuda()
        else:
            res_boundwise = torch.stack(res_bounds_mean)
            res = res_boundwise.mean()
            # ipdb.set_trace()
            return res
        # res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel) if len(res_bounds_lst[i]) > 0]
        # res_bounds_mean = [res_bound.mean() for res_bound in res_bounds if len(res_bound) > 0]
        # if len(res_bounds_mean) == 0:
        #     return torch.tensor(0.0).cuda()
        # else:
        #     res_boundwise = torch.stack(res_bounds_mean)
        #     res = res_boundwise.mean()
        #     # ipdb.set_trace()
        #     return res

class FocalWeightedHausdorffDistanceDoubleBoundLoss_v2(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5,sigma_1=0.5,sigma_2=0.5):
        """ whd loss for inner and outer bound separately
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5
        :param sigma_1: float, sigma for inner bound
        :param sigma_2: float, sigma for outer bound
        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio
        self.sigma = torch.stack([torch.tensor(sigma_1), torch.tensor(sigma_2)]).cuda().to(torch.float32)

    def __call__(self, prob_map, gt):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)

        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same length'

        max_dist = math.sqrt(height ** 2 + width ** 2)

        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(),
                                            torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # print(height, width, all_img_locations.size())
        # here we consider inner bound and outer bound respectively
        res_bounds_lst = [[] for _ in range(0, n_channel)]
        term1_lst = [[] for _ in range(0, n_channel)]
        term2_lst = [[] for _ in range(0, n_channel)]
        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())

                    # n_est_pts = (p**beta).sum()
                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)


                    term_1_max =  torch.max((p**beta) * torch.min(d_matrix, 1)[0])
                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    term_2_max = torch.max(torch.clamp(d_div_p, 0, max_dist))

                    # set different ratio for inner and outer bound
                    term1_lst[bound_inx].append(term_1_max)
                    term2_lst[bound_inx].append(term_2_max)
        term_1s_list = [torch.stack(term1_lst[i]) for i in range(1, n_channel) if len(term1_lst[i]) > 0]
        term_2s_list = [torch.stack(term2_lst[i]) for i in range(1, n_channel) if len(term2_lst[i]) > 0]

        term_1s_max = torch.stack([torch.max(term_1s_list[i]) for i in range(len(term_1s_list))]).detach()
        term_2s_max = torch.stack([torch.max(term_2s_list[i]) for i in range(len(term_2s_list))]).detach()


        assert len(term_1s_max) == len(term_2s_max) == n_channel - 1
        # ipdb.set_trace()

        # n_est_pts = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        # n_est_gt_pts = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        # term_1 = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        # term_2 = torch.zeros(n_channel - 1).cuda().to(torch.float)+eps
        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())


                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                    rp_term_1 = ((p**beta)* torch.min(d_matrix, 1)[0]/term_1s_max[bound_inx-1])**self.sigma[bound_inx-1]
                    # ipdb.set_trace()

                    # ratio matrix for term1 with probability map size
                    n_est_pts= (p**beta).sum()
                    term_1= (1 / (n_est_pts + eps))*torch.sum(rp_term_1*p**beta * torch.min(d_matrix, 1)[0])
                    # ipdb.set_trace()



                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)

                    rp_term_2 = (d_div_p/term_2s_max[bound_inx-1])**self.sigma[bound_inx-1]
                    # ratio matrix for term2 with probability map size
                    term_2= torch.mean(rp_term_2 * d_div_p)
                    res_bounds_lst[bound_inx].append(term_1 + term_2)
                    # set different ratio for inner and outer bound
                    # res_bounds_lst[bound_inx].append(term_1 + term_2)
                    # ipdb.set_trace()
        # ipdb.set_trace()
        res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel) if len(res_bounds_lst[i]) > 0]
        res_bounds_mean = [res_bound.mean() for res_bound in res_bounds if len(res_bound) > 0]
        if len(res_bounds_mean) == 0:
            return torch.tensor(0.0).cuda()
        else:
            res_boundwise = torch.stack(res_bounds_mean)
            res = res_boundwise.mean()
            # ipdb.set_trace()
            return res
        # res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel) if len(res_bounds_lst[i]) > 0]
        # res_bounds_mean = [res_bound.mean() for res_bound in res_bounds if len(res_bound) > 0]
        # if len(res_bounds_mean) == 0:
        #     return torch.tensor(0.0).cuda()
        # else:
        #     res_boundwise = torch.stack(res_bounds_mean)
        #     res = res_boundwise.mean()
        #     # ipdb.set_trace()
        #     return res

class BRWWeightedHausdorffDistanceDoubleBoundLoss(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5):
        """ Batch Region weighted whd loss for inner and outer bound separately
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5

        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio

    def __call__(self, prob_map, gt,regionweight_map):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        :param regionweight_map: ((H//r x W//r) x H//r x W//r) Tensor of the regionweight_map
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)
        _assert_no_grad(regionweight_map)

        B,C,H,W = prob_map.size()
        _,H_,W_ = regionweight_map.size()
        # ipdb.set_trace()
        region_test = regionweight_map.unsqueeze(0).repeat(1,1,1,W//W_)
        regionweight_map=regionweight_map.unsqueeze(0).unsqueeze(4).repeat(B,1,1,H//H_,W//W_).view(-1,H,W) # (B*9,3*32,3*32)
        # ipdb.set_trace()
        regionweight_map_ = regionweight_map.unsqueeze(1).repeat(1,prob_map.shape[1],1,1) # (B*9,3,3*32,3*32)


        #expand gt H//r * W//r in B chaannel
        gt = gt.unsqueeze(1).repeat(1,H_*W_,1,1).view(-1, H,W) #(B*9,96,96)
        # ipdb.set_trace()
        gt = gt*regionweight_map
        prob_map = prob_map.unsqueeze(1).expand(-1,H_*W_,-1,-1,-1).reshape(-1,C,H,W )#(B*9,3,96,96)
        prob_map = prob_map*regionweight_map_.to(torch.float32)
        # ipdb.set_trace()








        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same length'

        max_dist = math.sqrt(height ** 2 + width ** 2)

        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(),
                                            torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # print(height, width, all_img_locations.size())
        # here we consider inner bound and outer bound respectively
        res_bounds_lst = [[] for _ in range(0, n_channel)]
        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())

                    n_est_pts = (p**beta).sum()
                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                    term_1 = (1 / (n_est_pts + eps)) * torch.sum(p**beta * torch.min(d_matrix, 1)[0])
                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)
                    term_2 = torch.mean(d_div_p)
                    # set different ratio for inner and outer bound
                    res_bounds_lst[bound_inx].append(term_1 + term_2)
                # ipdb.set_trace()

        res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel) if len(res_bounds_lst[i]) > 0]

        selected_losses = []
        for i in range(len(res_bounds)):
            if len(res_bounds[i]) > 0:
                losses = res_bounds[i]
                top_losses, _ = torch.topk(losses, B//8, largest=True)
                selected_losses.append(top_losses)
        selected_losses_mean = [selected_loss.mean() for selected_loss in selected_losses if len(selected_loss) > 0]
        if len(selected_losses_mean) == 0:
            return torch.tensor(0.0).cuda()
        else:
            res = torch.stack(selected_losses_mean).mean()
            return res

        #select top 5 high loss in each element in res_bounds


        # res_bounds_mean = [res_bound.mean() for res_bound in res_bounds if len(res_bound) > 0]
        # if len(res_bounds_mean) == 0:
        #     return torch.tensor(0.0).cuda(), torch.tensor([0.0,0.0]).cuda()
        # else:
        #     res_boundwise = torch.stack(res_bounds_mean)
        #     res = res_boundwise
        #     # ipdb.set_trace()
        #     return res
            # convert list into torch array
        # ratio: inner bound ratio


        # if self.return_boundwise_loss:  # return inner bound loss and outer bound loss respectively
        #     return res, res_boundwise
        # else:
        #     return res


class GTWeightedHausdorffDistanceDoubleBoundLoss(Function):
    def __init__(self, return_boundwise_loss=False, alpha=4, beta=1, ratio=0.5):
        """ Batch Region weighted whd loss for inner and outer bound separately
            inner bound -- 1, outer bound -- 2
        :param return_2_terms: bool, Whether to return the 2 terms of the WHD instead of their sum. Default: False.
        :param alpha: int, decay factor
        :param ratio: float, ratio of inner bound, default is 0.5

        """
        self.return_boundwise_loss = return_boundwise_loss
        self.alpha = alpha
        self.beta = beta
        self.ratio = ratio

    def __call__(self, prob_map, gt,regionweight_map):
        """ Compute the Weighted Hausdorff Distance function between the estimated probability map
        and ground truth points. The output is the WHD averaged through all the batch.
        :param prob_map: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        :param regionweight_map: ((H//r x W//r) x H//r x W//r) Tensor of the regionweight_map
        """
        eps = 1e-6
        alpha = self.alpha
        beta = self.beta
        _assert_no_grad(gt)
        _assert_no_grad(regionweight_map)

        B,C,H,W = prob_map.size()
        _,H_,W_ = regionweight_map.size()
        # ipdb.set_trace()
        region_copy = regionweight_map.clone()
        region_test = regionweight_map.unsqueeze(3).repeat(1,1,1,W//W_)
        regionweight_map=regionweight_map.unsqueeze(3).repeat(1,1,H//H_,W//W_).view(-1,H,W) # (B,3*32,3*32)
        # ipdb.set_trace()
        regionweight_map_ = regionweight_map.unsqueeze(1).repeat(1,prob_map.shape[1],1,1) # (B,3,3*32,3*32)


        #expand gt H//r * W//r in B chaannel #(B*9,96,96)
        # ipdb.set_trace()
        gt = gt*regionweight_map
        prob_map = prob_map*regionweight_map_.to(torch.float32)
        # ipdb.set_trace()








        # assert prob_map.dim() == 4, 'The probability map must be (B x C x H x W)'
        # prob_map size [B, C, T, H, W] or [B, C, H, W] | gt size [B, T, H, W] or [B, H, W]
        if prob_map.dim() == 5: # 3D volume
            prob_map = prob_map.permute(0, 2, 1, 3, 4)
            prob_map = prob_map.contiguous().view(-1, *prob_map.size()[2:]) # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims

        batch_size, n_channel, height, width = prob_map.size()
        assert batch_size == len(gt), 'prob map and GT must have the same length'

        max_dist = math.sqrt(height ** 2 + width ** 2)

        n_pixels = height * width
        all_img_locations = torch.meshgrid([torch.arange(height).cuda(),
                                            torch.arange(width).cuda()])
        all_img_locations = torch.stack(all_img_locations).permute(1, 2, 0).view(n_pixels, -1).float()  # H*W, 2

        # print(height, width, all_img_locations.size())
        # here we consider inner bound and outer bound respectively
        res_bounds_lst = [[] for _ in range(0, n_channel)]
        for b in range(batch_size):

            prob_map_b, gt_b = prob_map[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx) # for different bounds (1 - inner, 2 - outer)
                gt_pts = torch.nonzero(gt_bb).float()  # N, 2
                n_gt_pts = gt_pts.size()[0]
                prob_map_bb = prob_map_b[bound_inx]
                if n_gt_pts > 0:
                    d_matrix = cdist(all_img_locations, gt_pts)
                    p = prob_map_bb.view(prob_map_bb.numel())

                    n_est_pts = (p**beta).sum()
                    p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

                    term_1 = (1 / (n_est_pts + eps)) * torch.sum(p**beta * torch.min(d_matrix, 1)[0])
                    d_div_p = torch.min((d_matrix + eps) /
                                        (p_replicated**alpha + eps / max_dist), 0)[0]
                    d_div_p = torch.clamp(d_div_p, 0, max_dist)
                    term_2 = torch.mean(d_div_p)
                    # set different ratio for inner and outer bound
                    res_bounds_lst[bound_inx].append(term_1 + term_2)
                # ipdb.set_trace()

        res_bounds = [torch.stack(res_bounds_lst[i]) for i in range(1, n_channel) if len(res_bounds_lst[i]) > 0]

        selected_losses = []
        for i in range(len(res_bounds)):
            if len(res_bounds[i]) > 0:
                losses = res_bounds[i]
                if len(losses) > B//8:
                    losses, _ = torch.topk(losses, B//8, largest=True)
                # top_losses, _ = torch.topk(losses, B//8, largest=True)
                selected_losses.append(losses)
        selected_losses_mean = [selected_loss.mean() for selected_loss in selected_losses if len(selected_loss) > 0]
        if len(selected_losses_mean) == 0:
            return torch.tensor(0.0).cuda()
        else:
            res = torch.stack(selected_losses_mean).mean()
            return res

        #select top 5 high loss in each element in res_bounds


        # res_bounds_mean = [res_bound.mean() for res_bound in res_bounds if len(res_bound) > 0]
        # if len(res_bounds_mean) == 0:
        #     return torch.tensor(0.0).cuda(), torch.tensor([0.0,0.0]).cuda()
        # else:
        #     res_boundwise = torch.stack(res_bounds_mean)
        #     res = res_boundwise
        #     # ipdb.set_trace()
        #     return res
        # convert list into torch array
        # ratio: inner bound ratio


        # if self.return_boundwise_loss:  # return inner bound loss and outer bound loss respectively
        #     return res, res_boundwise
        # else:
        #     return res
class BECELoss(Function):
    """ boundary-enhanced cross-entropy loss for boundary segmentation defined in　
    ”H. Oda et. al, BESNet: Boundary-Enhanced　Segmentation of Cells　in Histopathological Images, MICCAI 2018”
    """

    def __init__(self, alpha=1.0, beta=0.1):
        super(BECELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def __call__(self, output_bdp, output_mdp, target_bdp, target_mdp):
        """ calculate boundary
        :param output_bdp: Variable, N x C x *, probabilities for each boundary class
        :param output_mdp: Variable, N x C x *, probabilities for each segmentation class
        :param target_bdp: Variable, N x *, GT labels for boundary
        :param target_mdp: Variable, N x *, GT labels for segmentation
        """

        ## BDP loss (simply weighted cross entropy loss)
        loss_bdp = WeightedCrossEntropy()(output_bdp, target_bdp)

        ## MDP loss
        encoded_target = output_mdp.data.clone().zero_()
        encoded_target.scatter_(1, target_mdp.unsqueeze(1), 1)
        encoded_target = Variable(encoded_target, requires_grad=False)

        # calculate w = positive / negative
        w = (target_mdp != 0).sum() / (target_mdp == 0).sum()

        # calculate weight for MDP loss
        prob_bdp = F.softmax(output_bdp, 1)
        weight = self.alpha * torch.max(input=self.beta - prob_bdp, other=torch.zeros_like(prob_bdp)) + 1.0
        weight[:, 0] = w
        weight = weight.detach()

        loss_mdp = F.binary_cross_entropy_with_logits(output_mdp, encoded_target, weight=weight,
                                                      reduction="mean")

        return loss_bdp + loss_mdp


if __name__ == "__main__":
    pass