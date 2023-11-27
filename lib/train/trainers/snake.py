import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import net_utils
import torch
import sys
from .loss import *
import ipdb


def create_region_map(size =4):
    region_map = torch.zeros((size*size,size,size),dtype=torch.int64).cuda()
    for i in range(size):
        for j in range(size):
            region_map[i*size+j,i,j]= 1
    return region_map

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.hybrid_crit=WeightedHausdorffDistanceDoubleBoundLoss()
        self.focalhdf_crit=FocalWeightedHausdorffDistanceDoubleBoundLoss()
        self.focalhdf_tanh_v2=FocalWeightedHausdorffDistanceDoubleBoundLoss_tanh_v2()
        self.focalhdf_tanh_v3=FocalWeightedHausdorffDistanceDoubleBoundLoss_tanh_v3()
        self.BRWhybrid_crit=BRWWeightedHausdorffDistanceDoubleBoundLoss()
        self.gt_crit = GTWeightedHausdorffDistanceDoubleBoundLoss()
        self.hybrid_crit_seg =GeneralizedDiceLoss()
        self.l2 = nn.MSELoss()
        self.BCE = nn.BCELoss(weight=torch.Tensor([1,5]))
        self.WCE = WeightedCrossEntropy()

        self.ct_crit = net_utils.FocalLoss()
        self.wh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.reg_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.ex_crit = torch.nn.functional.smooth_l1_loss
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.clstm_loss = torch.nn.functional.smooth_l1_loss
        self.focal_py_crit = FocalSmoothL1Loss( beta=1.0, alpha=0.2,gamma=2.0)
        self.focal_wh_crit = FocalSmoothL1Loss( beta=1.0, alpha=2,gamma=2.0)
        # self.focal_ex_crit = FocalSmoothL1Loss()
    # def forward(self, batch):
    #     # print("batch:",batch)
    #     output = self.net(batch['inp'], batch)
    #
    #     scalar_stats = {}
    #     loss = 0
    #     # print("output['prob_map']_shape",output['prob_map'].shape)
    #     # print("batch['mask']_shape", batch['mask'].shape)
    #     # sys.exit()
    #     # sys.exit()
    #     # output['prob_map']_shape torch.Size([3, 3, 192, 192])
    #     # batch['mask']_shape torch.Size([3, 192, 192])
    #     risk_label = batch['risk_label']
    #     risk_label = risk_label.type(torch.cuda.FloatTensor)
    #
    #     risk_loss = self.l2(F.softmax(output['prob'],dim=1), risk_label)
    #     scalar_stats.update({'risk_loss':risk_loss})
    #     loss += risk_loss
    #
    #
    #     scalar_stats.update({'loss': loss})
    #     image_stats = {}
    #
    #     return output, loss, scalar_stats, image_stats
    def shape_loss(self, pred, targ_shape):
        pre_dis = torch.cat((pred[:,1:], pred[:,0].unsqueeze(1)), dim=1)
        pred_shape = pre_dis-pred
        # targ_shape = targ(:,:,1:)-targ(:,:,:-1)
        loss = self.py_crit(pred_shape, targ_shape)
        return loss
    def forward(self, batch,epoch):
        # print("batch:",batch)


        # output = self.net(batch['inp_c'], batch)
        output = self.net(batch['inp'], batch)
        # if epoch <= 0:
        #     scalar_stats = {}
        #     loss = 0
        #     # print("output['prob_map']_shape",output['prob_map'].shape)
        #     # print("batch['mask']_shape", batch['mask'].shape)
        #     # sys.exit()
        #     # sys.exit()
        #     # output['prob_map']_shape torch.Size([3, 3, 192, 192])
        #     # batch['mask']_shape torch.Size([3, 192, 192])
        #     # hybrid_health_loss = self.hybrid_crit(F.softmax(output['output_health'], dim=1), batch['mask'])
        #     # scalar_stats.update({'hybrid_health_loss': hybrid_health_loss})
        #     # loss += hybrid_health_loss
        #     # mask_seg = batch['mask_seg_plaque'].type(torch.cuda.LongTensor)
        #     # hybrid_loss_seg = self.hybrid_crit_seg(F.softmax(output['prob_map_seg'], dim=1), mask_seg)
        #     hybrid_loss = self.hybrid_crit(F.softmax(output['prob_map'], dim=1), batch['mask'])
        #     scalar_stats.update({'hybrid_loss': hybrid_loss})
        #
        #     label_health = batch['label_health'].type(torch.cuda.FloatTensor)
        #     class_health_loss = self.BCE(output['prob_health'], label_health)
        #     scalar_stats.update({'Class_health_loss': class_health_loss})
        #     loss += class_health_loss
        #     scalar_stats.update({'loss': loss})
        #     image_stats = {}
        #
        #     return output, loss, scalar_stats, image_stats



        scalar_stats = {}
        loss = 0
        # print("output['prob_map']_shape",output['prob_map'].shape)
        # print("batch['mask']_shape", batch['mask'].shape)
        # sys.exit()
        # sys.exit()
        # output['prob_map']_shape torch.Size([3, 3, 192, 192])
        # batch['mask']_shape torch.Size([3, 192, 192])

        # print("output['prob_map']:",output['prob_map'])
        # print("output['prob_map'].shape:",output['prob_map'].shape)
        # print("batch['mask]:",batch['mask'].shape)
        #
        # mask = batch['mask'].detach().cpu().numpy()
        # print("mask_unique:",np.unique(mask))

        # print('batch_unique:',np.unique(batch['mask'].cpu()))
#plaque segmentation

        # mask_seg = batch["mask_seg"].type(torch.cuda.LongTensor)

        # mask_seg = batch["mask_seg_volume"].type(torch.cuda.LongTensor)
        # hybrid_loss_seg = self.hybrid_crit_seg(output['prob_map_seg'], mask_seg)
        # scalar_stats.update({'hybrid_loss_seg':hybrid_loss_seg})
        # loss += 10*hybrid_loss_seg

# boundary
#         focalhdf_loss = self.focalhdf_crit(F.softmax(output['prob_map_boundary'],dim=1), batch['mask'])
#         scalar_stats.update({'focal_loss':focalhdf_loss})
#
#         focalhdf_loss,focal_term1,focal_term2 = self.focalhdf_tanh_v3(F.softmax(output['prob_map_boundary'],dim=1), batch['mask'])
#         scalar_stats.update({'focalhdftanh_v3_loss':focalhdf_loss,'focal_term1':focal_term1,'focal_term2':focal_term2})


        hybrid_loss,term1,term2 = self.hybrid_crit(F.softmax(output['prob_map_boundary'],dim=1), batch['mask'])
    #     # hybrid_loss_clone = hybrid_loss.clone().detach()
    #
    # # #
        scalar_stats.update({'hybrid_loss':hybrid_loss,'term1':term1,'term2':term2})
        loss += hybrid_loss


        #
        # focalhdf_loss,focal_term1,focal_term2 = self.focalhdf_tanh_v3(F.softmax(output['prob_map_boundary'],dim=1), batch['mask'])
        # scalar_stats.update({'focalhdftanh_v3_loss':focalhdf_loss,'focal_term1':focal_term1,'focal_term2':focal_term2})
        # # sum_r_res_bounds_div_0 = r_res_bounds_div[0].sum()
        # # sum_r_res_bounds_div_1 = r_res_bounds_div[1].sum()
        # # ratio_r_res_bounds_div_0 = sum_r_res_bounds_div_0/len(r_res_bounds_div[0])
        # # ratio_r_res_bounds_div_1 = sum_r_res_bounds_div_1/len(r_res_bounds_div[1])
        # # scalar_stats.update({'focalhdftanh_v3_loss':focalhdf_loss,'focal_term1':focal_term1,'focal_term2':focal_term2,"sum_r_res_bounds_div_0":sum_r_res_bounds_div_0,"sum_r_res_bounds_div_1":sum_r_res_bounds_div_1,'ratio_r_res_bounds_div_0':ratio_r_res_bounds_div_0,'ratio_r_res_bounds_div_1':ratio_r_res_bounds_div_1})
        # loss += focalhdf_loss

#         if epoch <=5:
#             # focalhdf_loss,focal_term1,focal_term2 = self.focalhdf_tanh_v2(F.softmax(output['prob_map_boundary'],dim=1), batch['mask'])
#             # scalar_stats.update({'focalhdftanh_v2_loss':focalhdf_loss,'focal_term1':focal_term1,'focal_term2':focal_term2})
#             # loss += focalhdf_loss
#             # hybrid_loss,term1,term2 = self.hybrid_crit(F.softmax(output['prob_map_boundary'],dim=1), batch['mask'])
#             #
#             # scalar_stats.update({'hybrid_loss':hybrid_loss,'term1':term1,'term2':term2})
#             hybrid_loss,_,_ = self.hybrid_crit(F.softmax(output['prob_map_boundary'],dim=1), batch['mask'])
#
#             scalar_stats.update({'hybrid_loss':hybrid_loss})
#             loss += hybrid_loss
#
#
#
# #focalhdf loss
#         if epoch >5:
#             focalhdf_loss = self.focalhdf_crit(F.softmax(output['prob_map_boundary'],dim=1), batch['mask'])
#             scalar_stats.update({'focalhdftanh_loss':focalhdf_loss})
#             loss += 0.05*focalhdf_loss
#             hybrid_loss,_,_= self.hybrid_crit(F.softmax(output['prob_map_boundary'],dim=1), batch['mask'])
#             loss += hybrid_loss
#
#             scalar_stats.update({'hybrid_loss':hybrid_loss})


# # physiology-aware loss:
#
        # physology_loss = 0
        # mask_p_expand=batch['plaque_patch'].type(torch.cuda.FloatTensor).repeat(1,1,32,32) #(B,3,96,96)
        # mask_expand = batch['mask'].type(torch.cuda.FloatTensor).unsqueeze(1).repeat(1,3,1,1) #(B,3,96,96)
        # #output['prob_map_masked']:#(B,,3-1,3,96,96)
        # #batch['plaque_patch']:(B,3, 96//32, 96//32)
        # #batch['mask']:(B,96,96)
        # mask_masked = mask_expand*mask_p_expand #(B,3,96,96)
        # for i in range(1,output['prob_map_masked'].size(1)):
        #     physology_loss += self.hybrid_crit(output['prob_map_masked'][:,i,:,:],mask_masked[:,i,:,:])
        # if physology_loss is not None:
        #     scalar_stats.update({'physology_loss':physology_loss})
        #     loss += physology_loss
#
# # physiology-aware loss:
#
        # if epoch >5:
        #     region_map= create_region_map(3)
        #     # region_map= create_region_map(2)
        #
        #     physology_loss = self.BRWhybrid_crit(F.softmax(output['prob_map_boundary'],dim=1),batch['mask'],region_map)
        #     scalar_stats.update({'physology_loss':physology_loss})
        #     loss += 0.3*physology_loss

# physiology-aware loss - gt label:


        # region_map= create_region_map(3)
        # if epoch >5:
        #     region_map = batch['plaque_patch_2d'].type(torch.cuda.LongTensor)
        #
        #
        #     physology_loss = self.gt_crit(F.softmax(output['prob_map_boundary'],dim=1),batch['mask'],region_map)
        #     scalar_stats.update({'physology_loss_gt':physology_loss})
        #     loss += 0.5*physology_loss
# #clstm start
#         clstm_loss = 0
#
#         output_clstm = torch.squeeze(output['out_clstm'],2)
#
#
#         # for i in range(output['out_clstm'].size(1)-1):
#         #     # # print("output_clstm[:,i,:,:]:",output_clstm[:,i,:,:].shape)
#         #     # print("output['out_clstm']:",output['out_clstm'].shape)
#         #     # print("batch['inp']:",batch['inp'].shape)
#         #     clstm_loss += self.clstm_loss(output_clstm[:,i,:,:], batch['inp'][:,i+1,:,:])
#         #
#         #
#         #
#         # scalar_stats.update({'clstm_loss':clstm_loss})
#         # loss += 0.01*clstm_loss
# #clstm end
#ellipse
        # hybrid_health_loss = self.hybrid_crit(F.softmax(output['output_health'], dim=1), batch['mask'])
        # scalar_stats.update({'hybrid_health_loss': hybrid_health_loss})
        # loss += hybrid_health_loss
        # mask_seg_plq_only_fromcls = batch['mask_seg_plq_only'].type(torch.cuda.LongTensor)
        # hybrid_loss_seg_plq_only_fromcls = self.hybrid_crit_seg(output['plq_seg_from_cls'], mask_seg_plq_only_fromcls)
        # scalar_stats.update({'hybrid_loss_seg_plq_only_fromcls': hybrid_loss_seg_plq_only_fromcls})
        # loss += hybrid_loss_seg_plq_only_fromcls

# #segwall
#         mask_seg_wall = batch['mask_seg_wall'].type(torch.cuda.LongTensor)
#         hybrid_loss_seg_wall = self.hybrid_crit_seg(output['prob_map_seg'], mask_seg_wall)
#         scalar_stats.update({'hybrid_loss_seg_wall': hybrid_loss_seg_wall})
#         loss += hybrid_loss_seg_wall

 #seg_from_cls
        # mask_seg_wall_cls = batch['mask_seg_wall'].type(torch.cuda.LongTensor)
        # hybrid_loss_seg_wall_cls = self.hybrid_crit_seg(output['seg_from_cls'], mask_seg_wall_cls)
        # scalar_stats.update({'hybrid_loss_seg_wall_from_cls': hybrid_loss_seg_wall_cls})
        # loss +=hybrid_loss_seg_wall_cls


        # mask_seg_plq_only = batch['mask_seg_plq_only'].type(torch.cuda.LongTensor)
        # check = torch.sum(mask_seg_plq_only)
        # print("check_outLoss:",check)
        # hybrid_loss_seg_plq_only = self.hybrid_crit_seg(output['prob_map_seg_1'], mask_seg_plq_only)



        # scalar_stats.update({'hybrid_loss_seg_plq_only': hybrid_loss_seg_plq_only})
        # loss += hybrid_loss_seg_plq_only
#
#class_health
        # label_health = batch['label_health'].type(torch.cuda.FloatTensor)
        # class_health_loss = self.BCE(output['prob_health'],label_health)
        # scalar_stats.update({'Class_health_loss': class_health_loss})
        # # if epoch <= 6:
        # loss += class_health_loss

#patch_class
        # plaque_patch_2d = batch['plaque_patch_2d'].type(torch.cuda.LongTensor)
        # plaque_patch_loss = self.WCE(output['prob_map_patch'],plaque_patch_2d)
        # scalar_stats.update({'plaque_patch_loss': plaque_patch_loss})
        # # if epoch <= 6:
        # loss += plaque_patch_loss
#ellipse



        # print("epoch known:")

        # mask_dis = batch['mask_dis'].type(torch.cuda.FloatTensor)
        # hybrid_loss_seg = self.l2(output['prob_map_seg'],mask_dis)
        # scalar_stats.update({'hybrid_loss_seg': hybrid_loss_seg})
        # loss += 5*hybrid_loss_seg

        # wh_loss = self.wh_crit(output['wh'], batch['wh'], batch['ct_ind'], batch['ct_01'])
        # scalar_stats.update({'wh_loss': wh_loss})
        # loss += 0.1 * wh_loss c

        # reg_loss = self.reg_crit(output['reg'], batch['reg'], batch['ct_ind'], batch['ct_01'])c
        # scalar_stats.update({'reg_loss': reg_loss})
        # loss += reg_loss
        # print("i_gt_4py.shape:",output['i_gt_4py'].shape)
        # sys.exit()
        # ex_loss = self.ex_crit(output['ex_pred'], output['i_gt_4py'])
        # scalar_stats.update({'ex_loss': ex_loss})
        # loss += ex_loss
        if not self.training:
            scalar_stats.update({'loss': loss})
            image_stats = {}

            return output, loss, scalar_stats, image_stats

# #wh loss
#         # wh_loss = self.py_crit(output['poly_init'], output['i_gt_py']) # for normal smooth l1 loss
#         # scalar_stats.update({'wh_loss': 0.1 * wh_loss}) # for normal smooth l1 loss
#         # loss += 0.1 * wh_loss  # for normal smooth l1 loss
#         focal_wh_loss = self.focal_wh_crit(output['poly_init'], output['i_gt_py']) # for focal smooth l1 loss
#         scalar_stats.update({'focal_wh_loss': 0.1 * focal_wh_loss})
#         loss += 0.1 * focal_wh_loss
#         # ipdb.set_trace()
#
#
# # poly snake
#         n_predictions = len(output['py_pred'])
#         py_loss = 0.0
#         focal_py_loss = 0.0
#         shape_loss = 0.0
#         py_dis = torch.cat((output['i_gt_py'][:,1:], output['i_gt_py'][:,0].unsqueeze(1)), dim=1)
#         # ipdb.set_trace()
#         tar_shape = py_dis - output['i_gt_py']
#         for i in range(n_predictions):
#             i_weight = 0.8**(n_predictions - i - 1)
#             # py_loss += i_weight * self.py_crit(output['py_pred'][i], output['i_gt_py']) # for normal smooth l1 loss
#             focal_py_loss += i_weight * self.focal_py_crit(output['py_pred'][i], output['i_gt_py']) # for focal smooth l1 loss
#             shape_loss += i_weight * self.shape_loss(output['py_pred'][i], tar_shape)
#
#         # py_loss = py_loss / n_predictions  # for normal smooth l1 loss
#         focal_py_loss = focal_py_loss / n_predictions # for focal smooth l1 loss
#         shape_loss = shape_loss / n_predictions
#         # scalar_stats.update({'py_loss': py_loss})  # for normal smooth l1 loss
#         scalar_stats.update({'focal_py_loss': focal_py_loss})
#         scalar_stats.update({'shape_loss': shape_loss})
#         # loss += py_loss # for normal smooth l1 loss
#         loss += focal_py_loss # for focal smooth l1 loss
#         loss += shape_loss
#
#         scalar_stats.update({'loss': loss})
#         image_stats = {}
#         # ipdb.set_trace()
#
# #deep snake

        py_loss = 0
        output['py_pred'] = [output['py_pred'][-1]]
        for i in range(len(output['py_pred'])):
            py_loss += self.py_crit(output['py_pred'][i], output['i_gt_py']) / len(output['py_pred'])
        scalar_stats.update({'py_loss': py_loss})
        loss += py_loss
#
        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

