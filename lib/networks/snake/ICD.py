from .snake_psnake import Snake, Snake2d
from .update import BasicUpdateBlock, BasicUpdateBlock2d
from lib.utils import data_utils
from lib.utils.snake import snake_gcn_utils_psnake, snake_config_psnake, snake_decode

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import time



class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()
        self.iter = 6  # iteration number

        self.evolve_gcn = Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid', need_fea=True)
        self.evolve_gcn2d=Snake2d(state_dim=128, feature_dim=64 + 2,kernel_h=2, conv_type='dgrid2d', need_fea=True)
        self.update_block = BasicUpdateBlock()
        self.update_block2d = BasicUpdateBlock2d()

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = snake_gcn_utils_psnake.prepare_training(output, batch)
        output.update({'i_gt_py': init['i_gt_py']})
        return init


    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils_psnake.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        i_poly_fea = snake(init_input)
        return i_poly_fea

    def evolve_doublepoly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils_psnake.get_gcn_doublefeature(cnn_feature, i_it_poly, ind, h, w)
        # ipdb.set_trace()
        init_input = torch.cat([init_feature, c_it_poly.permute(0,3,1,2)], dim=1)
        # ipdb.set_trace()
        i_poly_fea = snake(init_input)

        return i_poly_fea

    def evolve_poly_forcefield(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils_psnake.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        i_poly_fea = snake(init_input)
        return i_poly_fea


    def get_ct(self,prob_map):
        y, x = torch.meshgrid(torch.arange(96), torch.arange(96))
        x = x.to(torch.float32).cuda()
        y = y.to(torch.float32).cuda()
        # ipdb.set_trace()

        # Calculate the weighted sum of coordinates for each channel
        weighted_x = torch.sum(x * prob_map, dim=(2, 3))
        weighted_y = torch.sum(y * prob_map, dim=(2, 3))
        # ipdb.set_trace()

        # Calculate the total weights for each channel
        total_weights = torch.sum(prob_map, dim=(2, 3))
        ct_x = weighted_x / total_weights
        ct_y = weighted_y / total_weights
        # ipdb.set_trace()
        ct_x = ct_x.unsqueeze(2)
        ct_y = ct_y.unsqueeze(2)

        ct = torch.cat([ct_x, ct_y], dim=2)
        # ipdb.set_trace()
        return ct
    def clip_to_image(self, poly, h, w):
        poly[..., :2] = torch.clamp(poly[..., :2], min=0)
        poly[..., 0] = torch.clamp(poly[..., 0], max=w - 1)
        poly[..., 1] = torch.clamp(poly[..., 1], max=h - 1)
        return poly
    # def get_forcefield(self, prob_map):
    #     B,C,H,W = prob_map.size()
    #     forcefield = torch.zeros((B,C,2,H,W)).cuda()
    #     force_x = torch.zeros((B,C,H,W)).cuda()
    #     force_y = torch.zeros((B,C,H,W)).cuda()
    #     y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    #     x = x.to(torch.float32).cuda()
    #     y = y.to(torch.float32).cuda()
    #     start_time = time.time()
    #     # ipdb.set_trace()
    #
    #
    #     prob_repeat = prob_map.unsqueeze(2).unsqueeze(2).expand(B,C,H,W,H,W)
    #
    #     for i in range(B):
    #         for j in range(C):
    #             for k in range(H):
    #                 for m in range(W):
    #                     force_x[i,j,k,m] = torch.sum((x - k) * prob_map[i,j]/((x-k)**2+(y-m)**2 + 1e-6)**0.5)
    #                     # force_y[i,j,k,m] = torch.sum((y_all[i,j,k,m] - y_current[i,j,k,m]+1e-10) * prob_repeat[i,j,k,m]/((x_all[i,j,k,m] - x_current[i,j,k,m])**2 + (y_all[i,j,k,m] - y_current[i,j,k,m])**2 + 1e-6)**0.5)
    #                     # ipdb.set_trace()
    #     end_time = time.time()
    #     ipdb.set_trace()
    #
    #
    #             #force_x = torch.sum((x_all - x_current) * prob_repeat/((x_all - x_current)**2 + (y_all - y_current)**2 + 1e-6)**0.5,dim=(4,5))
    #
    #     ipdb.set_trace()
    #     return force_x






    def use_gt_detection(self, output, batch):
        batchsize, _, height, width = output['prob_map_boundary'].size()
        wh_pred = output['wh'] #B,512,H,W

        ct_01 = batch['ct_01'].byte()
        ct_ind = batch['ct_ind'][ct_01]
        ct_img_idx = batch['ct_img_idx'][ct_01]

        ct_x, ct_y = ct_ind % width//snake_config_psnake.ro, ct_ind // width//snake_config_psnake.ro
        # ct_x_delta = torch.randint(-1, 1 + 1, (ct_x.size(0),)).cuda(ct_x.device)
        # ct_y_delta = torch.randint(-1, 1 + 1, (ct_y.size(0),)).cuda(ct_y.device)
        # ct_x = ct_x + ct_x_delta
        # ct_y = ct_y + ct_y_delta
        # ipdb.set_trace()

        ct_img_idx = ct_img_idx % batchsize
        ct_cls_ind = batch['ct_cls'][ct_01]
        # ipdb.set_trace()

        if ct_x.size(0) == 0:
            ct_offset = wh_pred[ct_img_idx, ct_cls_ind,..., ct_y, ct_x].view(ct_x.size(0), 1,  2)
        else:
            ct_offset = wh_pred[ct_img_idx, ct_cls_ind,..., ct_y, ct_x].view(ct_x.size(0), -1, 2)
        # ipdb.set_trace()

        ct_x, ct_y = ct_x[:, None].float(), ct_y[:, None].float()
        ct = torch.cat([ct_x, ct_y], dim=1)

        init_polys = ct_offset + ct.unsqueeze(1).expand(ct_offset.size(0), ct_offset.size(1), ct_offset.size(2))
        # ipdb.set_trace()
        # init_polys = self.clip_to_image(init_polys, height//snake_config_psnake.ro, width//snake_config_psnake.ro)
        # ipdb.set_trace()

        output.update({'poly_init': init_polys * snake_config_psnake.ro})
        # ipdb.set_trace()
        return init_polys

    def use_gt_detection_doublepoly(self, output, batch):
        batchsize, _, height, width = output['prob_map_boundary'].size()
        wh_pred = output['wh'] #B,512,H,W

        ct_01 = batch['ct_01'].byte()
        # assert len(ct_01) % 2 == 0
        # ipdb.set_trace()
        ct_ind = batch['ct_ind'][ct_01]
        ct_img_idx = batch['ct_img_idx'][ct_01]

        ct_x, ct_y = ct_ind % width//snake_config_psnake.ro, ct_ind // width//snake_config_psnake.ro
        ct_x_delta = torch.randint(-1, 1 + 1, (ct_x.size(0),)).cuda(ct_x.device)
        ct_y_delta = torch.randint(-1, 1 + 1, (ct_y.size(0),)).cuda(ct_y.device)
        ct_x = ct_x + ct_x_delta
        ct_y = ct_y + ct_y_delta
        # ipdb.set_trace()

        ct_img_idx = ct_img_idx % batchsize
        ct_cls_ind = batch['ct_cls'][ct_01]
        # ipdb.set_trace()

        if ct_x.size(0) == 0:
            ct_offset = wh_pred[ct_img_idx, ct_cls_ind,..., ct_y, ct_x].view(ct_x.size(0), 1,  2)
        else:
            ct_offset = wh_pred[ct_img_idx, ct_cls_ind,..., ct_y, ct_x].view(ct_x.size(0)//2,2, -1, 2)

        # ipdb.set_trace()

        ct_x, ct_y = ct_x[:, None].float(), ct_y[:, None].float()
        ct = torch.cat([ct_x, ct_y], dim=1).view(ct_x.size(0)//2,2, 2)

        init_polys = ct_offset + ct.unsqueeze(2).expand(ct_offset.size(0), ct_offset.size(1), ct_offset.size(2), ct_offset.size(3))
        # ipdb.set_trace()
        # init_polys = self.clip_to_image(init_polys, height//snake_config_psnake.ro, width//snake_config_psnake.ro)
        # ipdb.set_trace()

        output.update({'poly_init': snake_config_psnake.ro*init_polys.view(-1,128,2) })
        # ipdb.set_trace()
        return init_polys


    def decode_detection(self, output, h, w):
        ct_hm = output['ct_hm']
        wh = output['wh']

        poly_init, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh)

        valid = detection[0, :, 2] >= 0.05  # min_ct_score
        poly_init, detection = poly_init[0][valid], detection[0][valid]

        init_polys = self.clip_to_image(poly_init, h, w)
        output.update({'poly_init_infer': init_polys * snake_config_psnake.ro, 'detection': detection})
        return poly_init, detection
    def prepare_testing_evolve_hybrid(self, output, h, w,batch):
        prob = F.softmax(output['prob_map_boundary'], dim=1)
        wh = output['wh']
        init_poly= snake_gcn_utils_psnake.decode_prob_wh(output,prob,wh,batch)
        return init_poly

    def prepare_testing_evolve_hybrid_double(self, output, h, w,batch):
        isdouble = True
        prob = F.softmax(output['prob_map_boundary'], dim=1)
        wh = output['wh']
        init_poly= snake_gcn_utils_psnake.decode_prob_wh_double(output,prob,wh,batch)
        if init_poly.size(1)<2: # B,1,128,2 => B,2,128,2
            init_poly = init_poly.repeat(1,2,1,1)
            isdouble = False
        return init_poly,isdouble
# # for polysnake 1d
#     def forward(self, output, cnn_feature, batch):
#         ret = output
#         # force_x = self.get_forcefield(F.softmax(output['prob_map_boundary'], dim=1)) # test force field
#         ct = self.get_ct(output["prob_map_boundary"])
#         # ipdb.set_trace()
#         if batch is not None and self.training:
#             with torch.no_grad():
#                 init = self.prepare_training(output, batch)
#             # ipdb.set_trace()
#
#
#             # double_poly_init = self.use_gt_detection_doublepoly(output, batch)
#             # double_poly_init = double_poly_init.detach()
#             # double_py_pred = double_poly_init * snake_config_psnake.ro
#             # c_double_py_pred = snake_gcn_utils_psnake.img_poly_to_can_poly(double_poly_init)
#             # # ipdb.set_trace()
#             # i_double_poly_fea = self.evolve_doublepoly(self.evolve_gcn2d, cnn_feature, double_poly_init, c_double_py_pred, init['py_ind'])  # n*c*128
#             poly_init = self.use_gt_detection(output, batch)
#
#             poly_init = poly_init.detach()
#             py_pred = poly_init * snake_config_psnake.ro
#             c_py_pred = snake_gcn_utils_psnake.img_poly_to_can_poly(poly_init)
#             i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred, init['py_ind'])  # n*c*128
#             net = torch.tanh(i_poly_fea)
#             i_poly_fea = F.leaky_relu(i_poly_fea)
#             py_preds = []
#             for i in range(self.iter):
#                 net, offset = self.update_block(net, i_poly_fea)
#                 py_pred = py_pred + snake_config_psnake.ro * offset
#                 py_preds.append(py_pred)
#
#                 py_pred_sm = py_pred / snake_config_psnake.ro
#                 c_py_pred = snake_gcn_utils_psnake.img_poly_to_can_poly(py_pred_sm)
#                 i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'])
#                 i_poly_fea = F.leaky_relu(i_poly_fea)
#             ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config_psnake.ro})
#
#         # ipdb.set_trace()
#
#
#         if not self.training:
#             with torch.no_grad():
#                 # poly_init, detection = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))
#
#                 poly_init = self.prepare_testing_evolve_hybrid(output, cnn_feature.size(2), cnn_feature.size(3),batch)
#
#
#
#                 ind = torch.zeros((poly_init.size(0)))
#
#                 py_pred = poly_init * snake_config_psnake.ro
#                 c_py_pred = snake_gcn_utils_psnake.img_poly_to_can_poly(poly_init)
#                 # ipdb.set_trace()
#                 i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred,
#                                               ind)
#                 # ipdb.set_trace()
#
#                 if len(py_pred) != 0:
#                     net = torch.tanh(i_poly_fea)
#                     i_poly_fea = F.leaky_relu(i_poly_fea)
#                     for i in range(self.iter):
#                         net, offset = self.update_block(net, i_poly_fea)
#                         py_pred = py_pred + snake_config_psnake.ro * offset
#                         py_pred_sm = py_pred / snake_config_psnake.ro
#
#                         if i != (self.iter - 1):
#                             c_py_pred = snake_gcn_utils_psnake.img_poly_to_can_poly(py_pred_sm)
#                             i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, ind) #init['ind'])
#                             i_poly_fea = F.leaky_relu(i_poly_fea)
#                     py_preds = [py_pred_sm]
#                 else:
#                     py_preds = [i_poly_fea]
#                 ret.update({'py': py_preds})
#         return output

# for polysnake 2d
    def forward(self, output, cnn_feature, batch):
        ret = output
        # force_x = self.get_forcefield(F.softmax(output['prob_map_boundary'], dim=1)) # test force field
        ct = self.get_ct(output["prob_map_boundary"])
        # ipdb.set_trace()
        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)

            double_poly_init = self.use_gt_detection_doublepoly(output, batch)
            double_poly_init = double_poly_init.detach()
            double_py_pred = double_poly_init * snake_config_psnake.ro
            c_double_py_pred = snake_gcn_utils_psnake.img_poly_to_can_poly(double_poly_init)
            # ipdb.set_trace()
            i_double_poly_fea = self.evolve_doublepoly(self.evolve_gcn2d, cnn_feature, double_poly_init, c_double_py_pred, init['py_ind'])  # n*c*128


            double_net = torch.tanh(i_double_poly_fea)
            i_double_poly_fea = F.leaky_relu(i_double_poly_fea)
            double_py_preds = []
            for i in range(self.iter):
                double_net, double_offset = self.update_block2d(double_net, i_double_poly_fea)
                # ipdb.set_trace()
                double_py_pred = double_py_pred + snake_config_psnake.ro * double_offset
                double_py_preds.append(double_py_pred.view(-1, 128, 2))

                double_py_pred_sm = double_py_pred / snake_config_psnake.ro
                c_double_py_pred = snake_gcn_utils_psnake.img_poly_to_can_poly(double_py_pred_sm)
                i_double_poly_fea = self.evolve_doublepoly(self.evolve_gcn2d, cnn_feature, double_py_pred_sm, c_double_py_pred, init['py_ind'])
                i_double_poly_fea = F.leaky_relu(i_double_poly_fea)
            # py_preds = []
            # for i in range(self.iter):
            #     net, offset = self.update_block(net, i_poly_fea)
            #     py_pred = py_pred + snake_config_psnake.ro * offset
            #     py_preds.append(py_pred)
            #
            #     py_pred_sm = py_pred / snake_config_psnake.ro
            #     c_py_pred = snake_gcn_utils_psnake.img_poly_to_can_poly(py_pred_sm)
            #     i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'])
            #     i_poly_fea = F.leaky_relu(i_poly_fea)
            ret.update({'py_pred':double_py_preds, 'i_gt_py': output['i_gt_py'] * snake_config_psnake.ro})

        # ipdb.set_trace()


        if not self.training:
            with torch.no_grad():
                # poly_init, detection = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))

                double_poly_init,isdouble= self.prepare_testing_evolve_hybrid_double(output, cnn_feature.size(2), cnn_feature.size(3),batch)



                ind = torch.zeros((double_poly_init.size(0)))

                double_py_pred = double_poly_init * snake_config_psnake.ro
                c_double_py_pred = snake_gcn_utils_psnake.img_poly_to_can_poly(double_poly_init)
                # ipdb.set_trace()
                i_double_poly_fea = self.evolve_doublepoly(self.evolve_gcn2d, cnn_feature, double_poly_init, c_double_py_pred,
                                              ind)
                # ipdb.set_trace()

                if len(double_py_pred) != 0:
                    double_net = torch.tanh(i_double_poly_fea)
                    i_double_poly_fea = F.leaky_relu(i_double_poly_fea)
                    for i in range(self.iter):
                        double_net, double_offset = self.update_block2d(double_net, i_double_poly_fea)
                        double_py_pred = double_py_pred + snake_config_psnake.ro * double_offset
                        double_py_pred_sm = double_py_pred / snake_config_psnake.ro

                        if i != (self.iter - 1):
                            c_double_py_pred = snake_gcn_utils_psnake.img_poly_to_can_poly(double_py_pred_sm)
                            i_double_poly_fea = self.evolve_doublepoly(self.evolve_gcn2d, cnn_feature, double_py_pred_sm, c_double_py_pred, ind) #init['ind'])
                            i_double_poly_fea = F.leaky_relu(i_double_poly_fea)
                    double_py_preds = [double_py_pred_sm.view(-1, 128, 2)]
                    if not isdouble:
                        double_py_preds = [double_py_pred_sm.view(-1, 128, 2)[1].unsqueeze(0)]
                else:
                    double_py_preds = [i_double_poly_fea]
                ret.update({'py': double_py_preds})
        return output

