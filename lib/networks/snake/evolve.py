import torch.nn as nn
from .snake import Snake
from lib.utils.snake import snake_gcn_utils, snake_config, snake_decode, active_spline
import torch
from lib.utils import data_utils
import sys


class Evolution(nn.Module):
    def __init__(self):
        super(Evolution, self).__init__()

        self.fuse = nn.Conv1d(128, 64, 1)#64->32
        self.init_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')#64->32
        self.evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid') #64->32
        self.iter = 2
        for i in range(self.iter):
            evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
            self.__setattr__('evolve_gcn'+str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = snake_gcn_utils.prepare_training(output, batch)
        output.update({'i_it_4py': init['i_it_4py'], 'i_it_py': init['i_it_py']})
        output.update({'i_gt_4py': init['i_gt_4py'], 'i_gt_py': init['i_gt_py']})
        return init

    def prepare_training_evolve(self, output, batch, init):
        evolve = snake_gcn_utils.prepare_training_evolve(output['ex_pred'], init)
        output.update({'i_it_py': evolve['i_it_py'], 'c_it_py': evolve['c_it_py'], 'i_gt_py': evolve['i_gt_py']})
        evolve.update({'py_ind': init['py_ind']})
        return evolve

    def prepare_testing_init(self, output):
        init = snake_gcn_utils.prepare_testing_init(output['detection'][..., :4], output['detection'][..., 4])
        output['detection'] = output['detection'][output['detection'][..., 4] > snake_config.ct_score]
        output.update({'it_ex': init['i_it_4py']})
        return init

    def prepare_testing_evolve(self, output, h, w):
        ex = output['ex']
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w-1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h-1)
        evolve = snake_gcn_utils.prepare_testing_evolve(ex)
        output.update({'it_py': evolve['i_it_py']})
        return evolve
    def prepare_testing_evolve_hybrid(self, output, h, w,batch):
        # ex ,center,points, point_0_indx, point_1_indx, point_2_indx, point_3_indx\
        #     , point_4_indx, point_5_indx, point_6_indx, point_7_indx, point_8_indx, point_9_indx\
        #     , point_10_indx, point_11_indx,ex_8,cls= snake_gcn_utils.get_8_points(output,snake_config.threshold)
        # ex = ex.cuda()

        # ex,cls = snake_gcn_utils.get_octogan_ND(output, snake_config.threshold,batch)
        ex,cls = snake_gcn_utils.get_octogan(output, snake_config.threshold,batch)
        # ex,cls = snake_gcn_utils.get_octagon_by_ellipse(output, snake_config.threshold,batch)
        if not output['check']:
            return output

        # write a function to reverse the list
        # ex = ex.cuda()



        ex=ex.cuda()
        ex = ex/snake_config.ro
        # ex = torch.Tensor(ex,dtype=torch.float).cuda()
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w-1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h-1)
        evolve = snake_gcn_utils.prepare_testing_evolve(ex)
        # print("evolve_shape:",evolve["i_it_py"].shape)
        # sys.exit()
        # print("center_in_evo:",center)
        # print("cls:",cls)
        output.update({'it_py': evolve['i_it_py'],'cls': cls})
        # output.update({'it_py': evolve['i_it_py'],'center':center,'points':points,'point_0_indx':point_0_indx,'point_1_indx':point_1_indx
        #                ,'point_2_indx':point_2_indx,'point_3_indx':point_3_indx,'point_4_indx':point_4_indx,'point_5_indx':point_5_indx
        #                ,'point_6_indx':point_6_indx,'point_7_indx':point_7_indx,'point_8_indx':point_8_indx,'point_9_indx':point_9_indx
        #                ,'point_10_indx':point_10_indx,'point_11_indx':point_11_indx,'ex_8':ex_8,'cls': cls})
        return evolve

    def init_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros([0, 4, 2]).to(i_it_poly)

        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        center = (torch.min(i_it_poly, dim=1)[0] + torch.max(i_it_poly, dim=1)[0]) * 0.5
        ct_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)
        init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature)], dim=1)
        # [36,128,40]

        init_feature = self.fuse(init_feature)
        # [36,64,40]

        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        #[36,66,40]
        # print("init_input_shape:", init_input.shape)
        # sys.exit()
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)

        i_poly = i_it_poly + snake(init_input, adj).permute(0, 2, 1)
        # [36,40,2]
        # print("i_poly_shape1:", i_poly.shape)
        i_poly = i_poly[:, ::snake_config.init_poly_num//4]
        # print("i_poly_shape2:", i_poly.shape)
        # sys.exit()

        return i_poly

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature,i_it_poly, ind, h, w)
        c_it_poly = c_it_poly * snake_config.ro
        # c_it_poly = c_it_poly
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        i_poly = i_it_poly * snake_config.ro + snake(init_input, adj).permute(0, 2, 1)
        # i_poly = i_it_poly  + snake(init_input, adj).permute(0, 2, 1)
        return i_poly

    def evolve_poly_seg(self, snake, cnn_feature, cnn_feature_seg, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature,init_feature_seg = snake_gcn_utils.get_gcn_feature_seg(cnn_feature, cnn_feature_seg, i_it_poly, ind, h, w)
        c_it_poly = c_it_poly * snake_config.ro
        # c_it_poly = c_it_poly
        init_input = torch.cat([init_feature,init_feature_seg , c_it_poly.permute(0, 2, 1)], dim=1)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        i_poly = i_it_poly * snake_config.ro + snake(init_input, adj).permute(0, 2, 1)
        # i_poly = i_it_poly  + snake(init_input, adj).permute(0, 2, 1)
        return i_poly

    def forward(self, output, cnn_feature, batch=None):
    # def forward(self, output, cnn_feature,cnn_feature_seg, batch=None):
        ret = output
        # ret.update({'check':True})


        # print("batch:",batch)

        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)
            # print("in training")

            # print("cnn_feature:",cnn_feature.shape)
            # print("init['i_it_4py']shape:", init['i_it_4py'].shape)
            # print("init['c_it_4py']_shape:", init['c_it_4py'].shape)


            # ex_pred = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['4py_ind'])
            # ret.update({'ex_pred': ex_pred, 'i_gt_4py': output['i_gt_4py']})

            # with torch.no_grad():
            #     init = self.prepare_training_evolve(output, batch, init)

            py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, init['i_it_py'], init['c_it_py'], init['py_ind'])
            # py_pred = self.evolve_poly_seg(self.evolve_gcn, cnn_feature, cnn_feature_seg, init['i_it_py'],init['c_it_py'], init['py_ind'])
            py_preds = [py_pred]
            for i in range(self.iter):
                py_pred = py_pred / snake_config.ro
                c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred)
                evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                py_pred = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'])
                # py_pred = self.evolve_poly_seg(evolve_gcn, cnn_feature, cnn_feature_seg, py_pred, c_py_pred,init['py_ind'])
                py_preds.append(py_pred)
            ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro})
        #     print("self.traning:", self.training)
        # print("evolve: mid")
        # sys.exit()

        if not self.training:
            with torch.no_grad():
                ret.update({'check': True})
                ret.update({'mean_0': 0,'mean_1': 0,'cov_0': 0,'cov_1': 0,})
                # init = self.prepare_testing_init(output)
                # ex = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['ind'])
                # ret.update({'ex': ex})
                ind=[]
                # print("output_out_shape:",output["prob_map"].shape)



                # print("ind:",ind)

                evolve = self.prepare_testing_evolve_hybrid(output, cnn_feature.size(2), cnn_feature.size(3),batch)
                if not output['check']:
                    return output
                for i in range(len(output["prob_map_boundary"])):
                    for j in range(len(evolve['i_it_py'])):
                        # print("len(evolve['i_it_py']):",len(evolve['i_it_py']))
                        ind.append(i)
                ind = torch.Tensor(ind)

                # print("evolve['i_it_py']_shape:", evolve['i_it_py'])
                # print("cnn_feature_shape:",cnn_feature.shape)
                # print("cnn_feature_seg_shape:", cnn_feature_seg.shape)
                py = self.evolve_poly(self.evolve_gcn, cnn_feature, evolve['i_it_py'], evolve['c_it_py'], ind)
                # py = self.evolve_poly_seg(self.evolve_gcn, cnn_feature, cnn_feature_seg, evolve['i_it_py'],evolve['c_it_py'], ind)
                pys = [py]
                for i in range(self.iter):
                    py = py / snake_config.ro
                    c_py = snake_gcn_utils.img_poly_to_can_poly(py)
                    evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                    # py = self.evolve_poly_seg(evolve_gcn, cnn_feature,cnn_feature_seg, py, c_py, ind)
                    py = self.evolve_poly(evolve_gcn, cnn_feature,  py , c_py, ind)
                    pys.append(py / snake_config.ro)
                ret.update({'py': pys,'i_it_py':evolve['i_it_py']})

        return output

