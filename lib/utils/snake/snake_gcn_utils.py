import torch
import numpy as np
from lib.utils.snake import snake_decode, snake_config
from lib.csrc.extreme_utils import _ext as extreme_utils
import torch.nn.functional as F
from lib.utils import data_utils
from lib.utils import net_utils
import sys
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from cv2 import cv2

def collect_training(poly, ct_01):
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0)
    # print("ct_01:",ct_01,"ct_01_shape:",ct_01.shape)
    # ct_01 [batch_size,ct_num] show class number in samples
    # print("poly:", poly,"poly_shape:",poly.shape)

    # sys.exit()
    return poly


def prepare_training_init(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}
    init.update({'i_it_4py': collect_training(batch['i_it_4py'], ct_01)})
    init.update({'c_it_4py': collect_training(batch['c_it_4py'], ct_01)})
    init.update({'i_gt_4py': collect_training(batch['i_gt_4py'], ct_01)})
    init.update({'c_gt_4py': collect_training(batch['c_gt_4py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    print("ct_num:",ct_num)
    sys.exit()
    init.update({'ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})

    return init


def prepare_testing_init(box, score):
    i_it_4pys = snake_decode.get_init(box)
    i_it_4pys = uniform_upsample(i_it_4pys, snake_config.init_poly_num)
    c_it_4pys = img_poly_to_can_poly(i_it_4pys)

    ind = score > snake_config.ct_score
    i_it_4pys = i_it_4pys[ind]
    c_it_4pys = c_it_4pys[ind]
    ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(ind.size(0))], dim=0)
    print('i_it_4py_shape:',i_it_4pys.shape)
    sys.exit()
    init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'ind': ind}

    return init


def get_box_match_ind(pred_box, score, gt_poly):
    if gt_poly.size(0) == 0:
        return [], []

    gt_box = torch.cat([torch.min(gt_poly, dim=1)[0], torch.max(gt_poly, dim=1)[0]], dim=1)
    iou_matrix = data_utils.box_iou(pred_box, gt_box)
    iou, gt_ind = iou_matrix.max(dim=1)
    box_ind = ((iou > snake_config.box_iou) * (score > snake_config.confidence)).nonzero().view(-1)
    gt_ind = gt_ind[box_ind]

    ind = np.unique(gt_ind.detach().cpu().numpy(), return_index=True)[1]
    box_ind = box_ind[ind]
    gt_ind = gt_ind[ind]

    return box_ind, gt_ind


def prepare_training_box(ret, batch, init):
    box = ret['detection'][..., :4]
    score = ret['detection'][..., 4]
    batch_size = box.size(0)
    i_gt_4py = batch['i_gt_4py']
    ct_01 = batch['ct_01'].byte()
    ind = [get_box_match_ind(box[i], score[i], i_gt_4py[i][ct_01[i]]) for i in range(batch_size)]
    box_ind = [ind_[0] for ind_ in ind]
    gt_ind = [ind_[1] for ind_ in ind]

    i_it_4py = torch.cat([snake_decode.get_init(box[i][box_ind[i]][None]) for i in range(batch_size)], dim=1)
    if i_it_4py.size(1) == 0:
        return

    i_it_4py = uniform_upsample(i_it_4py, snake_config.init_poly_num)[0]
    c_it_4py = img_poly_to_can_poly(i_it_4py)
    i_gt_4py = torch.cat([batch['i_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    c_gt_4py = torch.cat([batch['c_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_4py = {'i_it_4py': i_it_4py, 'c_it_4py': c_it_4py, 'i_gt_4py': i_gt_4py, 'c_gt_4py': c_gt_4py}

    i_it_py = snake_decode.get_octagon(i_gt_4py[None])
    i_it_py = uniform_upsample(i_it_py, snake_config.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    i_gt_py = torch.cat([batch['i_gt_py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_py = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    ind = torch.cat([torch.full([len(gt_ind[i])], i) for i in range(batch_size)], dim=0)

    if snake_config.train_pred_box_only:
        for k, v in init_4py.items():
            init[k] = v
        for k, v in init_py.items():
            init[k] = v
        init['4py_ind'] = ind
        init['py_ind'] = ind
    else:
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_4py.items()})
        init.update({'4py_ind': torch.cat([init['4py_ind'], ind], dim=0)})
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_py.items()})
        init.update({'py_ind': torch.cat([init['py_ind'], ind], dim=0)})


def prepare_training(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}
    init.update({'i_it_4py': collect_training(batch['i_it_4py'], ct_01)})
    init.update({'c_it_4py': collect_training(batch['c_it_4py'], ct_01)})
    init.update({'i_gt_4py': collect_training(batch['i_gt_4py'], ct_01)})
    init.update({'c_gt_4py': collect_training(batch['c_gt_4py'], ct_01)})

    init.update({'i_it_py': collect_training(batch['i_it_py'], ct_01)})
    init.update({'c_it_py': collect_training(batch['c_it_py'], ct_01)})
    init.update({'i_gt_py': collect_training(batch['i_gt_py'], ct_01)})
    init.update({'c_gt_py': collect_training(batch['c_gt_py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    # print("ct_num:", ct_num)
    # ct_num[batch_size,cls_num]

    # init.update({'4py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update(
        {'4py_ind': torch.cat([torch.full([ct_num[i]], i, dtype=torch.long) for i in range(ct_01.size(0))], dim=0)})

    # print("4py_ind:",init["4py_ind"])
    # [0,0,1,1,2,2...batch_size-1,batchsize-1]
    # sys.exit()
    init.update({'py_ind': init['4py_ind']})

    if snake_config.train_pred_box:
        prepare_training_box(ret, batch, init)

    init['4py_ind'] = init['4py_ind'].to(ct_01.device)
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init


def prepare_training_evolve(ex, init):
    if not snake_config.train_pred_ex:
        evolve = {'i_it_py': init['i_it_py'], 'c_it_py': init['c_it_py'], 'i_gt_py': init['i_gt_py']}
        return evolve

    i_gt_py = init['i_gt_py']

    if snake_config.train_nearest_gt:
        shift = -(ex[:, :1] - i_gt_py).pow(2).sum(2).argmin(1)
        i_gt_py = extreme_utils.roll_array(i_gt_py, shift)

    i_it_py = snake_decode.get_octagon(ex[None])
    i_it_py = uniform_upsample(i_it_py, snake_config.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    evolve = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    return evolve


def prepare_testing_evolve(ex):
    if len(ex) == 0:
        i_it_pys = torch.zeros([0, snake_config.poly_num, 2]).to(ex)
        c_it_pys = torch.zeros_like(i_it_pys)
    else:
        # i_it_pys = snake_decode.get_octagon(ex[None])
        # i_it_pys = uniform_upsample(i_it_pys, snake_config.poly_num)[0]
        i_it_pys = uniform_upsample(ex, snake_config.poly_num)[0]
        c_it_pys = img_poly_to_can_poly(i_it_pys)
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
    return evolve


def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        # print("poly_shape:",poly.shape)
        # [1,2,40,2]
        # sys.exit()
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
        # print("feature_shape:",feature.shape)
        # [2,64,40]
        # sys.exit()
        gcn_feature[ind == i] = feature

    return gcn_feature
def get_gcn_feature_seg(cnn_feature,cnn_feature_seg, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    gcn_feature_seg = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        # print("poly_shape:",poly.shape)
        # [1,2,40,2]
        # sys.exit()
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
        feature_seg = torch.nn.functional.grid_sample(cnn_feature_seg[i:i + 1], poly)[0].permute(1, 0, 2)
        # print("feature_shape:",feature.shape)
        # [2,64,40]
        # sys.exit()
        gcn_feature[ind == i] = feature
        gcn_feature_seg[ind == i] = feature_seg

    return gcn_feature,gcn_feature_seg



def update_ellipse( bound ,ellipse_points_temp ,center,ex):
    # for every point in ellipse_points, check the around
    import math
    ellipse_points = ellipse_points_temp
    for i in range(ellipse_points.shape[0]):
        x = int(ellipse_points[i,0])
        y = int(ellipse_points[i,1])
        arc_elipse = np.arctan2(x-center[0],y-center[1])*180/math.pi
        arc=np.arctan2(ex[:,0]-center[0],ex[:,1]-center[1])*180 / np.pi
        # print("ex:",ex[:4])
        # print("arc:", arc[:4])


        point_indx = np.argwhere((arc<= arc_elipse+5 if arc_elipse+5<=180 else 180 ) & (arc >= arc_elipse-5 if arc_elipse-5>=-180 else -180))
        # print("point_indx:",point_indx)

        dominator = 0
        x_center = 0
        y_center = 0
        for idx in point_indx:
            x_p = int(ex[int(idx),0])
            y_p = int(ex[int(idx),1])
            x_center += x_p
            y_center += y_p
            dominator += 1
        if dominator == 0:
            ellipse_points[i,0] = y
            ellipse_points[i,1] = x
            continue
        x = int(x_center/dominator)
        y = int(y_center/dominator)

        ellipse_points[i,0] = y
        ellipse_points[i,1] = x
    return ellipse_points


def get_octagon_by_ellipse(output1,threshold=0.5,batch=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import ipdb
    import ipdb
    # ipdb.set_trace()
    output=F.softmax(output1["prob_map_boundary"],dim=1).cpu().numpy()
    # print("output_shape:",output.shape)
    # sys.exit()
    # inp = img_utils.unnormalize_img(batch['inp'][0][15 // 2, :, :], mean[:, :, 15 // 2], std[:, :, 15 // 2])
    output_test = output[0]
    inner = output_test[1]
    inner_f = np.zeros([96, 96])
    inner_f[inner >= 0.5] = 1
    outer_f = np.zeros([96, 96])
    outer = output_test[2]
    outer_f[outer >= 0.5] = 1
    path = batch["path"][0].split('/')[5] + '+' + batch["path"][0].split('/')[6] + '+' + \
           batch["path"][0].split('/')[9]

    # ipdb.set_trace()

    ex_ellipse = []

    for i in range(len(output)):
        ex_cls_new = []
        cls_new=[]
        for j in range(1, 3):


            ex= np.argwhere(output[i][j] >= 0.5)
            if len(ex)<5:
                continue
            cls_new.append(j-1)

            bound = output[i][j]
            bound_f = np.zeros([96, 96])
            bound_f[bound >= 0.05] = 1
            # if j==1:
            #     bound_f[bound >= 0.1] = 1

            num_points = 45
            ellipse   =   cv2.fitEllipse (ex)
            import math
            if math.isnan(ellipse[ 0 ][ 0 ]) or math.isnan(ellipse[ 0 ][ 1 ]) or math.isnan(ellipse[ 1 ][ 0 ]) or math.isnan(ellipse[ 1 ][ 1 ]) or math.isnan(ellipse[ 2 ]):
                continue
            # ellipse_points   =   cv2.ellipse2Poly ( ( int ( ellipse[ 0 ][ 0 ]),   int ( ellipse[ 0 ][ 1 ])),   ( int ( ellipse[ 1 ][ 0 ] / 2 ),   int ( ellipse[ 1 ][ 1 ] / 2 )),   int ( ellipse[ 2 ]),   0 ,   360 ,   num_points )
            ellipse_points_copy =   cv2.ellipse2Poly ( ( int ( ellipse[ 0 ][ 0 ]),   int ( ellipse[ 0 ][ 1 ])),   ( int ( ellipse[ 1 ][ 0 ] / 2 ),   int ( ellipse[ 1 ][ 1 ] / 2 )),   int ( ellipse[ 2 ]),   0 ,   360 ,   num_points )
            ellipse_points_new = update_ellipse(bound_f,ellipse_points_copy, ellipse[0],ex)
            if len(ellipse_points_new)<9:
            #     repeat last element to make it 9
                for k in range(9-len(ellipse_points_new)):
                    ellipse_points_new = np.vstack((ellipse_points_new,ellipse_points_new[-1]))
                # ex_ellipse.append(ellipse_points_new)
            # print("ellipse_points_new:",ellipse_points_new)
            # print("ellipse_points_new:shape",ellipse_points_new.shape)
            ex_cls_new.append(ellipse_points_new)
            # fig, ax = plt.subplots( 2, figsize=(20, 10))
            # ax[0].imshow(bound , cmap='gray')
            # ax[0].scatter(ellipse_points[:, 1], ellipse_points[:, 0], c='b', marker='o',linewidths=1)
            # ax[1].imshow(bound , cmap='gray')
            # ax[1].scatter(ellipse_points_new[:, 1], ellipse_points_new[:, 0], c='b', marker='o',linewidths=1)
            # plt.savefig('ellipse/ellipse_{}_{}.png'.format(path,j))

    ex_cls_new = torch.Tensor(ex_cls_new)
    ex_ellipse.append(ex_cls_new)
    # print("ex_cls_new.shape:",ex_cls_new.shape)

    # ex_8=torch.Tensor(ex_8).squeeze(2)
    # temp = torch.Tensor(ex_8)
    # print("ex_8:", ex_8)
    # print("ex_8_shape:", temp.shape)
    if not len(ex_cls_new):
        output1.update({'check':False})
        return ex_ellipse,ex_cls_new




    octagon = torch.stack(ex_ellipse, dim=2).view(len(ex_ellipse), ex_cls_new.size(0), 9, 2)
    return octagon,cls_new




def get_octogan(output1,threshold=0.5,batch=None):
    import matplotlib.pyplot as plt
    import numpy as np
    output=F.softmax(output1["prob_map_boundary"],dim=1).cpu().numpy()
    # print("output_shape:",output.shape)
    # sys.exit()
    # inp = img_utils.unnormalize_img(batch['inp'][0][15 // 2, :, :], mean[:, :, 15 // 2], std[:, :, 15 // 2])
    output_test = output[0]
    inner = output_test[1]
    inner_f = np.zeros([96, 96])
    inner_f[inner >= 0.5] = 1
    outer_f = np.zeros([96, 96])
    outer = output_test[2]
    outer_f[outer >= 0.5] = 1

    path = batch["path"][0].split('/')[5] + '+' + batch["path"][0].split('/')[6] + '+' + \
           batch["path"][0].split('/')[9]





    # for i in range(len(output)):
    #     for j in range(1, 3):
    #
    #
    #         ex= np.argwhere(output[i][j] >= 0.5)
    #         bound = output[i][j]
    #         bound_f = np.zeros([96, 96])
    #         bound_f[bound >= 0.5] = 1
    #         num_points = 45
    #         ellipse   =   cv2.fitEllipse (ex)
    #         ellipse_points   =   cv2.ellipse2Poly ( ( int ( ellipse[ 0 ][ 0 ]),   int ( ellipse[ 0 ][ 1 ])),   ( int ( ellipse[ 1 ][ 0 ] / 2 ),   int ( ellipse[ 1 ][ 1 ] / 2 )),   int ( ellipse[ 2 ]),   0 ,   360 ,   num_points )
    #         fig, ax = plt.subplots( 2, figsize=(20, 10))
    #         ax[0].imshow(bound , cmap='gray')
    #         ax[0].scatter(ellipse_points[:, 1], ellipse_points[:, 0], c='b', marker='o')
    #         plt.savefig('ellipse/ellipse_{}_{}.png'.format(path,j))



    # sys.exit()

    # fit ellipse and then get ellipse points


    # fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    # fig.tight_layout()


    # ax[0, 1].imshow(inner_f , cmap='gray')
    # ax[0, 0].imshow(inner , cmap='gray')
    # ax[1, 1].imshow(outer_f , cmap='gray')
    # ax[1, 0].imshow(outer , cmap='gray')
    # plt.savefig("demo_prob/{}.png".format(path))
    # plt.close("all")
    # np.savetxt("demo_prob/inner_{}.txt".format(path),inner*255,fmt='%d')
    # np.savetxt("demo_prob/outer_{}.txt".format(path), outer*255,fmt='%d')
    # sys.exit()


    for i in range(len(output)):
        ex_4=[]
        ex_cls = []
        cls=[]

        for j in range(1, 3):


            ex= np.argwhere(output[i][j] >= 0.5)
            if not len(ex):
                continue
            cls.append(j-1)

            if j ==1:
                # print("sum=", int(np.sum(inner_f)), "len_ex:", len(ex))
                # ax[0, 0].imshow(inp, cmap='gray')

                # ax[1, 0].imshow(inp, cmap='gray')


                # print(path)


                assert int(np.sum(inner_f))==len(ex)
                # print("sum=",int(np.sum(inner_f)),"len_ex:",len(ex))
            if j ==2:
                # print("sum=", int(np.sum(outer_f)), "len_ex:", len(ex))
                assert int(np.sum(outer_f))==len(ex)



            # print("ex:",ex)
            # print("ex.shape:",ex.shape)
            y_min, x_min = np.min(ex[:, 0]), np.min(ex[:, 1])
            y_max, x_max = np.max(ex[:, 0]), np.max(ex[:, 1])
            coord_ymin=np.argwhere(ex[:,0]==y_min)
            coord_xmin=np.argwhere(ex[:,1]==x_min)
            coord_ymax = np.argwhere(ex[:, 0] == y_max)
            coord_xmax = np.argwhere(ex[:, 1] == x_max)

            # print("x_min, y_min:", x_min, y_min)
            # print("x_max, y_max:", x_max, y_max)
            # print("coord_xmin:",coord_xmin )
            # print("coord_ymin:", ex[coord_ymin[len(coord_ymin)//2]])
            # print("coord_xmin:", ex[coord_xmin[len(coord_xmin) // 2]])
            # print("coord_ymax:", ex[coord_ymax[len(coord_ymax) // 2]])
            # print("coord_xmax:", ex[coord_xmax[len(coord_xmax) // 2]])
            # indx_ymin = torch.Tensor(ex[coord_ymin[len(coord_ymin)//2]]).squeeze(0)
            # indx_xmin =torch.Tensor(ex[coord_xmin[len(coord_xmin) // 2]]).squeeze(0)
            # indx_ymax =torch.Tensor(ex[coord_ymax[len(coord_ymax) // 2]]).squeeze(0)
            # indx_xmax =torch.Tensor(ex[coord_xmax[len(coord_xmax) // 2]]).squeeze(0)
            indx_ymin = ex[coord_ymin[len(coord_ymin) // 2]].squeeze(0)
            indx_xmin = ex[coord_xmin[len(coord_xmin) // 2]].squeeze(0)
            indx_ymax = ex[coord_ymax[len(coord_ymax) // 2]].squeeze(0)
            indx_xmax = ex[coord_xmax[len(coord_xmax) // 2]].squeeze(0)
            # print("coord_ymin:", indx_ymin)
            # print("coord_xmin:", indx_xmin)
            # print("coord_ymax:", indx_ymax)
            # print("coord_xmax:", indx_xmax)
            # ex.shape: (84, 2)
            # x_min, y_min: 36 43
            # x_max, y_max: 57 61
            # coord_xmin: [[0]
            #  [1]]
            # coord_ymin: [[37 43]]
            # coord_xmin: [[36 52]]
            # coord_ymax: [[45 61]]
            # coord_xmax: [[57 59]]
            # coord_ymin: tensor([37., 43.])
            # coord_xmin: tensor([36., 52.])
            # coord_ymax: tensor([45., 61.])
            # coord_xmax: tensor([57., 59.])
            # sys.exit()
            ex_cls.append([[indx_ymin[1],indx_ymin[0]],[indx_xmin[1],indx_xmin[0]],[indx_ymax[1],indx_ymax[0]],[indx_xmax[1],indx_xmax[0]]])
        # ex_4=np.array(ex_4)
        if not len(ex_cls):
            output1.update({'check':False})
            return ex_cls,ex_cls
        ex_4.append(ex_cls)
        ex_4=torch.Tensor(ex_4)
        # print("ex_4:", ex_4)
        # print("ex_4_shape:", ex_4.shape)
        octagon=snake_decode.get_octagon(ex_4)
        # print("octagon_shape:", octagon.shape)

    return octagon,cls
        #
        # print("octagon:", octagon)
        # sys.exit()

def Gaussian_Distribution(N=2, M=1000, mean=0, cov=1):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差

    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''


    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)

    return data, Gaussian


def get_octogan_ND(output1,threshold=0.5,batch=None):
    import matplotlib.pyplot as plt
    import numpy as np

    output=F.softmax(output1["prob_map"],dim=1).cpu().numpy()
    # print("output_shape:",output.shape)
    # sys.exit()
    # inp = img_utils.unnormalize_img(batch['inp'][0][15 // 2, :, :], mean[:, :, 15 // 2], std[:, :, 15 // 2])
    output_test = output[0]
    inner = output_test[1]
    inner_f = np.zeros([96, 96])
    inner_f[inner >= 0.5] = 1
    outer_f = np.zeros([96, 96])
    outer = output_test[2]
    outer_f[outer >= 0.5] = 1






    # fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    # fig.tight_layout()
    # path = batch["path"][0].split('/')[5] + '+' + batch["path"][0].split('/')[6] + '+' + \
    #        batch["path"][0].split('/')[9]

    # ax[0, 1].imshow(inner_f , cmap='gray')
    # ax[0, 0].imshow(inner , cmap='gray')
    # ax[1, 1].imshow(outer_f , cmap='gray')
    # ax[1, 0].imshow(outer , cmap='gray')
    # plt.savefig("demo_prob/{}.png".format(path))
    # plt.close("all")
    # np.savetxt("demo_prob/inner_{}.txt".format(path),inner*255,fmt='%d')
    # np.savetxt("demo_prob/outer_{}.txt".format(path), outer*255,fmt='%d')
    # sys.exit()


    for i in range(len(output)):
        ex_4=[]
        ex_cls = []
        cls=[]

        for j in range(1, 3):


            ex_f= np.argwhere(output[i][j] >= 0.5)
            if not len(ex_f):
                continue


            path = batch["path"][0].split('/')[5] + '+' + batch["path"][0].split('/')[6] + '+' + \
                   batch["path"][0].split('/')[9]

            M = 1000
            mean =  np.mean(ex_f,axis=0)
            cov = np.cov(ex_f.T)
            if  np.any(np.isnan(cov)):
                continue
            if j ==1:
                output1.update({'mean_0': mean})
                output1.update({'cov_0': cov})
            if j ==2:
                output1.update({'mean_1': mean})
                output1.update({'cov_1': cov})

            # print("mean:", mean)
            # print("cov:", cov)
            # data, Gaussian = Gaussian_Distribution(N=2, mean=mean, cov=cov)
            # # 生成二维网格平面
            # X, Y = np.meshgrid(np.linspace(-1,1,M), np.linspace(-1,1,M))
            # # 二维坐标数据
            # d = np.dstack([X,Y])
            # # 计算二维联合高斯概率
            # Z = Gaussian.pdf(d).reshape(M,M)
            # plt.figure()
            # plt.xlabel("X")
            # plt.ylabel("Y")
            # x, y = data.T
            # x_0,y_0 = ex_f.T
            # plt.plot(x, y, 'ko', alpha=0.3)
            # plt.plot(x_0, y_0, 'bo')
            # plt.contour(X, Y, Z,  alpha =1.0, zorder=10)


            # plt.savefig("test_gussian/{}_{}.png".format(path,j))

            # sys.exit()
            # print("ex_f:",ex_f)

            center = np.mean(ex_f,axis=0)
            # print("center:",center)
            # print("len(ex)]:",len(ex))
            dists = np.zeros(len(ex_f))
            # print("dists:",dists)
            for k in range(len(ex_f)):
                dist = distance.euclidean(ex_f[k],center)
                # print("dist:",dist)
                dists[k]=dist
            dists = np.sort(dists)
            # print("dists[0:len(dists)/3]:",dists[0:len(dists)//3])
            # print("dists[-len(dists)/3:-1]:",dists[-len(dists)//3:])
            dis_short = np.mean(dists[0:len(dists)//3])
            dis_long = np.mean(dists[-len(dists)//3:])
            ex=np.zeros_like(ex_f)
            count =0
            for point in ex_f:
                dist = distance.euclidean(point,center)
                if dist >= dis_short and dist <= dis_long:
                    ex[count][0]=point[0]
                    ex[count][1]=point[1]
                    count+=1
                    continue
                if dist < dis_short:
                    sample = np.abs(np.random.normal(0,dis_short/40,1))
                    if sample > dis_short-dist:
                        ex[count][0]=point[0]
                        ex[count][1]=point[1]
                        count+=1
                        continue
                if dist > dis_long:
                    sample = np.abs(np.random.normal(0,dis_long/40,1))
                    if sample > dist-dis_long:
                        ex[count][0]=point[0]
                        ex[count][1]=point[1]
                        count+=1
                        continue

            # print("count:",count)
            # print("ex:",ex)
            ex = ex[:count]
            # print("ex:",ex)
            dists_f = np.zeros(len(ex))
            # print("dists:",dists)
            for k in range(len(ex)):
                dist = distance.euclidean(ex[k],center)
                # print("dist:",dist)
                dists_f[k]=dist
            dists_f = np.sort(dists_f)
            # print("dists_f:",dists_f)
            #
            #
            #
            #
            #
            # print("dis_short:",dis_short)
            # print("dis_long:",dis_long)
            # print("dists:",dists)
            # sys.exit()



            if not len(ex):
                continue
            cls.append(j-1)

            # if j ==1:
            #     # print("sum=", int(np.sum(inner_f)), "len_ex:", len(ex))
            #     # ax[0, 0].imshow(inp, cmap='gray')
            #
            #     # ax[1, 0].imshow(inp, cmap='gray')
            #
            #
            #     # print(path)
            #
            #
            #     assert int(np.sum(inner_f))==len(ex)
            #     # print("sum=",int(np.sum(inner_f)),"len_ex:",len(ex))
            # if j ==2:
            #     # print("sum=", int(np.sum(outer_f)), "len_ex:", len(ex))
            #     assert int(np.sum(outer_f))==len(ex)



            # print("ex:",ex)
            # print("ex.shape:",ex.shape)
            y_min, x_min = np.min(ex[:, 0]), np.min(ex[:, 1])
            y_max, x_max = np.max(ex[:, 0]), np.max(ex[:, 1])
            coord_ymin=np.argwhere(ex[:,0]==y_min)
            coord_xmin=np.argwhere(ex[:,1]==x_min)
            coord_ymax = np.argwhere(ex[:, 0] == y_max)
            coord_xmax = np.argwhere(ex[:, 1] == x_max)

            # print("x_min, y_min:", x_min, y_min)
            # print("x_max, y_max:", x_max, y_max)
            # print("coord_xmin:",coord_xmin )
            # print("coord_ymin:", ex[coord_ymin[len(coord_ymin)//2]])
            # print("coord_xmin:", ex[coord_xmin[len(coord_xmin) // 2]])
            # print("coord_ymax:", ex[coord_ymax[len(coord_ymax) // 2]])
            # print("coord_xmax:", ex[coord_xmax[len(coord_xmax) // 2]])
            # indx_ymin = torch.Tensor(ex[coord_ymin[len(coord_ymin)//2]]).squeeze(0)
            # indx_xmin =torch.Tensor(ex[coord_xmin[len(coord_xmin) // 2]]).squeeze(0)
            # indx_ymax =torch.Tensor(ex[coord_ymax[len(coord_ymax) // 2]]).squeeze(0)
            # indx_xmax =torch.Tensor(ex[coord_xmax[len(coord_xmax) // 2]]).squeeze(0)
            indx_ymin = ex[coord_ymin[len(coord_ymin) // 2]].squeeze(0)
            indx_xmin = ex[coord_xmin[len(coord_xmin) // 2]].squeeze(0)
            indx_ymax = ex[coord_ymax[len(coord_ymax) // 2]].squeeze(0)
            indx_xmax = ex[coord_xmax[len(coord_xmax) // 2]].squeeze(0)
            # print("coord_ymin:", indx_ymin)
            # print("coord_xmin:", indx_xmin)
            # print("coord_ymax:", indx_ymax)
            # print("coord_xmax:", indx_xmax)
            # ex.shape: (84, 2)
            # x_min, y_min: 36 43
            # x_max, y_max: 57 61
            # coord_xmin: [[0]
            #  [1]]
            # coord_ymin: [[37 43]]
            # coord_xmin: [[36 52]]
            # coord_ymax: [[45 61]]
            # coord_xmax: [[57 59]]
            # coord_ymin: tensor([37., 43.])
            # coord_xmin: tensor([36., 52.])
            # coord_ymax: tensor([45., 61.])
            # coord_xmax: tensor([57., 59.])
            # sys.exit()
            ex_cls.append([[indx_ymin[1],indx_ymin[0]],[indx_xmin[1],indx_xmin[0]],[indx_ymax[1],indx_ymax[0]],[indx_xmax[1],indx_xmax[0]]])
        # ex_4=np.array(ex_4)
        # sys.exit()
        if not len(ex_cls):
            output1.update({'check':False})
            return ex_cls,ex_cls
        ex_4.append(ex_cls)
        ex_4=torch.Tensor(ex_4)
        # print("ex_4:", ex_4)
        # print("ex_4_shape:", ex_4.shape)
        octagon=snake_decode.get_octagon(ex_4)
        # print("octagon_shape:", octagon.shape)

        return octagon,cls
        #
        # print("octagon:", octagon)
        # sys.exit()

def get_8_points(output,threshold=0.6):
    output=F.softmax(output["prob_map"],dim=1).cpu().numpy()
    # print("output_shape:",output.shape)
    # sys.exit()
    for k in range(len(output)):
        cls = []
        ex_cls=[]
        ex_8 = []
        ex_8_list = []
        center_cls=[]
        point_0_indxs = []
        point_1_indxs = []
        point_2_indxs = []
        point_3_indxs = []
        point_4_indxs = []
        point_5_indxs = []
        point_6_indxs = []
        point_7_indxs = []
        point_8_indxs = []
        point_9_indxs = []
        point_10_indxs = []
        point_11_indxs = []
        exs=[]
        ex_cls_np = []

        for j in range(1, 3):
            # if j==2:
            #     with open("demo_img/demo_points.txt", 'a') as outfile:
            #
            #         outfile.write("---NEW Outer ---- \n")
            cls.append(j - 1)
            ex_p=[]
            ex_np = []

            ex= np.argwhere(output[k][j] >= threshold)
            if not len(ex):
                continue
            exs.append(ex)
            center = np.mean(ex,axis=0)
            # print("center:",center)

            arc=np.arctan2(ex[:,0]-center[0],ex[:,1]-center[1])*180 / np.pi
            # print("ex:",ex[:4])
            # print("arc:", arc[:4])
            point_0_indx = np.argwhere((arc<= -90) & (arc > -120))
            point_1_indx = np.argwhere((arc <= -120 )& (arc > -145))
            point_2_indx = np.argwhere((arc <= -150) & (arc > -180))
            point_3_indx = np.argwhere((arc <= 180) & (arc > 150))
            point_4_indx = np.argwhere((arc <=150) & (arc > 120))
            point_5_indx = np.argwhere((arc <= 120) & (arc >90))
            point_6_indx = np.argwhere((arc <= 90 )& (arc > 60))
            point_7_indx = np.argwhere((arc <= 60) & (arc > 30))
            point_8_indx = np.argwhere((arc <= 30) & (arc > 0))
            point_9_indx = np.argwhere((arc <= 0) & (arc > -30))
            point_10_indx = np.argwhere((arc <= -30) & (arc > -60))
            point_11_indx = np.argwhere((arc <= -60) & (arc > -90))
            point_0_indxs.append(point_0_indx)
            point_1_indxs.append(point_1_indx)
            point_2_indxs.append(point_2_indx)
            point_3_indxs.append(point_3_indx)
            point_4_indxs.append(point_4_indx)
            point_5_indxs.append(point_5_indx)
            point_6_indxs.append(point_6_indx)
            point_7_indxs.append(point_7_indx)
            point_8_indxs.append(point_8_indx)
            point_9_indxs.append(point_9_indx)
            point_10_indxs.append(point_10_indx)
            point_11_indxs.append(point_11_indx)
            # if j == 2:
            #     with open("demo_img/demo_points.txt", 'a') as outfile:
            #         for i in point_0_indx:
            #             np.savetxt(outfile, ex[int(i)], fmt='%4.1f')
            #             outfile.write("-------- new point------ \n")
            #         outfile.write("-------- point_0_indx------\n")





            num=0
            if len(point_0_indx):
                indx=int(point_0_indx[0])
                # if j == 2:
                #     with open("demo_img/demo_points.txt", 'a') as outfile:
                #         outfile.write("-------- first indx------ \n")
                #         np.savetxt(outfile, ex[indx], fmt='%4.1f')
                min=192*192
                for i in point_0_indx:
                    # print("indx:", int(i))
                    if (ex[int(i)][0]-center[0])**2+(ex[int(i)][1]-center[1])**2<=min:
                        min = (ex[int(i)][0]-center[0])**2+(ex[int(i)][1]-center[1])**2
                        indx = int(i)
                        # if j == 2:
                        #     with open("demo_img/demo_points.txt", 'a') as outfile:
                        #         outfile.write("-------- select point distance------ \n")
                        #         outfile.write("distance:%f"%min)
                        #         outfile.write("\n-------- select point ------ \n")
                        #         np.savetxt(outfile, ex[indx], fmt='%4.1f')
                        # print("min_0:",min)

                # if j == 2:
                #     with open("demo_img/demo_points.txt", 'a') as outfile:
                #         outfile.write("-------- Find Point------ \n")
                #         np.savetxt(outfile, ex[indx], fmt='%4.1f')
                #         outfile.write("-------- point_0_indx finish------\n")





                point_0=ex[indx]
                ex_np.append(point_0)
                # print("point_0_shape:", point_0.shape)
                point_0 = torch.Tensor(point_0).squeeze(0)
                # print("point_0:",point_0)
                # print("point_0_shape:", point_0.shape)
                ex_p.append(point_0[1])
                ex_p.append(point_0[0])
                num+=1
                # print("point_0:",point_0)
                # print("point_0_arc:", arc[indx])

            if len(point_1_indx):
                indx = int(point_1_indx[0])
                min = 192 * 192
                for i in point_1_indx:
                    # print("indx:",i)
                    if (ex[int(i)][0]-center[0])**2+(ex[int(i)][1]-center[1])**2<=min:
                        min = (ex[int(i)][0]-center[0])**2+(ex[int(i)][1]-center[1])**2
                        indx = int(i)
                        # print("min_1:", min)

                point_1 = ex[indx]
                ex_np.append(point_1)
                point_1 = torch.Tensor(point_1).squeeze(0)
                # print("point_1:",point_1)
                # print("point_1_arc:", arc[indx])
                ex_p.append(point_1[1])
                ex_p.append(point_1[0])
                num += 1

            if len(point_2_indx):
                indx = int(point_2_indx[0])
                min = 192 * 192
                for i in point_2_indx:
                    i=int(i)
                    if (ex[i][0] - center[0]) ** 2 + (ex[i][1] - center[1]) ** 2 <= min:
                        min = (ex[int(i)][0]-center[0])**2+(ex[int(i)][1]-center[1])**2
                        indx = i
                        # print("min_2:", min)

                point_2 = ex[indx]
                ex_np.append(point_2)
                point_2 = torch.Tensor(point_2).squeeze(0)
                ex_p.append(point_2[1])
                ex_p.append(point_2[0])
                num += 1
                # print("point_2:",point_2)
                # print("point_2_arc:", arc[indx])
            if len(point_3_indx):
                indx = int(point_3_indx[0])
                min = 192 * 192
                for i in point_3_indx:
                    i = int(i)
                    if (ex[i][0] - center[0]) ** 2 + (ex[i][1] - center[1]) ** 2 <= min:
                        min = (ex[int(i)][0]-center[0])**2+(ex[int(i)][1]-center[1])**2
                        indx = i
                        # print("min_3:", min)

                point_3 = ex[indx]
                ex_np.append(point_3)
                point_3 = torch.Tensor(point_3).squeeze(0)
                ex_p.append(point_3[1])
                ex_p.append(point_3[0])
                num += 1
                # print("point_3:",point_3)
                # print("point_3_arc:", arc[indx])
            if len(point_4_indx):
                indx =int( point_4_indx[0])
                min = 192 * 192
                for i in point_4_indx:
                    i = int(i)
                    if (ex[i][0] - center[0]) ** 2 + (ex[i][1] - center[1]) ** 2 <= min:
                        min = (ex[int(i)][0]-center[0])**2+(ex[int(i)][1]-center[1])**2
                        indx = i
                        # print("min_4:", min)

                point_4 = ex[indx]
                ex_np.append(point_4)
                point_4 = torch.Tensor(point_4).squeeze(0)
                ex_p.append(point_4[1])
                ex_p.append(point_4[0])
                num += 1
                # print("point_4:",point_4)
                # print("point_4_arc:", arc[indx])
            if len(point_5_indx):
                indx = int(point_5_indx[0])
                min = 192 * 192
                for i in point_5_indx:
                    i = int(i)
                    if (ex[i][0] - center[0]) ** 2 + (ex[i][1] - center[1]) ** 2 <= min:
                        min = (ex[int(i)][0]-center[0])**2+(ex[int(i)][1]-center[1])**2
                        indx = i
                        # print("min_5:", min)

                point_5 = ex[indx]
                ex_np.append(point_5)
                point_5 = torch.Tensor(point_5).squeeze(0)
                ex_p.append(point_5[1])
                ex_p.append(point_5[0])
                num += 1
                # print("point_5:",point_5)
                # print("point_5_arc:", arc[indx])
            if len(point_6_indx):
                indx = int(point_6_indx[0])
                min = 192 * 192
                for i in point_6_indx:
                    i = int(i)
                    if (ex[i][0] - center[0]) ** 2 + (ex[i][1] - center[1]) ** 2 <= min:
                        min = (ex[int(i)][0]-center[0])**2+(ex[int(i)][1]-center[1])**2
                        indx = i
                        # print("min_6:", min)

                point_6 = ex[indx]
                ex_np.append(point_6)
                point_6 = torch.Tensor(point_6).squeeze(0)
                ex_p.append(point_6[1])
                ex_p.append(point_6[0])
                num += 1
                # print("point_6:",point_6)
                # print("point_6_arc:", arc[indx])
            if len(point_7_indx):
                indx = int(point_7_indx[0])
                min = 192 * 192
                for i in point_7_indx:
                    i = int(i)
                    if (ex[i][0] - center[0]) ** 2 + (ex[i][1] - center[1]) ** 2 <= min:
                        min = (ex[int(i)][0]-center[0])**2+(ex[int(i)][1]-center[1])**2
                        indx = i
                        # print("min_7:", min)

                point_7 = ex[indx]
                ex_np.append(point_7)
                point_7 = torch.Tensor(point_7).squeeze(0)
                ex_p.append(point_7[1])
                ex_p.append(point_7[0])
                num += 1
            if len(point_8_indx):
                indx = int(point_8_indx[0])
                min = 192 * 192
                for i in point_8_indx:
                    i = int(i)
                    if (ex[i][0] - center[0]) ** 2 + (ex[i][1] - center[1]) ** 2 <= min:
                        min = (ex[int(i)][0] - center[0]) ** 2 + (ex[int(i)][1] - center[1]) ** 2
                        indx = i
                            # print("min_7:", min)

                point_8 = ex[indx]
                ex_np.append(point_8)
                point_8 = torch.Tensor(point_8).squeeze(0)
                ex_p.append(point_8[1])
                ex_p.append(point_8[0])
                num += 1

            if len(point_9_indx):
                indx = int(point_9_indx[0])
                min = 192 * 192
                for i in point_9_indx:
                    i = int(i)
                    if (ex[i][0] - center[0]) ** 2 + (ex[i][1] - center[1]) ** 2 <= min:
                        min = (ex[int(i)][0] - center[0]) ** 2 + (ex[int(i)][1] - center[1]) ** 2
                        indx = i
                            # print("min_7:", min)

                point_9 = ex[indx]
                ex_np.append(point_9)
                point_9 = torch.Tensor(point_9).squeeze(0)
                ex_p.append(point_9[1])
                ex_p.append(point_9[0])
                num += 1

            if len(point_10_indx):
                indx = int(point_10_indx[0])
                min = 192 * 192
                for i in point_10_indx:
                    i = int(i)
                    if (ex[i][0] - center[0]) ** 2 + (ex[i][1] - center[1]) ** 2 <= min:
                        min = (ex[int(i)][0] - center[0]) ** 2 + (ex[int(i)][1] - center[1]) ** 2
                        indx = i
                            # print("min_7:", min)

                point_10 = ex[indx]
                ex_np.append(point_10)
                point_10 = torch.Tensor(point_10).squeeze(0)
                ex_p.append(point_10[1])
                ex_p.append(point_10[0])
                num += 1

            if len(point_11_indx):
                indx = int(point_11_indx[0])
                min = 192 * 192
                for i in point_11_indx:
                    i = int(i)
                    if (ex[i][0] - center[0]) ** 2 + (ex[i][1] - center[1]) ** 2 <= min:
                        min = (ex[int(i)][0] - center[0]) ** 2 + (ex[int(i)][1] - center[1]) ** 2
                        indx = i
                            # print("min_7:", min)

                point_11 = ex[indx]
                ex_np.append(point_11)
                point_11 = torch.Tensor(point_11).squeeze(0)
                ex_p.append(point_11[1])
                ex_p.append(point_11[0])
                num += 1
            # print("number_of_points:",num)
            # print("mean:", torch.mean(torch.Tensor([ex_p[0],ex_p[-2]])))
            # print("torch:", ex_p[0])
            # print("ex_np[0]:", ex_np[0])
            # print("(ex_np[0]+ex_np[-1])/2:", (ex_np[0]+ex_np[-1])/2)

            for i in range(12 - num):
                temp_x = torch.mean(torch.Tensor([ex_p[0],ex_p[-2]]))
                temp_y = torch.mean(torch.Tensor([ex_p[1],ex_p[-1]]))
                ex_p.append(temp_x)
                ex_p.append(temp_y)
                ex_np.append((ex_np[0]+ex_np[-1])/2)
                # print("point_7:",point_7)
                # print("point_7_arc:", arc[indx])
            # print("ex_p:",ex_p)

            # sys.exit()
            # print("ex.shape:",ex.shape)
            # y_min, x_min = np.min(ex[:, 0]), np.min(ex[:, 1])
            # y_max, x_max = np.max(ex[:, 0]), np.max(ex[:, 1])
            # coord_ymin=np.argwhere(ex[:,0]==y_min)
            # coord_xmin=np.argwhere(ex[:,1]==x_min)
            # coord_ymax = np.argwhere(ex[:, 0] == y_max)
            # coord_xmax = np.argwhere(ex[:, 1] == x_max)

            # print("x_min, y_min:", x_min, y_min)
            # print("x_max, y_max:", x_max, y_max)
            # print("coord_xmin:",coord_xmin )
            # print("coord_ymin:", ex[coord_ymin[len(coord_ymin)//2]])
            # print("coord_xmin:", ex[coord_xmin[len(coord_xmin) // 2]])
            # print("coord_ymax:", ex[coord_ymax[len(coord_ymax) // 2]])
            # print("coord_xmax:", ex[coord_xmax[len(coord_xmax) // 2]])
            # indx_ymin = torch.Tensor(ex[coord_ymin[len(coord_ymin)//2]]).squeeze(0)
            # indx_xmin =torch.Tensor(ex[coord_xmin[len(coord_xmin) // 2]]).squeeze(0)
            # indx_ymax =torch.Tensor(ex[coord_ymax[len(coord_ymax) // 2]]).squeeze(0)
            # indx_xmax =torch.Tensor(ex[coord_xmax[len(coord_xmax) // 2]]).squeeze(0)
            # indx_ymin = ex[coord_ymin[len(coord_ymin) // 2]].squeeze(0)
            # indx_xmin = ex[coord_xmin[len(coord_xmin) // 2]].squeeze(0)
            # indx_ymax = ex[coord_ymax[len(coord_ymax) // 2]].squeeze(0)
            # indx_xmax = ex[coord_xmax[len(coord_xmax) // 2]].squeeze(0)
            # print("coord_ymin:", indx_ymin)
            # print("coord_xmin:", indx_xmin)
            # print("coord_ymax:", indx_ymax)
            # print("coord_xmax:", indx_xmax)
            # ex.shape: (84, 2)
            # x_min, y_min: 36 43
            # x_max, y_max: 57 61
            # coord_xmin: [[0]
            #  [1]]
            # coord_ymin: [[37 43]]
            # coord_xmin: [[36 52]]
            # coord_ymax: [[45 61]]
            # coord_xmax: [[57 59]]
            # coord_ymin: tensor([37., 43.])
            # coord_xmin: tensor([36., 52.])
            # coord_ymax: tensor([45., 61.])
            # coord_xmax: tensor([57., 59.])
            # sys.exit()
            ex_cls.append(ex_p)
            ex_cls_np.append(ex_np)
            center_cls.append(center)
        # ex_4=np.array(ex_4)
        ex_8_list.append(ex_cls_np)
        ex_cls = torch.Tensor(ex_cls)
        ex_8.append(ex_cls)

        # ex_8=torch.Tensor(ex_8).squeeze(2)
        # temp = torch.Tensor(ex_8)
        # print("ex_8:", ex_8)
        # print("ex_8_shape:", temp.shape)


        octagon = torch.stack(ex_8, dim=2).view(len(ex_8), 2, 12, 2)
        # print(octagon)
        # print("octagon:",octagon.shape)
        # sys.exit()
        # octagon=snake_decode.get_octagon(ex_4)
        # print("octagon_shape:", octagon.shape)
        return octagon, center_cls,exs,  point_0_indxs,  point_1_indxs,  point_2_indxs,  point_3_indxs,\
               point_4_indxs,  point_5_indxs,  point_6_indxs, point_7_indxs, point_8_indxs, point_9_indxs, point_10_indxs, point_11_indxs,ex_8_list,cls
        #
        # print("octagon:", octagon)
        # sys.exit()



def get_adj_mat(n_adj, n_nodes, device):
    a = np.zeros([n_nodes, n_nodes])

    for i in range(n_nodes):
        for j in range(-n_adj // 2, n_adj // 2 + 1):
            if j != 0:
                a[i][(i + j) % n_nodes] = 1
                a[(i + j) % n_nodes][i] = 1

    a = torch.Tensor(a.astype(np.float32))
    return a.to(device)


def get_adj_ind(n_adj, n_nodes, device):
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0])
    ind = (torch.arange(n_nodes)[:, None] + ind[None]) % n_nodes
    return ind.to(device)


def get_pconv_ind(n_adj, n_nodes, device):
    n_outer_nodes = snake_config.poly_num
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1)])
    outer_ind = (torch.arange(n_outer_nodes)[:, None] + ind[None]) % n_outer_nodes
    inner_ind = outer_ind + n_outer_nodes
    ind = torch.cat([outer_ind, inner_ind], dim=1)
    return ind


def img_poly_to_can_poly(img_poly):
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    # x_max = torch.max(img_poly[..., 0], dim=-1)[0]
    # y_max = torch.max(img_poly[..., 1], dim=-1)[0]
    # h, w = y_max - y_min + 1, x_max - x_min + 1
    # long_side = torch.max(h, w)
    # can_poly = can_poly / long_side[..., None, None]
    return can_poly


def uniform_upsample(poly, p_num):
    # 1. assign point number for each edge
    # 2. calculate the coefficient for linear interpolation
    next_poly = torch.roll(poly, -1, 2)
    edge_len = (next_poly - poly).pow(2).sum(3).sqrt()
    edge_num = torch.round(edge_len * p_num / torch.sum(edge_len, dim=2)[..., None]).long()
    edge_num = torch.clamp(edge_num, min=1)
    edge_num_sum = torch.sum(edge_num, dim=2)
    edge_idx_sort = torch.argsort(edge_num, dim=2, descending=True)
    extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
    edge_num_sum = torch.sum(edge_num, dim=2)
    assert torch.all(edge_num_sum == p_num)

    edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
    weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
    poly1 = poly.gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly2 = poly.gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly = poly1 * (1 - weight) + poly2 * weight

    return poly


def zoom_poly(poly, scale):
    mean = (poly.min(dim=1, keepdim=True)[0] + poly.max(dim=1, keepdim=True)[0]) * 0.5
    poly = poly - mean
    poly = poly * scale + mean
    return poly

