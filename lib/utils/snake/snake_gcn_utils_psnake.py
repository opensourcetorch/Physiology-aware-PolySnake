import torch
import numpy as np
from lib.utils.snake import snake_decode, snake_config_psnake
from lib.csrc.extreme_utils import _ext as extreme_utils
from lib.utils import data_utils
import ipdb


def collect_training(poly, ct_01):
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0)
    return poly




def prepare_training_init(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}
    init.update({'i_it_4py': collect_training(batch['i_it_4py'], ct_01)})
    init.update({'c_it_4py': collect_training(batch['c_it_4py'], ct_01)})
    init.update({'i_gt_4py': collect_training(batch['i_gt_4py'], ct_01)})
    init.update({'c_gt_4py': collect_training(batch['c_gt_4py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})

    return init


# def prepare_testing_init(box, score):
    # i_it_4pys = snake_decode.get_init(box)
    # i_it_4pys = uniform_upsample(i_it_4pys, snake_config.init_poly_num)
    # c_it_4pys = img_poly_to_can_poly(i_it_4pys)

    # ind = score > snake_config.ct_score
    # i_it_4pys = i_it_4pys[ind]
    # c_it_4pys = c_it_4pys[ind]
    # ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(ind.size(0))], dim=0)
    # init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'ind': ind}

    # return init


def prepare_testing_init(score):
    #print('ind', score.shape)  # ind torch.Size([100])
    ind = score > snake_config_psnake.ct_score
    # print('snake_config.ct_score', snake_config.ct_score)  # snake_config.ct_score 0.05
    ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(ind.size(0))], dim=0)
    init = {'ind': ind}
    # print(ind)
    
    return init
    

def get_box_match_ind(pred_box, score, gt_poly):
    if gt_poly.size(0) == 0:
        return [], []

    gt_box = torch.cat([torch.min(gt_poly, dim=1)[0], torch.max(gt_poly, dim=1)[0]], dim=1)
    iou_matrix = data_utils.box_iou(pred_box, gt_box)
    iou, gt_ind = iou_matrix.max(dim=1)
    box_ind = ((iou > snake_config_psnake.box_iou) * (score > snake_config_psnake.confidence)).nonzero().view(-1)
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

    i_it_4py = uniform_upsample(i_it_4py, snake_config_psnake.init_poly_num)[0]
    c_it_4py = img_poly_to_can_poly(i_it_4py)
    i_gt_4py = torch.cat([batch['i_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    c_gt_4py = torch.cat([batch['c_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_4py = {'i_it_4py': i_it_4py, 'c_it_4py': c_it_4py, 'i_gt_4py': i_gt_4py, 'c_gt_4py': c_gt_4py}

    i_it_py = snake_decode.get_octagon(i_gt_4py[None])
    i_it_py = uniform_upsample(i_it_py, snake_config_psnake.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    i_gt_py = torch.cat([batch['i_gt_py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_py = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    ind = torch.cat([torch.full([len(gt_ind[i])], i) for i in range(batch_size)], dim=0)

    if snake_config_psnake.train_pred_box_only:
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

    init.update({'i_gt_py': collect_training(batch['i_gt_py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})

    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init


def prepare_training_evolve(ex, init):
    if not snake_config_psnake.train_pred_ex:
        evolve = {'i_it_py': init['i_it_py'], 'c_it_py': init['c_it_py'], 'i_gt_py': init['i_gt_py']}
        return evolve

    i_gt_py = init['i_gt_py']

    if snake_config_psnake.train_nearest_gt:
        shift = -(ex[:, :1] - i_gt_py).pow(2).sum(2).argmin(1)
        i_gt_py = extreme_utils.roll_array(i_gt_py, shift)

    i_it_py = snake_decode.get_octagon(ex[None])
    i_it_py = uniform_upsample(i_it_py, snake_config_psnake.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    evolve = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    return evolve


def prepare_testing_evolve(ex):
    if len(ex) == 0:
        i_it_pys = torch.zeros([0, snake_config_psnake.poly_num, 2]).to(ex)
        c_it_pys = torch.zeros_like(i_it_pys)
    else:
        #print(ex.shape)  # 1*4*2
        i_it_pys = snake_decode.get_octagon(ex[None])
        #print(i_it_pys.shape) # 1 * 1 * 12 * 2
        i_it_pys = uniform_upsample(i_it_pys, snake_config_psnake.poly_num)[0]
        #print(i_it_pys.shape)  # 1 * 128 * 2
        c_it_pys = img_poly_to_can_poly(i_it_pys)
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
    return evolve


def get_gcn_doublefeature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1
    # img_poly B,2,128,2
    #feature B 2,64,128

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1),img_poly.size(1), img_poly.size(2)]).to(img_poly.device)
    # gcn_feature B,2,64,128
    for i in range(batch_size):
        poly = img_poly[i].unsqueeze(0) # 1,2,128,2
        # ipdb.set_trace()
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly,)[0]#1,64,96,96 ; 1,2,128,2 =>64,2,128 =>64,2,128

        gcn_feature[i] = feature

    # ipdb.set_trace()

    return gcn_feature
def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        # ipdb.set_trace()
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly,)[0].permute(1, 0, 2)
        gcn_feature[ind == i] = feature

    # ipdb.set_trace()

    return gcn_feature


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
    n_outer_nodes = snake_config_psnake.poly_num
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1)])
    outer_ind = (torch.arange(n_outer_nodes)[:, None] + ind[None]) % n_outer_nodes
    inner_ind = outer_ind + n_outer_nodes
    ind = torch.cat([outer_ind, inner_ind], dim=1)
    return ind

def decode_prob_wh(output,prob,wh,batch):
    batchsize, _, height, width = output['prob_map_boundary'].size()
    prob_f = torch.zeros_like(prob)
    prob_f[prob>0.5] = 1
    ct_img_idx =[]
    ct_cls_ind=[]
    # ct=[]
    ct_x=[]
    ct_x_float=[]

    ct_y=[]
    ct_y_float=[]
    ct_01 = batch['ct_01'].byte()
    ct_ind = batch['ct_ind'][ct_01]
    # ct_x, ct_y = ct_ind % width//snake_config_psnake.ro, ct_ind // width//snake_config_psnake.ro
    for i in range(prob_f.shape[0]):

        for j in range(1,3):
            ex = torch.nonzero(prob_f[i,j]==1)
            ex = ex.to(torch.float32)

            if len(ex)>5:
                # x_min, y_min = np.min(ex[:, 0]), np.min(ex[:, 1])
                # x_max, y_max = np.max(ex[:, 0]), np.max(ex[:, 1])
                ct_img_idx.append(i)
                ct_cls_ind.append(j-1)
                # ipdb.set_trace()
                # ct.append((torch.mean(ex,dim=0)//snake_config_psnake.ro).cpu().numpy().astype(np.int32).tolist())
                # ct_x.append(torch.mean(ex[:, 1]//snake_config_psnake.ro, dim=0).cpu().numpy().astype(np.int32).tolist())
                x_ct = (torch.max(ex[:,1])+torch.min(ex[:,1]))/2//snake_config_psnake.ro
                y_ct = (torch.max(ex[:,0])+torch.min(ex[:,0]))/2//snake_config_psnake.ro
                ct_x.append(x_ct.cpu().numpy().astype(np.int32).tolist())
                ct_y.append(y_ct.cpu().numpy().astype(np.int32).tolist())

                # ct_y.append(torch.mean(ex[:, 0]//snake_config_psnake.ro, dim=0).cpu().numpy().astype(np.int32).tolist())
                # ct_x_float.append(torch.mean(ex[:, 1]/snake_config_psnake.ro, dim=0).cpu().numpy().astype(np.float32).tolist())
                # ct_y_float.append(torch.mean(ex[:, 0]/snake_config_psnake.ro, dim=0).cpu().numpy().astype(np.float32).tolist())





    # ipdb.set_trace()
    wh_pred = wh

    # ipdb.set_trace()
    if len(ct_x)>0:
        # ct_offset = wh_pred[ct_img_idx, ct_cls_ind,...,[ct[i][0] for i in range(len(ct))],[ct[i][1] for i in range(len(ct))]].view(len(ct), -1, 2)
        ct_offset = wh_pred[ct_img_idx, ct_cls_ind,...,ct_y,ct_x].view(len(ct_x), -1, 2)
    else:
        ct_offset = wh_pred[ct_img_idx, ct_cls_ind,..., ct_y, ct_x].view(len(ct_x), 1, 2)
    # if ct_x.size(0) == 0:
    #     ct_offset = wh_pred[ct_img_idx, ct_cls_ind,..., ct_y, ct_x].view(ct_x.size(0), 1,  2)
    # else:
    #     ct_offset = wh_pred[ct_img_idx, ct_cls_ind,..., ct_y, ct_x].view(ct_x.size(0), -1, 2)
    # ct_x, ct_y = ct_x[:, None].float(), ct_y[:, None].float()
    ct_x, ct_y = torch.tensor(ct_x[:]).to(ct_offset.device), torch.tensor(ct_y[:]).to(ct_offset.device)
    ct_x, ct_y = ct_x[:, None].float(), ct_y[:, None].float()
    # ipdb.set_trace()
    ct = torch.cat([ct_x, ct_y], dim=1)




    # ct = torch.tensor(ct).to(ct_offset.device).to(torch.float32)
    init_polys = ct_offset + ct.unsqueeze(1).expand(ct_offset.size(0), ct_offset.size(1), ct_offset.size(2))

    output.update({'poly_init': init_polys*snake_config_psnake.ro,'cls':ct_cls_ind})
    # if len(init_polys)==128:
    #     ipdb.set_trace()
    # ipdb.set_trace()
    return init_polys
def decode_prob_wh_double(output,prob,wh,batch):
    batchsize, _, height, width = output['prob_map_boundary'].size()
    prob_f = torch.zeros_like(prob)
    prob_f[prob>0.5] = 1
    ct_img_idx =[]
    ct_cls_ind=[]
    # ct=[]
    ct_x=[]
    ct_x_float=[]

    ct_y=[]
    ct_y_float=[]
    ct_01 = batch['ct_01'].byte()
    ct_ind = batch['ct_ind'][ct_01]
    # ct_x, ct_y = ct_ind % width//snake_config_psnake.ro, ct_ind // width//snake_config_psnake.ro
    for i in range(prob_f.shape[0]):

        for j in range(1,3):
            ex = torch.nonzero(prob_f[i,j]==1)
            ex = ex.to(torch.float32)

            if len(ex)>5:
                # x_min, y_min = np.min(ex[:, 0]), np.min(ex[:, 1])
                # x_max, y_max = np.max(ex[:, 0]), np.max(ex[:, 1])
                ct_img_idx.append(i)
                ct_cls_ind.append(j-1)
                # ipdb.set_trace()
                # ct.append((torch.mean(ex,dim=0)//snake_config_psnake.ro).cpu().numpy().astype(np.int32).tolist())
                # ct_x.append(torch.mean(ex[:, 1]//snake_config_psnake.ro, dim=0).cpu().numpy().astype(np.int32).tolist())
                x_ct = (torch.max(ex[:,1])+torch.min(ex[:,1]))/2//snake_config_psnake.ro
                y_ct = (torch.max(ex[:,0])+torch.min(ex[:,0]))/2//snake_config_psnake.ro
                ct_x.append(x_ct.cpu().numpy().astype(np.int32).tolist())
                ct_y.append(y_ct.cpu().numpy().astype(np.int32).tolist())


                # ct_y.append(torch.mean(ex[:, 0]//snake_config_psnake.ro, dim=0).cpu().numpy().astype(np.int32).tolist())
                # ct_x_float.append(torch.mean(ex[:, 1]/snake_config_psnake.ro, dim=0).cpu().numpy().astype(np.float32).tolist())
                # ct_y_float.append(torch.mean(ex[:, 0]/snake_config_psnake.ro, dim=0).cpu().numpy().astype(np.float32).tolist())





    # ipdb.set_trace()
    wh_pred = wh

    # ipdb.set_trace()
    if len(ct_x)>0:
        # ct_offset = wh_pred[ct_img_idx, ct_cls_ind,...,[ct[i][0] for i in range(len(ct))],[ct[i][1] for i in range(len(ct))]].view(len(ct), -1, 2)
        ct_offset = wh_pred[ct_img_idx, ct_cls_ind,...,ct_y,ct_x].view(len(ct_x), -1, 2)
    else:
        ct_offset = wh_pred[ct_img_idx, ct_cls_ind,..., ct_y, ct_x].view(len(ct_x), 1, 2)
    # if ct_x.size(0) == 0:
    #     ct_offset = wh_pred[ct_img_idx, ct_cls_ind,..., ct_y, ct_x].view(ct_x.size(0), 1,  2)
    # else:
    #     ct_offset = wh_pred[ct_img_idx, ct_cls_ind,..., ct_y, ct_x].view(ct_x.size(0), -1, 2)
    # ct_x, ct_y = ct_x[:, None].float(), ct_y[:, None].float()
    ct_x, ct_y = torch.tensor(ct_x[:]).to(ct_offset.device), torch.tensor(ct_y[:]).to(ct_offset.device)
    ct_x, ct_y = ct_x[:, None].float(), ct_y[:, None].float()
    # ipdb.set_trace()
    ct = torch.cat([ct_x, ct_y], dim=1)




    # ct = torch.tensor(ct).to(ct_offset.device).to(torch.float32)
    init_polys = ct_offset + ct.unsqueeze(1).expand(ct_offset.size(0), ct_offset.size(1), ct_offset.size(2))

    output.update({'poly_init': init_polys*snake_config_psnake.ro,'cls':ct_cls_ind})
    # if len(init_polys)==128:
    #     ipdb.set_trace()
    # ipdb.set_trace()
    return init_polys.view(1,-1,128,2)

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

