from lib.config import cfg, args
import numpy as np
import os
import cv2

def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch['inp'])
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    network = make_network(cfg).cuda()
    print(" cfg.model_dir:", cfg.model_dir)
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp)
        evaluator.evaluate(output, batch)
    evaluator.summarize()


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    import sys
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import matplotlib.mlab as mlab
    from scipy.stats import norm
    import ipdb

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    total_params = sum(p.numel() for p in network.parameters())
    print("Total number of parameters: ", total_params)
    # ipdb.set_trace()
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    id=0
    # for batch in tqdm.tqdm(data_loader):
    for batch in data_loader:
        # print("batch['inp']:",batch['inp'].shape)
        img=batch['inp'].numpy().squeeze(0)
        img = img[7]
        # print(batch["path"])
        path= batch["path"][0].split('/')[5]+'+'+batch["path"][0].split('/')[6]+'+'+batch["path"][0].split('/')[9]
        print(path)
        # if batch["path"][0].split('/')[5] != "S218928036_S1ff973cbbf73b8_20181024":
        #     continue
        # sys.exit()
        # print("img_shape:",img.shape)
        # cv2.imwrite("test_inp/{}.png".format(path),img)
        for k in batch:

            if k != 'meta' and k!= 'poly' and k!= 'path':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)

        id=id+1
        print("Test_%d:in length of %d"%(id,len(data_loader)))
        # print("Ok")


        haus,ct_cls,amount,ave_hdf_0,ave_hdf_1,_,_,_,_ = visualizer.visualize(output, batch,id)

    # np.savetxt("demo_img_dataall_3_dis_skip_bad/{}.txt".format("ct_cls"),np.array(ct_cls))
    # np.savetxt("demo_img_dataall_3_dis_skip_bad/{}.txt".format("amount"), np.array([amount]))

    hdf_indx = np.argsort(sort_hdf[0])
    print("hdf_indx_0:",hdf_indx[0])
    print("hdf_indx_-1:",hdf_indx[-1])
    print("hdf_indx_0_path:",sort_hdf[1][hdf_indx[0]])
    print("hdf_indx_-1_path:",sort_hdf[1][hdf_indx[-1]])

    print("ct_cls:",ct_cls)
    print("Amount:",amount)
    n, bins, patches = plt.hist(haus[0], "auto",density=True,alpha=0.5)

    mu, sigma = norm.fit(haus[0])
    y = norm.pdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.title('histogram of Huasdorff score for inner')
    plt.xlabel('Huasdorff score')
    plt.ylabel('Dense')
    plt.savefig("demo_img_dataall_3_dis_skip_bad/{}.png".format("hist_summary_inner"))
    plt.close('all')
    n, bins, patches = plt.hist(haus[1], "auto",density=True,alpha=0.5)

    mu, sigma = norm.fit(haus[1])
    y = norm.pdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.title('histogram of Huasdorff score for outer' )
    plt.xlabel('Huasdorff score')
    plt.ylabel('Dense')
    plt.savefig("demo_img_dataall_3_dis_skip_bad/{}.png".format("hist_summary_outer"))
    plt.close('all')

def run_boundary():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    id=0
    print("Buondary")
    # for batch in tqdm.tqdm(data_loader):
    for batch in data_loader:
        id = id + 1
        for k in batch:
            if k != 'meta' and k!= 'poly' and k!= 'path':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
            if not output['check']:
                continue


        print("sammple_%d:in length of %d"%(id,len(data_loader)))
        visualizer.boundary(output, batch,id)
def run_classification():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    id=0
    print("Classification")
    # for batch in tqdm.tqdm(data_loader):
    for batch in data_loader:
        id = id + 1
        for k in batch:
            if k != 'meta' and k!= 'poly' and k!= 'path':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)



        print("sammple_%d:in length of %d"%(id,len(data_loader)))
        visualizer.classification(output, batch,id)
def run_segmentation():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    id=0
    print("Segmentation")
    # for batch in tqdm.tqdm(data_loader):
    for batch in data_loader:
        id = id + 1
        for k in batch:
            if k != 'meta' and k!= 'poly' and k!= 'path':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)



        print("sammple_%d:in length of %d"%(id,len(data_loader)))
        visualizer.segmentation(output, batch,id)
def run_feature_save():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    id=0
    print("Feature and Risk Label")
    # for batch in tqdm.tqdm(data_loader):
    for batch in data_loader:
        for k in batch:
            if k != 'meta' and k != 'poly' and k != 'path':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)


        id=id+1
        print("sammple_%d:in length of %d"%(id,len(data_loader)))
        visualizer.feature_save(output, batch,id)
def run_mask_dis():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    # network = make_network(cfg).cuda()
    # load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    # network.eval()

    data_loader = make_data_loader(cfg, is_train=True)
    visualizer = make_visualizer(cfg)
    id=0
    print("Mask_dis_save")
    # for batch in tqdm.tqdm(data_loader):
    for batch in data_loader:

            # if k != 'meta' and k != 'poly' and k != 'path':
            #     batch[k] = batch[k].cuda()
        # with torch.no_grad():
        #     output = network(batch['inp'], batch)


        id=id+1
        print("sammple_%d:in length of %d"%(id,len(data_loader)))
        # visualizer.mask_dis_save(batch,id)

def run_sbd():
    from tools import convert_sbd
    convert_sbd.convert_sbd()


def run_demo():
    from tools import demo
    demo.demo()


if __name__ == '__main__':
    globals()['run_'+args.type]()
