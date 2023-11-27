import torch
from lib.utils.optimizer.radam import RAdam
import sys


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}


def make_optimizer(cfg, net,epoch):
    # print("sgd")
    import ipdb
    params = []
    lr = cfg.train.lr
    weight_decay = cfg.train.weight_decay

    for key, value in net.named_parameters():
        # print("key:",key,"value:",value,"value.requires_grad:",value.requires_grad)

        # if 'hybrid_cls' in key:
        #     # print("find_hybrid_cls")
        #     continue
        # if epoch >20 and 'hybrid_cls' in key:
        #     continue
        # if 'hybrid_cls' in key:
        #     continue

        if epoch > 50:
            lr = 0.00001
        # if epoch > 200:
        #     lr = 0.000001

        # p05int("key:",key)
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    # sys.exit()

    if 'adam' in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay)
    # if 'sgd' in cfg.train.optim:
    #     optimizer = _optimizer_factory[cfg.train.optim](params, lr=lr, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)
    # optimizer = _optimizer_factory[cfg.train.optim](params, lr=lr, momentum=0.9, weight_decay=0.0005)

    return optimizer
