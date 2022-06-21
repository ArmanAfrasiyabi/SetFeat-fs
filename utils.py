from einops import rearrange, repeat
from torch import nn
import numpy as np
import torch
import math
# from models.utils import *
# from dataloader.data_utils import *
from torch.autograd import Variable
import torch.nn.functional as F
import pprint
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random
import os
from dataloader.ml_dataFunctions import SetDataManager


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(item + '\n')
    f.close()


def fun_metaLoader(args, normalization, n_eposide=400, file_name='/val.json'):
    val_file = args.data_dir + args.dataset + file_name
    val_datamgr = SetDataManager(args, normalization, args.way,  args.shot, args.query, n_eposide)
    return val_datamgr.get_data_loader(val_file)


def set_seed(seed):
    if seed == 0:
        torch.backends.cudnn.benchmark = True
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def dataextractor(batch, ys, yq, num_class, shot):
    if torch.cuda.is_available():
        data, _ = [_.cuda() for _ in batch]
    else:
        data, _ = batch
    xs_, xq_ = data[:num_class * shot], data[num_class * shot:]
    for w in range(num_class):
        if w == 0:
            xs = xs_[ys == w]
            xq = xq_[yq == w]
        else:
            xs = torch.cat((xs, xs_[ys == w]), dim=0)
            xq = torch.cat((xq, xq_[yq == w]), dim=0)
    return torch.cat((xs, xq), dim=0)

 
