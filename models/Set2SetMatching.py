from einops import rearrange, repeat
from torch import nn 
import torch
import numpy as np
from models.SetFeat12 import SetFeat12
from models.SetFeat4 import SetFeat4
from torch.autograd import Variable
import torch.nn.functional as F
# from models.model_utils import *
from torchvision.utils import save_image
from utils import *


def network_set(args):
    n_heads_list = [1, 2, 3, 4]
    if args.backbone == 'SetFeat4' or args.backbone == 'SetFeat4_512':
        if args.backbone == 'SetFeat4':
            args.sqa_type = 'linear'
            n_filters = [64, 64, 64, 64]
        elif args.backbone == 'SetFeat4_512':
            args.sqa_type = 'convolution'
            # [96, 128, 256, 512]  # of parameters: 1.591M   # https://arxiv.org/pdf/1906.05186.pdf
            n_filters = [96, 128, 160, 200]       # [96, 128, 160, 200] of parameters: 1.583M
        n_heads = np.sum(n_heads_list)
        enc_out_chanal = n_filters[-1]
        encoder = SetFeat4(n_filters, n_heads_list, enc_out_chanal, args.sqa_type)
    elif args.backbone in ['SetFeat12', 'SetFeat12_11M']:
        args.sqa_type = 'convolution'
        if args.backbone == 'SetFeat12':
            n_filters = [128, 150, 180, 512]  # [64, 160, 320, 640]
        elif args.backbone == 'SetFeat12_11M':
            n_filters = [128, 150, 196, 480]
        enc_out_chanal = n_filters[-1]
        n_heads = np.sum(n_heads_list)
        encoder = SetFeat12(n_heads_list, n_filters, enc_out_chanal, args.sqa_type)
    if args.model_status == 'pertrain':
        clf = []
        for _ in range(n_heads):
            clf.append(nn.Linear(enc_out_chanal, args.num_class))
        clf = nn.ModuleList(clf)
    else:
        clf = None
    # N = num_params(encoder.layer1) + num_params(encoder.layer2) + num_params(encoder.layer3) + num_params(
    #     encoder.layer4)
    # print('# of parameters in backbones: %.3fM' % N)
    # M = num_params(encoder.atten1) + num_params(encoder.atten2) + num_params(encoder.atten3) + num_params(
    #     encoder.atten4)
    # print('# of parameters in mappers: %.3fM' % M)
    # print('# of parameters in totall: %.3f M' % (M+N))
    return encoder, clf, n_heads, enc_out_chanal


def get_similiarity_map(proto, query, metric='cosine'):
    way = proto.shape[0]
    num_query = query.shape[0]
    query = query.view(query.shape[0], query.shape[1], -1)
    proto = proto.view(proto.shape[0], proto.shape[1], -1)

    proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
    query = query.unsqueeze(1).repeat([1, way, 1, 1])
    proto = proto.permute(0, 1, 3, 2)
    query = query.permute(0, 1, 3, 2)
    feature_size = proto.shape[-2]

    if metric == 'cosine':
        proto = proto.unsqueeze(-3)
        query = query.unsqueeze(-2)
        query = query.repeat(1, 1, 1, feature_size, 1)
        similarity_map = F.cosine_similarity(proto, query, dim=-1)
    if metric == 'l2':
        proto = proto.unsqueeze(-3)
        query = query.unsqueeze(-2)
        query = query.repeat(1, 1, 1, feature_size, 1)
        similarity_map = (proto - query).pow(2).sum(-1)
        similarity_map = 1 - similarity_map
    return similarity_map


def cub_data_fun(x, way, shot, query):
    x_support = x[:, :shot, :, :, :].contiguous()
    x_support = x_support.view(way * shot, *x.size()[2:])
    x_query = x[:, shot:, :, :, :].contiguous()
    x_query = x_query.view(way * query, *x.size()[2:])
    x_support = x_support.cuda()
    x_query = x_query.cuda()
    return torch.cat((x_support, x_query), dim=0)


class Set2SetMatching(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.matching_method = args.matching_method
        self.encoder, self.clf, n_heads, enc_out_chanal = network_set(args)
        self.n_heads = n_heads
        self.top_k = n_heads
        self.num_class = args.num_class
        self.dataset = args.dataset

    def pretrainforward(self, batch):
        x, y_ = [_.cuda() for _ in batch]
        y = y_
        z = self.encoder(x)
        logits = self.clf[0](z[:, 0, :])
        for h in range(1, self.n_heads):
            logits = torch.cat((logits, self.clf[h](z[:, h, :])), dim=0)
            y = torch.cat((y, y_), dim=0)
        loss = F.cross_entropy(logits, y)
        acc = count_acc(logits, y)
        return loss, acc

    def metatrainforward(self, batch, ys, yq, n_class, n_support, n_query, matching_method='SumMin'):
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        k = n_class * n_support
        target_inds = Variable(target_inds, requires_grad=False).cuda()  # [15, 5, 1]
        self.matching_method = matching_method
        if self.dataset in ['cub']:
            x = cub_data_fun(batch[0], n_class, n_support, n_query)
        elif self.dataset in ['miniimagenet', 'tieredimagenet']:
            x = dataextractor(batch, ys, yq, n_class, n_support)
        z = self.encoder(x)
        zs, zq = z[:k, :, :], z[k:, :, :]
        proto = zs.view(n_class, n_support, zs.shape[1], zs.shape[2]).mean(1)
        similarity_map = get_similiarity_map(proto.view(-1, proto.size(2)), zq.view(-1, zq.size(2)))
        similarity_map = similarity_map.view(n_class, n_query, self.n_heads, -1, self.n_heads)
        similarity_map, frequency_y = similarity_map.max(-1)
        similarity_map, indexes = torch.sort(similarity_map, dim=2)
        similarity_map = similarity_map[:, :, self.n_heads - self.top_k:, :].sum(2)
        similarity_map = similarity_map.view(-1, similarity_map.shape[2])
        log_p_y = F.log_softmax(similarity_map, dim=1).view(n_class, n_query, -1)
        loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        return loss, acc, frequency_y





