# import tqdm
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from models.Set2SetMatching import Set2SetMatching as Set2Set
from dataloader.tl_dataFunctions import train_DataLaoder
from args import pertrain_argparser
from utils import fun_metaLoader
from utils import *
import os.path as osp


def best_model_testing(args, shot):
    args.shot = shot
    set_seed(args.seed)
    model = Set2Set(args)
    model = model.cuda()
    args.save_path = '%s/%s-%s/' % (args.dataset, args.backbone, args.sqa_type)
    args.save_path = osp.join('./checkpoint', args.save_path)
    save_name = 'metatrain_' + args.matching_method
    checkpoint = torch.load(osp.join(args.save_path, save_name + '_best_model.pth'))
    model.encoder.load_state_dict(checkpoint['model'])
    _, normalization = train_DataLaoder(args)
    loader = fun_metaLoader(args, normalization, n_eposide=600, file_name='/test.json')
    if not args.random_val_task:
        loader = [x for x in loader]
    ave_acc = Averager()
    test_acc_record = np.zeros((args.test_episode,))
    ys = torch.arange(args.way).repeat(args.shot)
    ys = ys.type(torch.cuda.LongTensor)
    yq = torch.arange(args.way).repeat(args.query)
    yq = yq.type(torch.cuda.LongTensor)
    model = model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            loss, acc, _ = model.metatrainforward(batch, ys, yq, args.way, args.shot, args.query)
            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
        m, pm = compute_confidence_interval(test_acc_record)
        result_list = [shot, ave_acc.item() * 100, pm * 100]
    return result_list


if __name__ == '__main__':
    opts = pertrain_argparser()
    opts.model_status = 'test'
    opts.set = 'test'
    matching_method = ['SumMin']
    pprint(vars(opts))
    print('------------------------------------------------')
    print(opts.dataset, ' | ', opts.backbone, ' | ' )
    print('------------------------------------------------')
    for i in range(len(matching_method)):
        opts.matching_method = matching_method[i]
        result_list1 = best_model_testing(opts, shot=1)
        result_list2 = best_model_testing(opts, shot=5)
        print(opts.matching_method, '   {}-shot: ({:.2f}±{:.2f}) | /{}-shot: ({:.2f}±{:.2f})'.format(
                                                                         result_list1[0], result_list1[1],
                                                                         result_list1[2], result_list2[0],
                                                                         result_list2[1], result_list2[2]))