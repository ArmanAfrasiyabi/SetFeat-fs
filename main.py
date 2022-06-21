import sys, os
from dataloader.tl_dataFunctions import train_DataLaoder
from torch.utils.data import DataLoader
from args import pertrain_argparser
from models.Set2SetMatching import Set2SetMatching
import os.path as osp
from utils import *
from test import best_model_testing
import math

args = pertrain_argparser()
set_seed(args.seed)

shot = 1
num_workers = 8
if args.backbone in ['SetFeat4', 'SetFeat4_512']:
    if shot == 1:
        tr_way = 5
        tr_shot = shot
        tr_query = 8
    else:
        tr_way = 10
        tr_shot = shot
        tr_query = 15
else:
    tr_way = 10
    tr_shot = 1
    tr_query = 5

tr_num_episode = 200


def dataloader_fun(opts):
    tr_tl_loader_, normalization = train_DataLaoder(opts)
    va_ml_loader_ = fun_metaLoader(opts, normalization, n_eposide=opts.val_episode, file_name='/val.json')
    args_ = pertrain_argparser()
    args_.way = tr_way
    args_.shot = tr_shot
    args_.query = tr_query
    tr_ml_loader_ = fun_metaLoader(args_, normalization, n_eposide=tr_num_episode, file_name='/train.json')
    print(opts.dataset, ' dataset is loading...')
    if not args.random_val_task:
        va_ml_loader_ = [x for x in va_ml_loader_]
    va_ys = torch.arange(args.way).repeat(args.shot)
    va_yq = torch.arange(args.way).repeat(args.query)
    va_ys = va_ys.type(torch.cuda.LongTensor)
    va_yq = va_yq.type(torch.cuda.LongTensor)

    tr_ys = torch.arange(tr_way).repeat(tr_shot)
    tr_yq = torch.arange(tr_way).repeat(tr_query)
    tr_ys = tr_ys.type(torch.cuda.LongTensor)
    tr_yq = tr_yq.type(torch.cuda.LongTensor)
    return tr_tl_loader_, tr_ml_loader_, va_ml_loader_, [va_ys, va_yq, tr_ys, tr_yq]


def optimizer_scheduler(opts):
    if args.backbone in ['SetFeat4', 'SetFeat4_512']:
        opts.lr = 0.001
        f = Set2SetMatching(args)
        f = f.cuda()
        optimizer_ = torch.optim.Adam(f.parameters(), lr=opts.lr, weight_decay=0.0005)
        lr_scheduler_ = None
        # print('ADAM optimizer is set!')
    elif args.backbone in ['SetFeat12', 'SetFeat12_11M']:
        f = Set2SetMatching(args)
        f = f.cuda()
        opts.lr = 0.1
        optimizer_ = torch.optim.SGD([{'params': f.encoder.parameters(), 'lr': opts.lr},
                                      {'params': f.clf.parameters(), 'lr': opts.lr}
                                      ], momentum=0.9, nesterov=True, weight_decay=0.0005)
        lr_scheduler_ = torch.optim.lr_scheduler.StepLR(optimizer_, step_size=opts.step_size, gamma=opts.gamma)
        # print('SGD optimizer is set!')
    return f, optimizer_, lr_scheduler_


args.save_path = '%s/%s-%s/' % (args.dataset, args.backbone, args.sqa_type)
args.save_path = osp.join('checkpoint', args.save_path)
ensure_path(args.save_path)
trlog = {'args': vars(args), 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
         'max_acc': 0.0, 'max_acc_epoch': 0}


def validation_fun(f, va_loader, episods_th):
    tl_, ta_ = Averager(), Averager()
    with torch.no_grad():
        for b, batch in enumerate(va_loader):
            loss, acc, _ = f.metatrainforward(batch, va_ys, va_yq, args.way, args.shot, args.query)
            tl_.add(loss.item())
            ta_.add(acc)
            if b == episods_th:
                break
    return tl_.item(), ta_.item() * 100


def training_fun(net, updater, scheduler, trainloader, valloader, matching_method):
    prt_text = '     ep.%d     l: %4.2f/%4.2f     a: %4.2f/%4.2f '
    if args.model_status == 'pertrain':
        episods_th = 150
        meta_save_name = '_'
    else:
        episods_th = 300
        meta_save_name = '_' + matching_method + '_'
    for epoch in range(1, args.max_epoch + 1):
        net.train()
        tl, ta = Averager(), Averager()
        for batch in trainloader:
            if args.model_status == 'pertrain':
                loss, acc = net.pretrainforward(batch)
            else:
                loss, acc, frequency_y = net.metatrainforward(batch, tr_ys, tr_yq, tr_way, tr_shot, tr_query,
                                                              matching_method)
            updater.zero_grad()
            loss.backward()
            updater.step()
            tl.add(loss.item())
            ta.add(acc)
        tl, ta = tl.item(), ta.item() * 100
        vl, va = validation_fun(net.eval(), valloader, episods_th)
        prt_text_ = prt_text
        if va >= trlog['max_acc']:
            prt_text_ += ' up!'
            trlog['max_acc'] = va
            trlog['max_acc_epoch'] = epoch
            torch.save({'epoch': epoch,
                        'args': args,
                        'model': net.encoder.state_dict(),
                        'fc': net.clf.state_dict(),
                        'optimizer': updater,
                        'trlog': trlog,
                        'result_list': result_list,
                        'lr_scheduler': lr_scheduler},
                       osp.join(args.save_path, args.model_status + meta_save_name + 'best_model.pth'))
        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)
        torch.save({'epoch': epoch,
                    'args': args,
                    'model': net.encoder.state_dict(),
                    'optimizer': updater,
                    'trlog': trlog,
                    'result_list': result_list,
                    'lr_scheduler': lr_scheduler},
                   osp.join(args.save_path, args.model_status + meta_save_name + 'ith_model.pth'))
        print(prt_text_ % (epoch, tl, vl, ta, va))
        result_list.append(prt_text_ % (epoch, tl, vl, ta, va))
        save_list_to_txt(os.path.join(args.save_path, args.model_status + meta_save_name + 'results.txt'), result_list)
        if scheduler is not None:
            scheduler.step()
        if (epoch == 40) and (args.model_status == 'pertrain'):
            episods_th = 600
        elif (epoch == 25) and (args.model_status == 'metatrain'):
            episods_th = 600
    args.model_status = 'metatrain'


print(args.dataset, '/', args.backbone, '/', args.shot)
tr_tl_loader, tr_ml_loader, va_ml_loader, [va_ys, va_yq, tr_ys, tr_yq] = dataloader_fun(args)
model, optimizer, lr_scheduler = optimizer_scheduler(args)
if args.model_status == 'pertrain':
    print('--------------------------------------------------------')
    print('--                      pertrain                      --')
    print('--------------------------------------------------------')
    result_list = [args.save_path, str(vars(args))]
    training_fun(model, optimizer, lr_scheduler, tr_tl_loader, va_ml_loader, matching_method='SumMin')

print('--------------------------------------------------------')
print('--              metatrain -', args.matching_method, '                  --')
print('--------------------------------------------------------')
args.model_status = 'metatrain'
args.max_epoch = 60
args.step_size = 30
args.lr = 0.0005
trlog = {'args': vars(args), 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
         'max_acc': 0.0, 'max_acc_epoch': 0}
checkpoint = torch.load(osp.join(args.save_path, 'pertrain_best_model.pth'))
model.encoder.load_state_dict(checkpoint['model'])
result_list = [args.save_path, str(vars(args))]
optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
training_fun(model, optimizer, lr_scheduler, tr_ml_loader, va_ml_loader, args.matching_method)

if args.backbone == 'SetFeat12_11M':
    backbone = 'SetFeat12*'
elif args.backbone == 'SetFeat4':
    backbone = 'SetFeat4-64'

print('------------------------------------------------')
print(' Testing ', ' | ', args.dataset, ' | ', backbone, ' | ')
print('------------------------------------------------')
result_list1 = best_model_testing(args, shot=1)
result_list2 = best_model_testing(args, shot=5)
print(args.matching_method, '   {}-shot: ({:.2f}±{:.2f}) | /{}-shot: ({:.2f}±{:.2f})'.format(
    result_list1[0], result_list1[1],
    result_list1[2], result_list2[0],
    result_list2[1], result_list2[2]))
