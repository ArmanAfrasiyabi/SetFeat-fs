import argparse


def pertrain_argparser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('-dataset', type=str, default='cub',
                        choices=['cub', 'miniimagenet', 'tieredimagenet'])
    parser.add_argument('-backbone', type=str, default='SetFeat12_11M',
                        choices=['SetFeat4', 'SetFeat4_512', 'SetFeat12', 'SetFeat12_11M'],
                        help='SetFeat12_11M referes to SetFeat12^* in the paper'
                             'SetFeat4 persents SetFeat4-64')
    parser.add_argument('--img_size', type=int, default=84, help='img_size')
    parser.add_argument('-data_dir', type=str, default='./benchmarks/')
    parser.add_argument('-model_status', type=str, default='pertrain', help=['pertrain | metatrain | test'])
    parser.add_argument('-max_epoch', type=int, default=400)
    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-step_size', type=int, default=30)
    parser.add_argument('-resume', action='store_true', help='true for saving')
    parser.add_argument('-gamma', type=float, default=0.2)
    parser.add_argument('-bs', type=int, default=64)

    # validation
    parser.add_argument('-set', type=str, default='val', choices=['val', 'test'],
                        help='the set for episodic testing')
    parser.add_argument('-way', type=int, default=5)
    parser.add_argument('-shot', type=int, default=1)
    parser.add_argument('-query', type=int, default=15)
    parser.add_argument('-sqa_type', type=str, default='linear', help=['convolution | linear '])
    parser.add_argument('-save_all', action='store_true', help='save models on each epoch')
    parser.add_argument('-random_val_task', action='store_true',
                        help='random samples tasks for validation in each epoch')

    parser.add_argument('-matching_method', type=str, default='SumMin', help=['SumMin | MinMin | PairwiseSum'])

    # about training
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-num_episode', type=int, default=100)
    parser.add_argument('-test_episode', type=int, default=600)
    parser.add_argument('-val_episode', type=int, default=300)

    parser.add_argument('--num_workers', default=8, help='the number of workers')
    parser.add_argument('--patch_size', type=int, default=1, help='patch_size')
    parser.add_argument('--top_k', type=int, default=10, help='take the top k heads in the distance matrix')
    parser.add_argument('--dist_type', type=str, default='cosine', help='dist_type!')
    parser.add_argument('--num_class', type=int, default=100, help='num of base classes')

    args = parser.parse_args()
    if args.dataset == 'cub':
        args.num_class = 100
        args.max_epoch = 200
        args.bs = 64 
    if args.backbone in ['SetFeat12', 'SetFeat12_11M']:
        args.sqa_type = 'convolution'
    elif args.backbone == 'SetFeat4':
        args.sqa_type = 'linear'
    return args

