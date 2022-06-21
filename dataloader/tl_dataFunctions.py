import torchvision.transforms as Transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image


def train_DataLaoder(args, section='train', shuffle=True):
    data_path = args.data_dir + args.dataset + '/' + section + '/'
    if args.dataset in ['cub']:
        # DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover's Distance and Structured Classifiers"
        # (CVPR 2020 oral) https://github.com/icoz69/DeepEMD/blob/master/Models/dataloader/cub/fcn/cub.py
        normalization = Transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    transformer_set = [
        Transforms.RandomResizedCrop(args.img_size),
        Transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # cub
        Transforms.RandomHorizontalFlip(),
        Transforms.ToTensor(),
        normalization,
    ]
    transforms = Transforms.Compose(transformer_set)
    dataset = ImageFolder(root=data_path, transform=transforms)
    return DataLoader(dataset=dataset,
                      batch_size=args.bs,
                      num_workers=args.num_workers,
                      shuffle=shuffle,
                      drop_last=False), normalization
