# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
# from dataloader.utils import ImageJitter
from abc import abstractmethod

identity = lambda x: x


class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


class SetDataManager(DataManager):
    def __init__(self, args, normalization, n_way, n_support, n_query, n_episodes=150):
        super(SetDataManager, self).__init__()
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episodes = n_episodes
        self.image_size = args.img_size
        self.dataset = args.dataset
        self.normalization = normalization

    def get_data_loader(self, data_file):
        # https://github.com/icoz69/DeepEMD/blob/master/Models/dataloader/miniimagenet/fcn/mini_imagenet.py
        transform = [transforms.Resize([92, 92]),
                     transforms.CenterCrop(self.image_size),
                     transforms.ToTensor(),
                     self.normalization
                     ]
        transform = transforms.Compose(transform)
        dataset = SetDataset(data_file, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episodes)
        data_loader_params = dict(batch_sampler=sampler, num_workers=12, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader
