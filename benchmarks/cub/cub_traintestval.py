import numpy as np
from os import listdir
from os.path import isfile, isdir, join
from distutils.dir_util import copy_tree
import os
import json
import random


def jason_crator(dataset='test'):
    data_path = join(os.getcwd(), dataset)
    savedir = './'
    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort()
    label_dict = dict(zip(folder_list, range(0, len(folder_list))))
    classfile_list_all = []
    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list_all.append(
            [join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path, cf)) and cf[0] != '.')])
        random.shuffle(classfile_list_all[i])

    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        file_list = file_list + classfile_list
        label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item for item in folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" % dataset)


cwd = os.getcwd()
data_path = join(cwd, 'CUB_200_2011/images')
savedir = './'
dataset_list = ['train', 'val', 'test']
folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list, range(0, len(folder_list))))

# datasplit is done according to: A Closer Look at Few-shot Classification, Chen (ICLR, 2019)
# https://github.com/wyharveychen/CloserLookFewShot/blob/master/filelists/CUB/write_CUB_filelist.py
clss_ex_path = []
for i, folder_dir in enumerate(folder_list):
    folder_path = join(data_path, folder_dir)
    if i % 2 == 0:
        os.makedirs('train/' + folder_dir)
        copy_tree('./CUB_200_2011/images/' + folder_dir, './train/' + folder_dir)

    elif i % 4 == 1:
        os.makedirs('val/' + folder_dir)
        copy_tree('./CUB_200_2011/images/' + folder_dir, './val/' + folder_dir)

    elif i % 4 == 3:
        os.makedirs('test/' + folder_dir)
        copy_tree('./CUB_200_2011/images/' + folder_dir, './test/' + folder_dir)


jason_crator(dataset='train')
jason_crator(dataset='val')
jason_crator(dataset='test')
