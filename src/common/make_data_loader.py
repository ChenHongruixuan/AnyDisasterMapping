import argparse
import os

import imageio
import numpy as np
from torch.utils.data import Dataset
 
import src.common.imutils as imutils
import random


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


class BRIGHTDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader, suffix='.tif'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix


        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            # pre_img, post_img, label = imutils.random_scale(pre_img, post_img, label, scales=(0.75, 1.0, 1.25))
            pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size, mean_rgb=[123.675, 116.28, 103.53])
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'pre-event', self.data_list[index] + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event', self.data_list[index] + '_post_disaster'  + self.suffix)
        label_path = os.path.join(self.dataset_path, 'target', self.data_list[index] + '_building_damage'  + self.suffix)
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)  
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        post_img = np.stack((post_img,)*3, axis=-1)
        clf_label = self.loader(label_path)
        

        if 'train' in self.data_pro_type:
            pre_img, post_img, clf_label = self.__transforms(True, pre_img, post_img, clf_label)
        else:
            pre_img, post_img, clf_label = self.__transforms(False, pre_img, post_img, clf_label)
            clf_label = np.asarray(clf_label)
        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)
    


class xBDDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader, suffix='.png'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix


        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            # pre_img, post_img, label = imutils.random_scale(pre_img, post_img, label, scales=(0.75, 1.0, 1.25))
            pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size, mean_rgb=[123.675, 116.28, 103.53])
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        data_id = self.data_list[index]
        post_name = data_id
        if post_name.endswith(self.suffix):
            post_name = post_name[:-len(self.suffix)]

        pre_name = post_name.replace('_post_disaster', '_pre_disaster')
        if pre_name == post_name and '_pre_disaster' not in post_name:
            pre_name = f"{post_name}_pre_disaster"
            post_name = f"{post_name}_post_disaster"

        pre_path = os.path.join(self.dataset_path, 'images', pre_name + self.suffix)
        post_path = os.path.join(self.dataset_path, 'images', post_name + self.suffix)
        label_path = os.path.join(self.dataset_path, 'masks', post_name  + self.suffix)
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)  
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        # post_img = np.stack((post_img,)*3, axis=-1)
        clf_label = self.loader(label_path)
        if len(clf_label.shape) == 3:
           clf_label = clf_label[:, :, 0]

        if 'train' in self.data_pro_type:
            pre_img, post_img, clf_label = self.__transforms(True, pre_img, post_img, clf_label)
        else:
            pre_img, post_img, clf_label = self.__transforms(False, pre_img, post_img, clf_label)
            clf_label = np.asarray(clf_label)
        

        loc_label = clf_label.copy()
        
        loc_label[clf_label > 1] = 1
        clf_label[clf_label > 4] = 255
        
        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)
    
