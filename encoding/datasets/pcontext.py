###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
import os
import numpy as np

import torch

from PIL import Image
from tqdm import trange

from .base import BaseDataset
from scipy.ndimage.interpolation import shift

class ContextSegmentation(BaseDataset):
    BASE_DIR = 'VOCdevkit/VOC2010'
    NUM_CLASS = 59
    def __init__(self, root=os.path.expanduser('./datasets/pcontext'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(ContextSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        from detail import Detail
        #from detail import mask
        root = os.path.join(root, self.BASE_DIR)
        annFile = os.path.join(root, 'trainval_merged.json')
        imgDir = os.path.join(root, 'JPEGImages')
        # training mode
        self.detail = Detail(annFile, imgDir, split)
        self.transform = transform
        self.target_transform = target_transform
        self.ids = self.detail.getImgs()
        # generate masks
        self._mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296, 
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424, 
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360, 
            98, 187, 104, 105, 366, 189, 368, 113, 115]))
        self._key = np.array(range(len(self._mapping))).astype('uint8')
        mask_file = os.path.join(root, self.split+'.pth')
        print('mask_file:', mask_file)
        if os.path.exists(mask_file):
            self.masks = torch.load(mask_file)
        else:
            self.masks = self._preprocess(mask_file)
    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _preprocess(self, mask_file):
        masks = {}
        tbar = trange(len(self.ids))
        print("Preprocessing mask, this will take a while." + \
            "But don't worry, it only run once for each split.")
        for i in tbar:
            img_id = self.ids[i]
            mask = Image.fromarray(self._class_to_index(
                self.detail.getMask(img_id)))
            masks[img_id['image_id']] = mask
            tbar.set_description("Preprocessing masks {}".format(img_id['image_id']))
        torch.save(masks, mask_file)
        return masks

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = img_id['file_name']
        iid = img_id['image_id']
        img = Image.open(os.path.join(self.detail.img_folder, path)).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(path)
        # convert mask to 60 categories
        mask = self.masks[iid]
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
            
        # onehot_mask = mask.clone()
        # onehot_mask[onehot_mask == -1] = self.NUM_CLASS
        # border = 2
        # onehot_mask = torch.nn.functional.pad(onehot_mask, (border, border, border, border), 'constant', self.NUM_CLASS)
        # onehot_label = torch.nn.functional.one_hot(onehot_mask, num_classes=self.NUM_CLASS+1)
        # h,w = mask.size()
        # label = torch.zeros((h,w, self.NUM_CLASS+1))
        # for i in range(0,border*2+1):
        #     for j in range(0, border*2+1):
        #         label += onehot_label[i:i+h, j:j+w, :]
        # label[label>1] = 1
        # label = label[:,:,:self.NUM_CLASS]
        # sum_label = torch.sum(label, dim=2, keepdim=False)
        # mask[sum_label>1]=-1
        
        # onehot_label = onehot_label.permute(2, 0, 1)
        
        # np_mask = np.array(onehot_mask).astype('int32')
        # for i in range(-border,border+1):
        #     for j in range(-border, border+1):
        #         shifted= shift(np_mask,(i,j), cval=self.NUM_CLASS)
        #         onehot_label += torch.nn.functional.one_hot(torch.from_numpy(shifted).long(), num_classes=self.NUM_CLASS+1)       
        
        # onehot_label[onehot_label>1] = 1
        # onehot_label = onehot_label[:,:,:self.NUM_CLASS]
        # sum_label = torch.sum(onehot_label, dim=2, keepdim=False)
        # mask[sum_label>1]=-1
        
        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32') - 1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.ids)

    @property
    def pred_offset(self):
        return 1
