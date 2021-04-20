import os
import os.path as osp
import json
import numpy as np
import torch
import sys
from .referring import REFERDB
from ..paths import get_file_path

from . import utils
from .utils import Corpus
sys.modules['utils'] = utils
import pdb


class Sampler(REFERDB):
    def __init__(self, db_config, split=None, sys_config=None):
        super(Sampler, self).__init__(db_config)
        self._mean    = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std     = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        
        self._dataset = sys_config.dataset
        self.data_dir = sys_config.data_dir
        
        # setting datasource
        if self._dataset == 'referit':
            self._data_root = osp.join(sys_config.data_dir, 'refer', 'ln_data', 'referit')
            self._im_dir = osp.join(self._data_root, 'images')
            self._split_dir = osp.join(self._data_root, 'splits')
        elif self._dataset == 'flickr':
            self._data_root = osp.join(sys_config.data_dir, 'refer', 'ln_data', 'Flickr30k')
            self._im_dir = osp.join(self.data_dir, 'flickr30k_images')
        else: #refer coco etc.
            self._data_root = osp.join(sys_config.data_dir, 'refer', 'ln_data', 'other')
            self._im_dir = osp.join(self._data_root, 'images', 'mscoco', 'images', 'train2014')

        dataset_root = osp.join(sys_config.data_dir, 'refer', 'data', self._dataset)
        valid_splits = SUPPORTED_DATASETS[self._dataset]['splits']
        
        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self._dataset, split))
        
        # setting database
        self.database = []
        splits = [split]
        if self._dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self._dataset, split)
            imgset_path = osp.join(dataset_root, imgset_file)
            self.database += torch.load(imgset_path, map_location = "cpu")

        # processing database
        if self._dataset == 'flickr':
            self.img_names, self.bboxs, self.phrases = zip(*self.database)
        else:
            self.img_names, _, self.bboxs, self.phrases, _ = zip(*self.database)

        self._db_inds = np.arange(len(self.database))
        self.corpus = torch.load(db_config["corpus_path"], map_location='cpu')
        
        self.covert_bbox = []
        if not (self._dataset == 'referit' or self._dataset == 'flickr'): # for refcoco, etc
            # covert x1y1wh to x1y1x2y2
            for bbox in self.bboxs:
                bbox = np.array(bbox, dtype=int)
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                self.covert_bbox.append(bbox)
        else:
            for bbox in self.bboxs:                                     # for referit, flickr
                bbox = np.array(bbox, dtype=int)
                self.covert_bbox.append(bbox)

    def image_path(self, ind): # notice: db index is the actual index of data.
        return osp.join(self._im_dir, self.img_names[ind])
    
    def annotation_box(self, ind):
        return self.covert_bbox[ind].copy()
    
    def phrase(self, ind):
        return self.phrases[ind]

        
# Meta Information
SUPPORTED_DATASETS = {
    'referit': {'splits': ('train', 'val', 'trainval', 'test')},
    'unc': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco', 'split_by': 'unc'}
    },
    'unc+': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
    },
    'gref': {
        'splits': ('train', 'val'),
        'params': {'dataset': 'refcocog', 'split_by': 'google'}
    },
    'flickr': {
        'splits': ('train', 'val', 'test')}
}









