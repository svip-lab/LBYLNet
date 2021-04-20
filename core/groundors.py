#!/usr/bin/env python
import cv2
import pdb
import torch
# from core.groundors import YoloRefer
import os.path as osp
import json
import pdb
import numpy as np
from core.dbs import datasets
from core.sampler.sampler import Referring
from core.sampler.collate_fn import collate_fn
from core.sampler.utils import letterbox, normalize_, resize_image_
from core.config import SystemConfig
from core.nnet.nnet_factory import NetworkFactory
from core.utils import make_anchors
from core.test.test import _decode, _bbox_iou, _visualize, _decode_anchorbased
from core.models.net import LBYLNet as Model
from core.paths import get_file_path

torch.backends.cudnn.benchmark = False

class Net(object):
    def __init__(self, cfg_file, iter):
        with open(osp.join("./configs", cfg_file + ".json"), "r") as f:
            config = json.load(f)
        
        config["system"]["snapshot_name"] = cfg_file
        system_config = SystemConfig().update_config(config["system"])
        system_config.lr = 0.001
        self.system_config = system_config
        anchors = make_anchors(system_config.dataset, 416)
        config["db"]["anchors"] = anchors
        config["db"]["corpus_path"] = get_file_path("..", "data", "refer", "data",  config["system"]["dataset"], "corpus.pth") 
        self.config = config
        model = Model(system_config, config['db'])
        self.model = model
        self.nnet = NetworkFactory(system_config, model)
        self.nnet.load_params(iter)
        self.nnet.eval_mode()
        split  = system_config.val_split
        self.db = datasets['refer'](config["db"], split=split, sys_config=system_config)
        self.input_size = self.db.configs["input_size"]
        self.dataset = Referring(self.db, system_config, data_aug=False, \
            debug=False, shuffle=False, test=True)
        self.original_shape = None
        print("initailize...")

    def tokenize(self, phrase):
        phrase =  self.dataset._tokenize_phrase(phrase)
        phrase = torch.stack([phrase])
        return phrase

    def postprocess(self, out):
        bbox = _decode_anchorbased(out)[0]
        height, width = self.original_shape[0:2]
        reshape_ratio = min(self.input_size[0] / height, \
                            self.input_size[1] / width)
        resize_shape = round(height * reshape_ratio), round(width * reshape_ratio)
        dh = (self.input_size[0] - resize_shape[0]) / 2 # height padding
        dw = (self.input_size[1] - resize_shape[1]) / 2 # width padding
        bbox[0:4:2] = (bbox[0:4:2] - dw) / reshape_ratio
        bbox[1:4:2] = (bbox[1:4:2] - dh) / reshape_ratio
        bbox[0:4:2] = np.clip(bbox[0:4:2], 0, width-1)
        bbox[1:4:2] = np.clip(bbox[1:4:2], 0, height-1)
        return bbox

    def prepocess_image(self, image):
        if not image.shape[-1] > 1:
            image = np.stack([image] * 3) # duplicate channel if gray image
        
        self.original_shape = image.shape
        dummy_bbox = [0, 100, 0, 100]
        image, bbox = resize_image_(image, dummy_bbox.copy(), self.input_size, \
                    padding_color=tuple((self.db.mean * 255).tolist()))
        
        image = image.astype(np.float32) / 255.
        normalize_(image, self.db.mean, self.db.std)
        image = image.transpose((2, 0, 1))
        image = np.stack([image])
        image = torch.from_numpy(image)
        return image

    @torch.no_grad()
    def __call__(self, image, phrase):
        image = self.prepocess_image(image)
        phrase = self.tokenize(phrase)
        out = self.nnet.test(image, phrase)
        bbox = self.postprocess(out)
        return bbox