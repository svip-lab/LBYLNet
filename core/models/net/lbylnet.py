import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import re
from ...paths import get_file_path
# from ..context.module import convolution, residual
from ..lang_encoder.RNNencoder import BertEncoder, RNNEncoder
from ..utils.losses import Loss
from .darknet import Darknet
from ..context.module import (convolution, residual, LandmarkP1,LandmarkP2, LandmarkP2x, LandmarkP4, LandmarkP4x, LandmarkP8, Nonlocal, Dilated)

context = {
    'LandmarkP8': LandmarkP8,
    'LandmarkP4': LandmarkP4,
    'LandmarkP4x':LandmarkP4x,
    'LandmarkP2': LandmarkP2,
    'LandmarkP2x':LandmarkP2x,
    'Nonlocal': Nonlocal,
    'Dilated' : Dilated,
}

class LBYLNet(nn.Module):
    def __init__(self, cfg_sys, cfg_db):
        super(LBYLNet, self).__init__()
        self.init_configs(cfg_sys, cfg_db)

        # visu_encoder
        weight_path = get_file_path("..", "ext", "yolov3.weights")
        darknet_cfg_path = get_file_path("..", "ext", "yolov3.cfg")
        self.visu_encoder = Darknet(config_path=darknet_cfg_path)
        self.visu_encoder.load_weights(weight_path)

        # lang_encoder
        if self.lstm:
            self.lang_encoder = RNNEncoder(self.cfg_db)
        else:
            self.lang_encoder = BertEncoder(self.cfg_db, cfg_sys=cfg_sys)
        
        # fusion module
        self.mapping_visu = nn.ModuleList([self._make_conv(dim, self.joint_embedding_size, 1) \
                                          for dim in [1024, 512, 256]])
        self.norms = nn.ModuleList([nn.InstanceNorm2d(self.joint_inp_dim) for _ in [1024, 512, 256]])
        self.mapping_lang = self._make_mlp(self.lang_dim, self.joint_embedding_size, self.joint_embedding_dropout)
        self.gamma = nn.ModuleList(nn.Linear(self.joint_embedding_size, self.joint_inp_dim) for _ in [1024, 512, 256])
        self.beta = nn.ModuleList(nn.Linear(self.joint_embedding_size, self.joint_inp_dim) for _ in [1024, 512, 256])

        self.joint_fusion = nn.ModuleList([self._make_conv(self.joint_inp_dim, self.joint_out_dim, 1) \
                                          for _ in [1024, 512, 256]])
        
        # landmark feature convolution module
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.refine = self._make_conv(self.joint_out_dim, self.joint_out_dim, 1)

        self.context_block = None
        if self.cfg_sys.context:
            self.context_block = context.get(cfg_sys.context)(self.joint_out_dim, mapdim=self.map_dim)
            # print("using context module {}".format(self.context_block))

        self.out_funcs = nn.ModuleList([self._make_pred(self.joint_out_dim, 3*5) 
                                  for _ in range(3)])
        # localization module
        self.loss = Loss(off_weight=5., anchors=self.anchors, input_size=self.input_size[0])


    def forward(self, images, phrases, masks=None, test=False):
        # visual module
        device = images.device
        batch_size = images.shape[0]
        # visual language encoders 
        visu_feats = self.visu_encoder(images) # fpn P3->P5
        coord_feats = [self._make_coord(batch_size, x.shape[2], x.shape[3]) for x in visu_feats]
        lang_feat = self.lang_encoder(phrases, mask=phrases.gt(0))
        # only use a global representation of language
        # you can also use more complex modeling using word-level representations
        # to improve performance, motivated by lots of previous works.
        # Usage: lang_feat = lang_feat['words'] shape [seq_len, dim]
        lang_feat = lang_feat['hidden']    
        lang_feat  = self.mapping_lang(lang_feat)

        # use FILM to enhance visual-language fusion
        gamma = [F.tanh(gamma(lang_feat)) for gamma in self.gamma]
        beta = [F.tanh(beta(lang_feat)) for beta in self.beta]
        visu_feat = []
        for ii, feat in enumerate(visu_feats):
            coord_feat = coord_feats[ii].to(device)
            feat = torch.cat([self.mapping_visu[ii](feat), coord_feat], dim=1)
            feat = self.norms[ii](feat)
            b = beta[ii].view(batch_size, -1, 1, 1).expand_as(feat)
            g = gamma[ii].view(batch_size, -1, 1, 1).expand_as(feat)
            feat = F.relu(g * feat + b)
            visu_feat.append(feat)

        # prior to modeling context, we tackle the scale problem for modeling context by simply sum them.
        joint_feats = [fusion(feat) for feat, fusion in zip(visu_feat, self.joint_fusion)]
        lower_feat = self.upsample(joint_feats[0])
        higher_feat = self.maxpool(joint_feats[-1])
        inter_feat = self.refine((joint_feats[1] + lower_feat + higher_feat) / 3.)
        
        # any context-aware module you want.
        if self.context_block:
            inter_feat = self.context_block(inter_feat)

        # distribute back to FPN accounting for scale problem for detection.
        joint_feats[0] = joint_feats[0] + self.maxpool(inter_feat)
        joint_feats[1] = joint_feats[1] + inter_feat
        joint_feats[2] = joint_feats[2] + self.upsample(inter_feat)

        # make predictions.
        outs = [func(feat) for func, feat in zip(self.out_funcs, joint_feats)]
        if not test:
            return outs
        else:
            return self.loss(outs, targets=None)

    def init_configs(self, cfg_sys, cfg_db):
        self.cfg_sys                = cfg_sys
        self.cfg_db                 = cfg_db
        self.lstm                   = cfg_sys.lstm
        # loading parameter
        self.lang                    = self.cfg_sys.lang_encoder
        self.anchors                 = self.cfg_db["anchors"]
        self.joint_embedding_size    = self.cfg_db['joint_embedding_size']
        self.joint_embedding_dropout = self.cfg_db['joint_embedding_dropout']
        self.joint_mlp_layers        = self.cfg_db['joint_mlp_layers']
        self.n_layers                = self.cfg_db['n_layers']
        self.output_sizes            = self.cfg_db['output_sizes']
        self.hidden_size             = self.cfg_db['hidden_size']
        self.input_size              = self.cfg_db['input_size']
        self.num_dirs                = 2 if self.cfg_db['bidirectional'] else 1
        self.lang_dim                = self.hidden_size * self.num_dirs
        self.coord_dim               = 8
        self.joint_inp_dim           = self.coord_dim + self.joint_embedding_size
        self.joint_out_dim           = self.cfg_db['joint_out_dim']
        self.map_dim                 = self.cfg_sys.ctx_dim
        if not self.lstm and self.lang == 'bert-base-uncased':
            self.lang_dim = 768
        else:
            self.lang_dim = 1024


    def _make_pred(self, input_dim, output_dim):
        pred = nn.Sequential(
            convolution(3, input_dim, input_dim, with_bn=False),
            nn.Conv2d(input_dim, output_dim, (1, 1))
        )
        return pred

    def _make_mlp(self, input_dim, output_dim, drop):
        return nn.Sequential(nn.Linear(input_dim, output_dim), 
                nn.BatchNorm1d(output_dim), 
                nn.ReLU(inplace=True), 
                nn.Dropout(drop),
                nn.Linear(output_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True))

    def _make_conv(self, input_dim, output_dim, k, stride=1):
        pad = (k - 1) // 2
        return nn.Sequential(
            nn.Conv2d(input_dim, output_dim, (k, k), padding=(pad, pad), stride=(stride, stride)),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )

    def _make_coord(self, batch, height, width):
        xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
        xv_min = (xv.float()*2 - width)/width
        yv_min = (yv.float()*2 - height)/height
        xv_max = ((xv+1).float()*2 - width)/width
        yv_max = ((yv+1).float()*2 - height)/height
        xv_ctr = (xv_min+xv_max)/2
        yv_ctr = (yv_min+yv_max)/2
        hmap = torch.ones(height, width)*(1./height)
        wmap = torch.ones(height, width)*(1./width)
        coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
            xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
            xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
            hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0))
        coord = coord.unsqueeze(0).repeat(batch,1,1,1)
        return coord

    def _normalize(self, feat, p=2, dim=1):
        return F.normalize(feat, p, dim)