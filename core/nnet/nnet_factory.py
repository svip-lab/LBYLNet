import os
import torch
import pickle
import importlib
import torch.nn as nn
import pdb
import logging
from ..utils.data_parallel import DataParallel
torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, images, phrases, labels, **kwargs):
        preds = self.model(images, phrases, **kwargs)
        loss, focal_loss, off_loss = self.loss(preds, labels, **kwargs)
        return loss, focal_loss, off_loss

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, system_config, model, distributed=False, gpu=None):
        super(NetworkFactory, self).__init__()

        self.system_config = system_config

        self.gpu     = gpu
        self.model   = DummyModule(model)
        self.loss    = model.loss
        self.network = Network(self.model, self.loss)

        if distributed:
            from apex.parallel import DistributedDataParallel, convert_syncbn_model
            torch.cuda.set_device(gpu)
            print("gpu {} build distributed Dataparallel".format(gpu)) 
            self.network = self.network.cuda(gpu)
            self.network = convert_syncbn_model(self.network)
            self.network = DistributedDataParallel(self.network)
            print("gpu {} build distributed model finished".format(gpu)) 
        else:
            # self.network = DataParallel(self.network, chunk_sizes=system_config.chunk_sizes)
            self.network = nn.DataParallel(self.network).cuda(gpu)
        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        
        self.logger = logging.getLogger()
        self.logger.info("ground model total parameters: {:.2f} M".format(total_params / 1000000))


        if system_config.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr= system_config.lr
            )
        elif system_config.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_config.lr, 
                momentum=0.9, weight_decay=0.001
            )
        elif system_config.opt_algo == 'rmsprop':
            if system_config.debug:
                pdb.set_trace()

            self.optimizer = torch.optim.RMSprop(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_config.lr, 
                weight_decay=0.0005
            )
            # visu_params = self.model.module.visu_encoder.parameters()
            # visu_params = [param for param in self.model.module.visu_encoder.parameters()]
            # rest_params = [param for param in self.model.module.parameters() if param not in visu_params]
            # visu_params = list(visu_params)
            # self.optimizer = torch.optim.RMSprop(
                # [{'params':rest_params},
                # {'params':visu_params, 'lr':system_config.lr / 10.}],
                # filter(lambda p: p.requires_grad, self.model.parameters()), 
                # lr = system_config.lr,
                # weight_decay =0.0005
            # )
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def _t_cuda(self, xs):
        if type(xs) is list:
            return [x.cuda(self.gpu, non_blocking=True) for x in xs]
        return xs.cuda(self.gpu, non_blocking=True)

    def train(self, images, phrases, labels, **kwargs):
        images = images.contiguous()
        images = self._t_cuda(images)
        phrases = self._t_cuda(phrases)
        if isinstance(labels, list):
            labels = [self._t_cuda(label) for label in labels]
        else:
            labels = self._t_cuda(labels)
        self.optimizer.zero_grad()
        loss, focal_loss, off_loss = self.network(images, phrases, labels)
        loss = loss.mean()
        focal_loss = focal_loss.mean()
        off_loss = off_loss.mean()
        loss.backward()
        self.optimizer.step()

        return loss, focal_loss, off_loss

    def validate(self, images, phrases, labels, **kwargs):
        with torch.no_grad():
            images = images.contiguous()
            images = self._t_cuda(images)
            phrases = self._t_cuda(phrases)
            # labels = [self._t_cuda(label) for label in labels]
            if isinstance(labels, list):
                labels = [self._t_cuda(label) for label in labels]
            else:
                labels = self._t_cuda(labels)

            loss, focal_loss, off_loss = self.network(images, phrases, labels)
            loss = loss.mean()
            focal_loss = focal_loss.mean()
            off_loss = off_loss.mean()
            return loss, focal_loss, off_loss

    def test(self, images, phrases, **kwargs):
        with torch.no_grad():
            images = images.contiguous()
            images = self._t_cuda(images)
            phrases = self._t_cuda(phrases)
            return self.model(images, phrases, test=True)
            

    def set_lr(self, lr):
        self.logger.info("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            param_group['initial_lr'] = lr

    def get_lr(self):
        # for param_group in self.optimizer.param_groups:
        return self.optimizer.param_groups[0]["lr"]
        # return lr

    def load_pretrained_params(self, pretrained_model):
        self.logger.info("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f, map_location="cpu")
            self.model.load_state_dict(params)

    def load_params(self, epoch):
        cache_file = self.system_config.snapshot_file.format(epoch)
        self.logger.info("loading model from {}".format(cache_file))
        with open(cache_file, "rb") as f:
            check_pt = torch.load(f, map_location="cpu")
            if type(check_pt) == dict:
                params = check_pt['params']
            else:
                params = check_pt
            # params = torch.load(f, map_location="cpu")
            self.model.load_state_dict(params)

    def save_params(self, epoch):
        cache_file = self.system_config.snapshot_file.format(epoch)
        self.logger.info("saving model to {}".format(epoch))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            check_pt = {
                'params': params,
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(check_pt, f)

    def lr_poly(self, base_lr, iter, max_iter, power=0.9):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def lr_step(self, lr, decay_rate):
        return lr / decay_rate