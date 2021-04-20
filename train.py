#!/usr/bin/env python
import os
import json
import torch
import numpy as np
import queue
import pprint
import random 
import argparse
import importlib
import threading
import traceback
import time
import logging
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from tqdm import tqdm
from torch.multiprocessing import Process, Queue, Pool

from core.dbs import datasets
from core.utils import stdout_to_tqdm, AverageMeter, make_anchors, get_root_logger
from core.config import SystemConfig
from core.nnet.nnet_factory import NetworkFactory
from core.sampler.sampler import Referring
from core.sampler.collate_fn import collate_fn, collate_fn_bert
from core.optimizer.lr_scheduler import make_scheduler
from core.models.net.lbylnet import LBYLNet
from core.models.net.baseline import Baseline
from core.paths import get_file_path
import pdb
seed = 413
random.seed(seed)
np.random.seed(seed+1)
torch.manual_seed(seed+2)
torch.cuda.manual_seed_all(seed+3)
torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--start_epoch", dest="start_epoch",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--lr_type", default='step', type=str)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world_size", default=-1, type=int,
                        help="number of nodes of distributed training")
    parser.add_argument("--rank", default=0, type=int,
                        help="node rank for distributed training")
    parser.add_argument("--dist_url", default=None, type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--dist_backend", default="nccl", type=str)
    parser.add_argument("--dataset", default=None, type=str)
    args = parser.parse_args()
    return args

def val_epoch(nnet, val_loader, rank, epoch, lr, print_freq, args):
    logger = args.logger

    def reduce_tensor(inp, average=False):
        """
        Reduce the loss from all the process so 
        that process with rank 0 has average result.
        """
        if args.world_size < 2:
            return inp
        with torch.no_grad():
            reduced_inp = inp
            dist.reduce(reduced_inp, dst=0)
            if average: 
                reduced_inp = reduced_inp / args.world_size
        return reduced_inp
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    focal_losses = AverageMeter()
    off_losses = AverageMeter()
    end = time.time()
    nnet.eval_mode()
    for iter, batch in enumerate(val_loader):
        data_time.update(time.time()-end)
        loss, focal_loss, off_loss = nnet.validate(**batch)
        
        if args.distributed:
            loss = reduce_tensor(loss, average=True)
            focal_loss = reduce_tensor(focal_loss, average=True)
            off_loss = reduce_tensor(off_loss, average=True)

        losses.update(loss.item())
        focal_losses.update(focal_loss.item())
        off_losses.update(off_loss.item())
        batch_time.update(time.time() - end)

        
        if rank==0 and print_freq and (iter+1) % print_freq == 0:

            message = ('Process {}\t'
                'epoch[{}][{}/{}]\t' \
                'time {time.val:.3f} ({time.avg:.3f})\t' \
                'data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'loss {losses.val:.4f} ({losses.avg:.4f})\t'\
                'rank loss {focal_losses.val:.4f} ({focal_losses.avg:.4f})\t'\
                'offs loss {off_losses.val:.4f} ({off_losses.avg:4f})\t' \
                'lr {lr:.8f}'.format(
                    rank, epoch, iter+1, len(val_loader),
                    time=batch_time, data_time=data_time, losses=losses, 
                    focal_losses =focal_losses, 
                    off_losses=off_losses, lr=lr
                ))
            logger.info(message)
            print(message)

        end = time.time()

    return losses.avg, focal_losses.avg, off_losses.avg

def train_epoch(nnet, train_loader, rank, epoch, lr, print_freq, args):
    logger = args.logger

    def reduce_tensor(inp, average=False):
        """
        Reduce the loss from all the process so 
        that process with rank 0 has average result.
        """
        if args.world_size < 2:
            return inp
        with torch.no_grad():
            reduced_inp = inp
            dist.reduce(reduced_inp, dst=0)
            if average: 
                reduced_inp = reduced_inp / args.world_size
        return reduced_inp
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    focal_losses = AverageMeter()
    off_losses = AverageMeter()
    end = time.time()
    nnet.train_mode()
    for iter, batch in enumerate(train_loader):
        data_time.update(time.time()-end)
        loss, focal_loss, off_loss = nnet.train(**batch)
        
        if args.distributed:
            loss = reduce_tensor(loss, average=True)
            focal_loss = reduce_tensor(focal_loss, average=True)
            off_loss = reduce_tensor(off_loss, average=True)

        losses.update(loss.item())
        focal_losses.update(focal_loss.item())
        off_losses.update(off_loss.item())
        batch_time.update(time.time() - end)
        if rank==0 and print_freq and (iter+1) % print_freq == 0:

            message = ('Process {}\t'
                'epoch[{}][{}/{}]\t' \
                'time {time.val:.3f} ({time.avg:.3f})\t' \
                'data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'loss {losses.val:.4f} ({losses.avg:.4f})\t'\
                'rank loss {focal_losses.val:.4f} ({focal_losses.avg:.4f})\t'\
                'offs loss {off_losses.val:.4f} ({off_losses.avg:4f})\t' \
                'lr {lr:.8f}'.format(
                    rank, epoch, iter+1, len(train_loader),
                    time=batch_time, data_time=data_time, losses=losses, 
                    focal_losses =focal_losses, 
                    off_losses=off_losses, lr=lr
                ))

            logger.info(message)
            print(message)

        end = time.time()

    return losses.avg, focal_losses.avg, off_losses.avg


def train(model, 
        train_loader, 
        val_loader, 
        train_sampler, 
        val_sampler,
        system_config, 
        args):
    # reading arguments from command
    start_epoch  = args.start_epoch
    distributed  = args.distributed
    world_size   = args.world_size
    initialize   = args.initialize
    rank         = args.rank
    logger       = args.logger
    # reading arguments from json file
    args.batch_size  = system_config.batch_size
    learning_rate    = system_config.learning_rate * world_size \
                        if world_size > 0 else system_config.learning_rate
    warm_up          = system_config.warm_up
    warm_up_lr       = system_config.warm_up_lr
    base_lr          = warm_up_lr if warm_up else learning_rate
    pretrained_model = system_config.pretrain
    snapshot         = system_config.snapshot
    val_iter         = system_config.val_iter
    nb_epoch         = system_config.nb_epoch
    print_freq       = system_config.print_freq
    # for automatic test after finishing training
    args.test_split  = system_config.test_split
    args.test_epoch  = system_config.nb_epoch
    # system_config.learning_rate = base_lr
    system_config.lr = base_lr

    print("Process {}: building model...".format(rank))
    nnet = NetworkFactory(system_config, model, distributed=distributed, gpu=rank)
    if initialize:
        if rank == 0:
            nnet.save_params(0)
        exit(0)

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        logger.info("Process {}: loading from pretrained model".format(rank))
        nnet.load_pretrained_params(pretrained_model)

    if start_epoch:
        nnet.load_params(start_epoch)
        logger.info("Process {}: training starts from iteration {} with learning_rate {}".format(rank, start_epoch + 1, base_lr))

    if rank == 0:
        logger.info("training start...")
    
    nnet.cuda()
    nnet.train_mode()
    lr_scheduler = make_scheduler(nnet.optimizer, system_config, last_epoch=-1)
    # dummpy loop for lr_scheduler
    for epoch in range(start_epoch): #BUG HERE
        lr_scheduler.step(epoch)


    lr = nnet.get_lr()

    epoch_lr = []
    for epoch in range(start_epoch, nb_epoch):
        
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_epoch(nnet, train_loader, rank, epoch, lr, print_freq, args)
        
        epoch_lr.append(lr)
        lr_scheduler.step(epoch)
        lr = nnet.get_lr()

        if (epoch+1) % snapshot == 0 and rank == 0:
            nnet.save_params(epoch+1)
        
        if (epoch+1) % val_iter == 0:
            if rank == 0:
                logger.info('evaluating...')
            val_epoch(nnet, val_loader, rank, epoch, lr, print_freq, args)
            if rank ==0:
                logger.info('train...')

def main(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    rank = args.rank
    curvers = os.path.join('./curves', args.cfg_file+'_curves_from_epoch_{}.pth'.format(args.start_epoch))
    args.curves = curvers
    logger = get_root_logger(rank, filename=os.path.join("./logs", args.cfg_file+".out"))
    args.logger = logger
    
    logger.info("==================================================================")
    logger.info("train start from here ... \n")

    cfg_file = os.path.join("./configs", args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        config = json.load(f)

    config["system"]["snapshot_name"] = args.cfg_file
    if args.dataset is not None:
        config["system"]["dataset"] = args.dataset 
        
    system_config = SystemConfig().update_config(config["system"])
    anchors = make_anchors(system_config.dataset)
    config["db"]["anchors"] = anchors
    config["db"]["corpus_path"] = get_file_path("..", "data", "refer", "data",  config["system"]["dataset"], "corpus.pth") 
    print(config["db"]["corpus_path"])
    # if you want to access our baseline
    # model = Baseline(system_config, config["db"])
    model = LBYLNet(system_config, config["db"])
    train_split = system_config.train_split
    val_split   = system_config.val_split

    workers = args.workers
    logger.info("Process {}: loading all datasets...".format(rank))
    logger.info("Process {}: using {} workers".format(rank, workers))

    train_db = datasets['refer'](config["db"], split=train_split, sys_config=system_config)
    valid_db = datasets['refer'](config["db"], split=val_split, sys_config=system_config)

    if rank == 0:
        print("system config...")
        pprint.pprint(system_config.full)
        
        logger.info("system config...")
        logger.info(system_config.full)

        print("db config...")
        pprint.pprint(train_db.configs)

        logger.info("db config...")
        logger.info(train_db.configs)

        print("len of training db: {}".format(len(train_db.db_inds)))
        print("len of validate db: {}".format(len(valid_db.db_inds)))
        print("distributed: {}".format(args.distributed))
        logger.info("len of training db: {}".format(len(train_db.db_inds)))
        logger.info("len of validate db: {}".format(len(valid_db.db_inds)))
        logger.info("distributed: {}".format(args.distributed))

    trainset= Referring(train_db, system_config, debug=False)
    validset= Referring(valid_db, system_config, debug=False)

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(trainset, num_replicas=args.world_size, rank=rank)
        val_sampler = DistributedSampler(validset, num_replicas=args.world_size, rank=rank)

    collate_func = collate_fn_bert if not system_config.lstm else collate_fn

    batch_size = int(system_config.batch_size / args.world_size) \
        if args.distributed else system_config.batch_size
    train_loader = DataLoader(dataset=trainset, 
                             batch_size=batch_size,
                             shuffle=(train_sampler is None),
                             num_workers=workers,
                             collate_fn=collate_func,
                             pin_memory=True,
                             sampler=train_sampler)
    val_loader = DataLoader(dataset=validset, 
                             batch_size=batch_size, # validate require no grad.
                             shuffle=(val_sampler is None),
                             num_workers=workers,
                             collate_fn=collate_func,
                             pin_memory=True,
                             sampler=val_sampler)
        
    train(model, train_loader, val_loader, train_sampler, val_sampler, system_config, args)


if __name__ == "__main__":
    args = parse_args()

    distributed = args.distributed
    world_size  = args.world_size

    if distributed and world_size < 0:
        raise ValueError("world size must be greater than 0 in distributed training")

    ngpus_per_node  = torch.cuda.device_count()
    print("ngpus_per_node {}".format(ngpus_per_node))
    if distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main(None, ngpus_per_node, args)
    
    # evaulate 
    # print("evaluating...")
    # os.system("python evaluate.py {} --split {}  --testiter {} --batch_size {} >> evalute.out".format(args.cfg_file, "test", 100, 64))