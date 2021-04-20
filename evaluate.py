#!/usr/bin/env python
import os
import os.path as osp
import json
import torch
import pprint
import argparse
import importlib
import cv2
import time
from tqdm import tqdm
import numpy as np
import pdb
from core.dbs import datasets
from core.test.test import _decode, _bbox_iou, _visualize, _decode_anchorbased
from core.sampler.sampler import Referring
from core.sampler.collate_fn import collate_fn, collate_fn_bert
from core.sampler.utils import letterbox
from core.config import SystemConfig
from core.nnet.nnet_factory import NetworkFactory
from core.utils import make_anchors
from core.paths import get_file_path
torch.backends.cudnn.benchmark = False
import matplotlib.pyplot as plt
from matplotlib import cm
from core.models.net.lbylnet import LBYLNet 

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="validation", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--visulize", action="store_true")
    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not osp.exists(directory):
            os.makedirs(directory)

def test(db, system_config, model, args):
    split    = args.split
    testiter = args.testiter
    # debug    = args.debug
    suffix   = args.suffix
    bs       = args.batch_size
    result_dir = system_config.result_dir
    test_iter = system_config.nb_epoch if testiter is None else testiter
    result_dir = osp.join(result_dir, str(testiter), split)
    
    system_config.lr = system_config.learning_rate

    if suffix is not None:
        result_dir = osp.join(result_dir, suffix)

    make_dirs([result_dir])
    print("loading parameters at epoch: {}".format(test_iter))
    print("building neural network...")
    nnet = NetworkFactory(system_config, model)
    print("loading parameters...")
    nnet.load_params(test_iter)

    dataset = Referring(db, system_config, data_aug=False, \
        debug=False, shuffle=False, test=True)

    nnet.cuda()
    nnet.eval_mode()
    input_size = db.configs['input_size']
    output_size = db.configs['output_sizes']
    result = {}
    ap_metric = {}
    total_time = 0.0
    for i in range(9):
        ap_metric[0.5+0.05*i] = 0.0

    batch_size = bs
    for k_ind in tqdm(range(0, len(db.db_inds), batch_size)):
        
        
        end = time.time()
        if k_ind + batch_size > len(db.db_inds)-1:
            end_batch = len(db.db_inds) - 1
        else:
            end_batch = k_ind + batch_size

        db_inds = db.db_inds[k_ind:end_batch]
        image_paths = [db.image_path(ind) for ind in db_inds]
        expressions = [db.phrase(ind) for ind in db_inds]
        gt_bboxes   = [db.annotation_box(ind) for ind in db_inds]
        batch       = [dataset[ind] for ind in db_inds]
        images, phrases, original_shapes = tuple(zip(*tuple(batch)))
        images = np.stack(images)
        images = torch.from_numpy(images)
        phrases = torch.stack(phrases)

        out = nnet.test(images, phrases)
        bboxes = _decode_anchorbased(out)
        for ind, bbox in enumerate(bboxes):
            # revert to original scales
            height, width = original_shapes[ind][0:2]
            reshape_ratio = min(input_size[0] / height, \
                                input_size[1] / width)
            resize_shape = round(height * reshape_ratio), round(width * reshape_ratio)
            dh = (input_size[0] - resize_shape[0]) / 2 # height padding
            dw = (input_size[1] - resize_shape[1]) / 2 # width padding

            bbox[0:4:2] = (bbox[0:4:2] - dw) / reshape_ratio
            bbox[1:4:2] = (bbox[1:4:2] - dh) / reshape_ratio
            bbox[0:4:2] = np.clip(bbox[0:4:2], 0, width-1)
            bbox[1:4:2] = np.clip(bbox[1:4:2], 0, height-1)
            iou = _bbox_iou(bbox, gt_bboxes[ind])

            for i in range(0, 9):
                if iou > 0.5+0.05*i:
                    ap_metric[0.5+0.05*i] += 1.0

            value = {
                 'iou': iou, 
                 'image_name': osp.basename(image_paths[ind]),
                 'pred_bbox': bbox.tolist(),
                 'gt_bbox': gt_bboxes[ind].tolist(),
            }
            result[k_ind+ind] = value
            if (k_ind + ind) % 1000 == 0:
                visu_dir = osp.join(result_dir, 'visulize')
                if not osp.exists(visu_dir):
                    os.mkdir(visu_dir)

                image = cv2.imread(image_paths[ind])
                _visualize(image, bbox, gt_bboxes[ind], expressions[ind], \
                          save_path=osp.join(visu_dir, "{}-{}".format(k_ind+ind, osp.basename(image_paths[ind]))))


    for i in range(0, 9):
        key = 0.5+0.05*i
        ap_metric[key] /= len(db.db_inds)
        print("Average Precision  (AP) @[ IoU={:.2f}\t| area=\tall | = {:.4f}".format(key, ap_metric[key]))
    with open(osp.join(result_dir, "metrics.json"), "w") as f:
        json.dump(ap_metric, f)
    with open(osp.join(result_dir, "results.json"), "w") as f:
        json.dump(result, f)

    print("{:.2f} s / image ".format(total_time / len(db.db_inds)))
    print("evaluate finish...")


def main(args):
    if args.suffix is None:
        cfg_file = osp.join("./configs", args.cfg_file + ".json")
    else:
        cfg_file = osp.join("./configs", args.cfg_file + "-{}.json".format(args.suffix))
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        config = json.load(f)
            
    config["system"]["snapshot_name"] = args.cfg_file
    system_config = SystemConfig().update_config(config["system"])
    anchors = make_anchors(system_config.dataset, 416)
    config["db"]["anchors"] = anchors
    config["db"]["corpus_path"] = get_file_path("..", "data", "refer", "data",  config["system"]["dataset"], "corpus.pth") 
    
    model = LBYLNet(system_config, config["db"])
    train_split = system_config.train_split
    val_split   = system_config.val_split
    test_split  = system_config.test_split

    if args.split not in ["training", "validation", "testing"]:
        split = args.split
    else:
        split = {
            "training": train_split,
            "validation": val_split,
            "testing": test_split
        }[args.split]

    print("loading all datasets...")
    dataset = system_config.dataset
    print("split: {}".format(split))
    testing_db = datasets['refer'](config["db"], split=split, sys_config=system_config)

    print("system config...")
    pprint.pprint(system_config.full)

    print("db config...")
    pprint.pprint(testing_db.configs)

    test(testing_db, system_config, model, args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
