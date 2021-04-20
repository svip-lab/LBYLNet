import os
import cv2
import json
import numpy as np
import torch

from tqdm import tqdm

from ..utils import Timer
from ..vis_utils import draw_bboxes
# from ..sample.utils import crop_image
from ..sampler.sampler import Referring
import pdb


def _decode_anchorbased(out):
    # feature
    outs = torch.cat(out, dim=1)
    confidence, index = outs[..., 4].max(1)
    x = outs[range(outs.shape[0]), index, 0]
    y = outs[range(outs.shape[0]), index, 1]
    w = outs[range(outs.shape[0]), index, 2]
    h = outs[range(outs.shape[0]), index, 3]
    bboxes = torch.stack((x-w/2, y-h/2, x+w/2, y+h/2), dim=1).cpu().data.numpy()
    bboxes = [bbox for bbox in bboxes]
    return bboxes


def _topk(scores, k=1):
    batch, _, height, width = scores.size()
    topk_scroes, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_ys, topk_xs

def _decode(out):
    tl_heats, br_heats, tl_regrs, br_regrs = out
    batch, _, height, width = tl_heats.shape
    tl_ys, tl_xs = _topk(tl_heats, k=1)
    br_ys, br_xs = _topk(br_heats, k=1)
    bboxes = []
    for ind in range(batch):
        # TODO only support k = 1 here
        tly = tl_ys[ind][0]
        tlx = tl_xs[ind][0]
        bry = br_ys[ind][0]
        brx = br_xs[ind][0]
        tl_off_x, tl_off_y = tl_regrs[ind, :, tly.to(torch.int), tlx.to(torch.int)]
        br_off_x, br_off_y = br_regrs[ind, :, bry.to(torch.int), brx.to(torch.int)]
        bbox = tlx + tl_off_x, tly + tl_off_y, brx + br_off_x, bry + br_off_y
        bbox = [x.cpu().data.item() for x in bbox]
        bbox = np.array(bbox)
        bboxes.append(bbox)

    return bboxes

def _bbox_iou(bbox1, bbox2):
    iou = np.zeros((4, ))
    iou[:2] = np.where(bbox1[:2]>bbox2[:2],bbox1[:2], bbox2[:2]) # element wise max
    iou[2:] = np.where(bbox1[2:]<bbox2[2:],bbox1[2:], bbox2[2:]) # element wise min
    w, h = np.clip(iou[2:]-iou[:2], 0, iou[2:]-iou[:2])
    iou_area = w * h
    w, h = bbox1[2:] - bbox1[:2]
    bbox1_area = w * h
    w, h = bbox2[2:] - bbox2[:2]
    bbox2_area = w * h
    return iou_area / (bbox1_area + bbox2_area - iou_area + 1e-16)


def _visualize(input, pred_bbox=None, gt_bbox=None, phrase=None, font_size=0.5, save_path=None, draw_phrase=True, color=None):
    image = input.copy()
    h, w = image.shape[:2]
    gtbox_size = cv2.getTextSize("ground truth", cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
    prbox_size = cv2.getTextSize("prediction", cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
    expr_size  = cv2.getTextSize(phrase, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    
    def _draw_title(bbox, text, color):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
        if bbox[1] - text_size[1] - 2 < 0:
            cv2.rectangle(image, 
                          (bbox[0],bbox[1] + 2),
                          (bbox[0]+text_size[0], bbox[1]+text_size[1]),
                          color, -1)
            cv2.putText(image, text,
                        (bbox[0], bbox[1]+2+text_size[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1)
        else:
            cv2.rectangle(image, 
                          (bbox[0], bbox[1]-text_size[1]-2),
                          (bbox[0]+text_size[0], bbox[1]-2),
                          color, -1)
            cv2.putText(image, text,
                        (bbox[0], bbox[1]-2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1)

    pred_color = (255.0, 0.0, 0.0)
    gt_color   = (0.0, 255.0, 255.0)
    if color:
        pred_color = color
        gt_color = color
    # _draw_title(pred_bbox, "prediction", pred_color)
    # _draw_title(gt_bbox, "ground truth", gt_color)
    if gt_bbox is not None:
        gt_bbox   = gt_bbox.copy().astype(int)
        cv2.rectangle(image, 
                 (gt_bbox[0], gt_bbox[1]),
                 (gt_bbox[2], gt_bbox[3]),
                 gt_color, 10)

    if pred_bbox is not None:
        pred_bbox = pred_bbox.copy().astype(int)
        cv2.rectangle(image, 
                 (pred_bbox[0], pred_bbox[1]),
                 (pred_bbox[2], pred_bbox[3]),
                 pred_color, 10)


    bottom = expr_size[1] * 2
    oh, ow = h + expr_size[1] // 2 + 6, w // 2 - expr_size[0] // 2
    ow = 0 if ow < 0 else ow
    if draw_phrase and phrase:
        image = cv2.copyMakeBorder(image, \
                               0, bottom, \
                               0, 0, \
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))  # padded square
        cv2.putText(image, phrase, (ow, oh), \
            cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
        )
    if save_path:
        cv2.imwrite(save_path, image)
    return image