#!/usr/bin/env python
import cv2
import torch
from core.test.test import _visualize
from core.groundors import Net 
# pick one model
cfg_file = "lbyl_bert_unc+_batch64"
detector = Net(cfg_file, iter=100)
# inference
image = cv2.imread('imgs/demo.jpeg')
phrase = 'the green gaint'
bbox = detector(image, phrase)
_visualize(image, pred_bbox=bbox, phrase=phrase, save_path='imgs/demo_out.jpg', color=(1, 174, 245), draw_phrase=True)