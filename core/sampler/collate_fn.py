import torch
import numpy as np
import pdb

def collate_fn(batch_data):
    images, phrases, bboxes = tuple(zip(*tuple(batch_data)))
    images = np.stack(images)
    images = torch.from_numpy(images)
    phrases = torch.stack(phrases)
    bboxes = np.stack(bboxes)
    bboxes = torch.from_numpy(bboxes)
        
    return {
        "images": images,
        "phrases": phrases,
        "labels": bboxes
    }

def collate_fn_bert(batch_data):
    images, bboxes, inputs, masks = tuple(zip(*tuple(batch_data)))
    images = np.stack(images)
    images = torch.from_numpy(images)
    bboxes = np.stack(bboxes)
    bboxes = torch.from_numpy(bboxes)
    inputs = torch.tensor(np.stack(inputs), dtype=torch.long)
    masks =  torch.tensor(np.stack(masks), dtype=torch.long)
    return {
        'images': images,
        'phrases': inputs,
        'labels': bboxes,
        # 'masks': masks
    }