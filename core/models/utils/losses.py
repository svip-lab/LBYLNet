import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

def _build_target(pred_bboxes, targets, anchors):
    # pred_bboxes (batch_size, num_bboxes, 5) 
    # targets     (batch_size, targets, anchors)
    device = pred_bboxes.device
    anchors = anchors.to(device)
    # anchor = torch.from_numpy(anchors).float().to(device)
    nB = pred_bboxes.size(0)
    nA = pred_bboxes.size(1)
    nG = pred_bboxes.size(2) # number of grid
    
    # iou_scores = torch.FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    tx = torch.FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    ty = torch.FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    tw = torch.FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    th = torch.FloatTensor(nB, nA, nG, nG).fill_(0).to(device)
    obj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(0).to(device)

    targets = targets * nG
    gxy = (targets[:, :2] + targets[:, 2:]).float() / 2
    gwh = (targets[:, 2:] - targets[:, :2]).float()

    # find max overlap in original scales
    # calculate iou with bbox prior in the original scales
    ious = torch.stack([bbox_wh_iou(anchors[i], gwh) for i in range(len(anchors))])
    best_ious, best_n = ious.max(0)
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    tx[range(0, nB), best_n, gj, gi] = gx - gx.floor()
    ty[range(0, nB), best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[range(0, nB), best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[range(0, nB), best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    obj_mask[range(0, nB), best_n, gj, gi] = 1


    return gi, gj, gw, gh, best_ious, best_n, tx, ty, tw, th, obj_mask# iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    
class Loss(nn.Module):
    def __init__(self, off_weight, anchors, input_size):
        super(Loss, self).__init__()
        self.img_dim = input_size
        self.off_weight = off_weight
        self.anchors = np.array([list(anchor) for anchor in anchors])
        self.anchors = self.anchors.reshape(3, 3, 2)
        self.anchors = [self.anchors[i] for i in range(0, 3)]
        self.num_anchors = len(anchors)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss  = nn.CrossEntropyLoss(reduction='none')
        self.grid_size = 0

    def reshape(self, x):
        batch_size, _, h, w = x.shape
        return (x.view(batch_size, 3, 5, h, w) \
            .permute(0, 1, 3, 4, 2)       \
            .contiguous()                 \
            .view(batch_size, 3, h, w, 5) \
            .view(batch_size, 3*h*w,   5)
            )

    def loss(self, outs, targets):
        xs, best_ious, txs, tys, tws, ths, obj_masks = [],[],[],[],[],[],[]
        
        for i, (x, a) in enumerate(zip(outs, self.anchors)):
            _, _, _, _, best_iou, best_n, tx, ty, tw, th, obj_mask \
                = self.target_generator(x, a, targets)
            batch_size, _, h, w = x.shape
            x = self.reshape(x)
            xs.append(x)
            best_ious.append(best_iou)
            txs.append(tx.view(batch_size, -1))
            tys.append(ty.view(batch_size, -1))
            tws.append(tw.view(batch_size, -1))
            ths.append(th.view(batch_size, -1))
            obj_masks.append(obj_mask.view(batch_size, -1))

        index = torch.stack(best_ious, dim=1).max(dim=1)[1]
        for batch_id, mask_id in enumerate(index):
            for i in range(len(obj_masks)):
                if i == mask_id:
                    continue
                obj_masks[i][batch_id, ...] = 0

        obj_masks = torch.cat(obj_masks, dim=1)
        if hasattr(obj_mask, 'bool'):
            obj_masks = obj_masks.bool()

        assert(obj_masks.sum().item() == len(index))
        txs = torch.cat(txs, dim=1)
        tys = torch.cat(tys, dim=1)
        tws = torch.cat(tws, dim=1)
        ths = torch.cat(ths, dim=1)
        xs  = torch.cat(xs,  dim=1)
        
        loss_x = self.mse_loss(_sigmoid(xs[..., 0][obj_masks]), txs[obj_masks])
        loss_y = self.mse_loss(_sigmoid(xs[..., 1][obj_masks]), tys[obj_masks])
        loss_w = self.mse_loss(xs[..., 2][obj_masks], tws[obj_masks])
        loss_h = self.mse_loss(xs[..., 3][obj_masks], ths[obj_masks])
        off_loss = (loss_x + loss_y + loss_w + loss_h).mean() 

        loss_conf = self.ce_loss(xs[..., 4], obj_masks.max(1)[1]).mean()
        loss = off_loss * self.off_weight + loss_conf
        return loss.unsqueeze(0), loss_conf.unsqueeze(0), self.off_weight * off_loss.unsqueeze(0) 

    def forward(self, outs, targets):
        # map to (0, 1)
        if targets is None:
            return [self.target_generator(x, a) for x, a in zip(outs, self.anchors)]

        targets = targets.float() / self.img_dim
        return self.loss(outs, targets)

    def compute_grid_offsets(self, grid_size, anchors, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).to(self.device).float()
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).to(self.device).float()
        self.scaled_anchors = torch.FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in anchors]).to(self.device)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, len(anchors), 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, len(anchors), 1, 1))


    def target_generator(self, x, layer_anchors, targets=None):
        # x [b, 3, 5, h, w]
        self.device = x.device
        # targets = targets.to(self.device)
        batch_size, _, h, w = x.shape
        grid_size = h
        prediction = (
            x.view(batch_size, 3, 5, h, w)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        ).float()

        x = torch.sigmoid(prediction[..., 0]) # center x
        y = torch.sigmoid(prediction[..., 1]) # center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = prediction[..., 4]

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, layer_anchors, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = torch.FloatTensor(prediction[..., :4].shape).to(self.device)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4) * self.stride,
                pred_conf.view(batch_size, -1, 1)
            ),
            -1,
        )
        if targets is None:
            return output

        gi, gj, gw, gh, best_ious, best_n, tx, ty, tw, th, object_mask = _build_target(pred_boxes, targets, self.scaled_anchors)
        
        return gi, gj, gw, gh, best_ious, best_n, tx, ty, tw, th, object_mask
