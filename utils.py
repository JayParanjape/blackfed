import numpy as np
import torch
import torch.nn.functional as F
import argparse
import torch.nn as nn
# import monai

def no_blank_miou(y_true, y_pred, ignore_idx=-1):
    #y_true and y_pred are BXCXHXW
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    if ignore_idx==-1:
        collapsed_true = (1+torch.argmax(y_true, axis=1))*(torch.any(y_true, axis=1))
        tmp = 0.5*torch.ones((y_pred.shape[0],1,y_pred.shape[2],y_pred.shape[3]))
        collapsed_pred = torch.argmax(torch.cat([tmp, y_pred],dim=1), dim=1)

        intersection = torch.sum(((collapsed_pred == collapsed_true)&(collapsed_true>0)), axis=(-1,-2))
        union = torch.sum((collapsed_pred>0),axis=(-1,-2)) + torch.sum((collapsed_true>0),axis=(-1,-2)) - intersection
    else:
        collapsed_true = (torch.argmax(y_true, dim=1))*(torch.any(y_true, dim=1))
        collapsed_pred = torch.argmax(y_pred, dim=1)
        intersection = torch.sum(((collapsed_pred == collapsed_true)&(collapsed_true!=ignore_idx)), axis=(-1,-2))
        union = torch.sum((collapsed_pred!=ignore_idx),axis=(-1,-2)) + torch.sum((collapsed_true!=ignore_idx),axis=(-1,-2)) - intersection

    return torch.sum((intersection+(1e-5))/(union+(1e-5)))

def dice_coef(y_true, y_pred, smooth=1):
    # print(y_pred.shape, y_true.shape)
    intersection = torch.sum(y_true * y_pred,axis=(-1,-2))
    union = torch.sum(y_true, axis=(-1,-2)) + torch.sum(y_pred, axis=(-1,-2))
    dice = ((2. * intersection + smooth)/(union + smooth)).mean()
    # print(dice)
    return dice

def iou_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred),axis=(-1,-2))
    union = torch.sum(y_true,axis=(-1,-2))+torch.sum(y_pred,axis=(-1,-2))-intersection
    iou = ((intersection + smooth) / (union + smooth)).mean()
    return iou

def running_stats(y_true, y_pred, smooth = 1):
    intersection = torch.sum(y_true * y_pred,axis=(-1,-2))
    union = torch.sum(y_true, axis=(-1,-2)) + torch.sum(y_pred, axis=(-1,-2))
    return intersection, union

def dice_collated(running_intersection, running_union, smooth =1):
    if len(running_intersection.size())>=2:
        dice = (torch.mean((2. * running_intersection + smooth)/(running_union + smooth),dim=1)).sum()
    else:
        dice = ((2. * running_intersection + smooth)/(running_union + smooth)).sum()
    return dice

def dice_batchwise(running_intersection, running_union, smooth =1):
    dice = ((2. * running_intersection + smooth)/(running_union + smooth))
    return dice

def compute_hd95(preds, gt):
    # print(preds.shape)
    # print(gt.shape)
    cd = monai.metrics.compute_hausdorff_distance(preds, gt, include_background=True, percentile=95)
    return cd.mean().item()


def dice_loss(y_pred, y_true, ignore_idx=-1):
    # print('ytrue shape: ', y_true.shape)
    # print('ypred shape: ', y_pred.shape)
    if ignore_idx>-1:
        numerator_ignore = 2*torch.sum(y_true[:,ignore_idx,:,:]*y_pred[:,ignore_idx,:,:])
        denominator_ignore = torch.sum(y_true[:,ignore_idx,:,:]+y_pred[:,ignore_idx,:,:])
    else:
        numerator_ignore = 0
        denominator_ignore = 0

    numerator = (2 * torch.sum(y_true * y_pred)) - numerator_ignore
    denominator = torch.sum(y_true + y_pred) - denominator_ignore

    return 1 - ((numerator+1) / (denominator+1))

def weighted_ce_loss(y_pred, y_true, alpha=64, smooth=1):
    weight1 = torch.sum(y_true==1,dim=(-1,-2))+smooth
    weight0 = torch.sum(y_true==0, dim=(-1,-2))+smooth
    multiplier_1 = weight0/(weight1*alpha)
    multiplier_1 = multiplier_1.view(-1,1,1)
    # print(multiplier_1.shape)
    # print(y_pred.shape)
    # print(y_true.shape)

    loss = -torch.mean(torch.mean((multiplier_1*y_true*torch.log(y_pred)) + (1-y_true)*(torch.log(1-y_pred)),dim=(-1,-2)))
    return loss

def focal_loss(y_pred, y_true, alpha_def=0.75, gamma=2):
    # print('going back to the default value of alpha')
    alpha = alpha_def
    ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true.float(), reduction="none")
    assert (ce_loss>=0).all()
    p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    # 1/0
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    loss = alpha_t * loss
    loss = torch.mean(loss, dim=(-1,-2))
    return loss.mean()

def multiclass_focal_loss(y_pred, y_true, alpha = 0.75, gamma=3):
    if len(y_pred.shape)==4:
        y_pred = y_pred.squeeze()
    ce = y_true*(-torch.log(y_pred))
    weight = y_true * ((1-y_pred)**gamma)
    fl = torch.sum(alpha*weight*ce, dim=(-1,-2))
    return torch.mean(fl)

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""
