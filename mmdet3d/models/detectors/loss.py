import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def Distill_loss(feats_s, feats_t, target, ratio):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 17
    valid_mask = target != 255
    valid = (valid_mask * nonzeros).squeeze(0)
    feats_s = feats_s.squeeze()[valid]
    feats_t = feats_t.squeeze()[valid]
    loss = nn.KLDivLoss(reduction="mean")(F.log_softmax(feats_s, dim=1), feats_t)
    return loss * ratio


def CE_ssc_loss(pred, target, class_weights, ignore_index):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss


def sem_scal_loss(pred, ssc_target, ignore_index, camera_mask=None):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != ignore_index

    if camera_mask is not None:
        mask = torch.logical_and(mask, camera_mask)

    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


def geo_scal_loss(pred, ssc_target, ignore_index, non_empty_idx=0, camera_mask=None):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, non_empty_idx, :, :, :]
    nonempty_probs = 1 - empty_probs

    mask = ssc_target != non_empty_idx

    if camera_mask is not None:
        mask = torch.logical_and(mask, camera_mask)

    nonempty_target = torch.zeros_like(nonempty_probs).to(nonempty_probs.device)
    nonempty_target[mask] = 1
    nonempty_target = nonempty_target.float()

    nonempty_target = nonempty_target.reshape(-1)
    empty_probs = empty_probs.reshape(-1)
    nonempty_probs = nonempty_probs.reshape(-1)

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


class l1_loss(nn.Module):
    def __init__(self):
        super(l1_loss, self).__init__()
    
    def forward(self, color_est, color_gt):
        loss = torch.sum(torch.mean(torch.abs(color_est - color_gt), dim=0))
        return loss


class l2_loss(nn.Module):
    def __init__(self):
        super(l2_loss, self).__init__()
    
    def forward(self, traj_est, traj_gt):
        loss = torch.sum(torch.mean((torch.abs(traj_est - traj_gt) ** 2), dim=0))
        return loss