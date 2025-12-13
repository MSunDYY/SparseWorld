import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import argparse
import time
import torch
import sys, platform
from sklearn.neighbors import KDTree
from termcolor import colored
from pathlib import Path
from copy import deepcopy
from functools import reduce

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M ** 2 * cells[:, 2]).shape[0]


class Metric_mIoU():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.occ_names = ['free', 'occupied']
        self.occ_classes = 2
        self.occ_hist = np.zeros((self.occ_classes, self.occ_classes))

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist
    
    def compute_IoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist


    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

            # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)
        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        
        masked_occ_pred = np.zeros_like(masked_semantics_pred)
        masked_occ_pred[masked_semantics_pred != 17] = 1
        masked_occ_gt = np.zeros_like(masked_semantics_gt)
        masked_occ_gt[masked_semantics_gt != 17] = 1
        _, _occ_hist = self.compute_IoU(masked_occ_pred, masked_occ_gt, self.occ_classes)

        self.hist += _hist
        self.occ_hist += _occ_hist

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)))
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))

        # print(f'====================')
        # IoU = self.per_class_iu(self.occ_hist)
        # for ind_class in range(self.occ_classes):
        #     print(f'===> {self.occ_names[ind_class]} - IoU = ' + str(round(IoU[ind_class] * 100, 2)))

        miou_res = round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)

        return self.class_names, mIoU, self.cnt, miou_res
    
    def count_iou(self):
        IoU = self.per_class_iu(self.occ_hist)
        for ind_class in range(self.occ_classes):
            print(f'===> {self.occ_names[ind_class]} - IoU = ' + str(round(IoU[ind_class] * 100, 2)))
        
        iou_res = round(IoU[-1] * 100, 2)
        
        return self.occ_names, IoU, self.cnt, iou_res


class NuPlan_Metric_mIoU():
    def __init__(self,
                 save_dir='.',
                 num_classes=12,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.class_names = ['vehicle','place_holder1','place_holder2','place_holder3',
                            'czone_sign','bicycle','generic_object','pedestrian',
                            'traffic_cone','barrier','background','free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.occ_names = ['free', 'occupied']
        self.occ_classes = 2
        self.occ_hist = np.zeros((self.occ_classes, self.occ_classes))

        self.point_cloud_range = [-50., -50., -4., 50., 50., 4.]
        self.occupancy_size = [0.5, 0.5, 0.5]
        self.voxel_size = 0.5
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-10
        free voxel class: 11

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist
    
    def compute_IoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist


    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        
        masked_occ_pred = np.zeros_like(masked_semantics_pred)
        masked_occ_pred[masked_semantics_pred != 11] = 1
        masked_occ_gt = np.zeros_like(masked_semantics_gt)
        masked_occ_gt[masked_semantics_gt != 11] = 1
        _, _occ_hist = self.compute_IoU(masked_occ_pred, masked_occ_gt, self.occ_classes)

        self.hist += _hist
        self.occ_hist += _occ_hist

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)))
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))

        # print(f'====================')
        # IoU = self.per_class_iu(self.occ_hist)
        # for ind_class in range(self.occ_classes):
        #     print(f'===> {self.occ_names[ind_class]} - IoU = ' + str(round(IoU[ind_class] * 100, 2)))

        miou_res = round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)

        return self.class_names, mIoU, self.cnt, miou_res
    
    def count_iou(self):
        IoU = self.per_class_iu(self.occ_hist)
        for ind_class in range(self.occ_classes):
            print(f'===> {self.occ_names[ind_class]} - IoU = ' + str(round(IoU[ind_class] * 100, 2)))
        
        iou_res = round(IoU[-1] * 100, 2)
        
        return self.occ_names, IoU, self.cnt, iou_res


class Metric_FScore():
    def __init__(self,

                 leaf_size=10,
                 threshold_acc=0.6,
                 threshold_complete=0.6,
                 voxel_size=[0.4, 0.4, 0.4],
                 range=[-40, -40, -1, 40, 40, 5.4],
                 void=[17, 255],
                 use_lidar_mask=False,
                 use_image_mask=False, ) -> None:

        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.cnt=0
        self.tot_acc = 0.
        self.tot_cmpl = 0.
        self.tot_f1_mean = 0.
        self.eps = 1e-8



    def voxel2points(self, voxel):
        # occIdx = torch.where(torch.logical_and(voxel != FREE, voxel != NOT_OBSERVED))
        # if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        mask = np.logical_not(reduce(np.logical_or, [voxel == self.void[i] for i in range(len(self.void))]))
        occIdx = np.where(mask)

        points = np.concatenate((occIdx[0][:, None] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.range[0], \
                                 occIdx[1][:, None] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.range[1], \
                                 occIdx[2][:, None] * self.voxel_size[2] + self.voxel_size[2] / 2 + self.range[2]),
                                axis=1)
        return points

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera ):

        # for scene_token in tqdm(preds_dict.keys()):
        self.cnt += 1

        if self.use_image_mask:

            semantics_gt[mask_camera == False] = 255
            semantics_pred[mask_camera == False] = 255
        elif self.use_lidar_mask:
            semantics_gt[mask_lidar == False] = 255
            semantics_pred[mask_lidar == False] = 255
        else:
            pass

        ground_truth = self.voxel2points(semantics_gt)
        prediction = self.voxel2points(semantics_pred)
        if prediction.shape[0] == 0:
            accuracy=0
            completeness=0
            fmean=0

        else:
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            complete_distance, _ = prediction_tree.query(ground_truth)
            complete_distance = complete_distance.flatten()

            accuracy_distance, _ = ground_truth_tree.query(prediction)
            accuracy_distance = accuracy_distance.flatten()

            # evaluate completeness
            complete_mask = complete_distance < self.threshold_complete
            completeness = complete_mask.mean()

            # evalute accuracy
            accuracy_mask = accuracy_distance < self.threshold_acc
            accuracy = accuracy_mask.mean()

            fmean = 2.0 / (1 / (accuracy+self.eps) + 1 / (completeness+self.eps))

        self.tot_acc += accuracy
        self.tot_cmpl += completeness
        self.tot_f1_mean += fmean

    def count_fscore(self,):
        base_color, attrs = 'red', ['bold', 'dark']
        print(pcolor('\n######## F score: {} #######'.format(self.tot_f1_mean / self.cnt), base_color, attrs=attrs))



class Metric_mIoU_Temporal():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.occ_names = ['free', 'occupied']
        self.occ_classes = 2
        self.occ_hist_0s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_1s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_2s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_3s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_4s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_5s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_6s = np.zeros((self.occ_classes, self.occ_classes))

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist_0s = np.zeros((self.num_classes, self.num_classes))
        self.hist_1s = np.zeros((self.num_classes, self.num_classes))
        self.hist_2s = np.zeros((self.num_classes, self.num_classes))
        self.hist_3s = np.zeros((self.num_classes, self.num_classes))
        self.hist_4s = np.zeros((self.num_classes, self.num_classes))
        self.hist_5s = np.zeros((self.num_classes, self.num_classes))
        self.hist_6s = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist
    
    def compute_IoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist


    def add_batch(self,semantics_pred,semantics_gt_temp,mask_lidar_temp,mask_camera_temp):
        self.cnt += 1

        for idx in semantics_gt_temp.keys():
            semantics_gt = semantics_gt_temp[idx]
            mask_lidar = mask_lidar_temp[idx]
            mask_camera = mask_camera_temp[idx]

            curr_semantics_pred = semantics_pred[idx // 2]

            if self.use_image_mask:
                masked_semantics_gt = semantics_gt[mask_camera]
                masked_semantics_pred = curr_semantics_pred[mask_camera]
            elif self.use_lidar_mask:
                masked_semantics_gt = semantics_gt[mask_lidar]
                masked_semantics_pred = curr_semantics_pred[mask_lidar]
            else:
                masked_semantics_gt = semantics_gt
                masked_semantics_pred = curr_semantics_pred

                # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)

            # if True:
            #     mask = (masked_semantics_pred==15) & (masked_semantics_gt==17)
            #     masked_semantics_pred[mask] = masked_semantics_gt[mask]
            _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
            
            masked_occ_pred = np.zeros_like(masked_semantics_pred)
            masked_occ_pred[masked_semantics_pred != 17] = 1
            # masked_occ_pred[np.logical_and(masked_semantics_pred>=2, masked_semantics_pred<=10)] = 1
            # masked_occ_pred[np.logical_or(masked_semantics_pred<2,(masked_semantics_pred>10) * (masked_semantics_pred!=17))] =2
            masked_occ_gt = np.zeros_like(masked_semantics_gt)
            masked_occ_gt[masked_semantics_gt != 17] = 1
            # masked_occ_gt[np.logical_and(masked_semantics_gt>=2,masked_semantics_gt<=10)] =1
            # masked_occ_gt[np.logical_or(masked_semantics_gt<2,(masked_semantics_gt>10) * (masked_semantics_gt!=17))] =2

            _, _occ_hist = self.compute_IoU(masked_occ_pred, masked_occ_gt, self.occ_classes)

            if idx == 0:
                self.hist_0s += _hist
                self.occ_hist_0s += _occ_hist
            elif idx == 2:
                self.hist_1s += _hist
                self.occ_hist_1s += _occ_hist
            elif idx == 4:
                self.hist_2s += _hist
                self.occ_hist_2s += _occ_hist
            elif idx == 6:
                self.hist_3s += _hist
                self.occ_hist_3s += _occ_hist


    def count_miou(self):
        miou_res_list = []

        mIoU_0s = self.per_class_iu(self.hist_0s)
        print(f'===> mIoU of {self.cnt} samples at 0s: ' + str(
            round(np.nanmean(mIoU_0s[:self.num_classes - 1]) * 100, 2)))
        print(f'===> per class IoU of {self.cnt} samples at 0s:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_0s[ind_class] * 100, 2)))
        
        mIoU_1s = self.per_class_iu(self.hist_1s)
        print(f'===> mIoU of {self.cnt} samples at 1s: ' + str(round(np.nanmean(mIoU_1s[:self.num_classes-1]) * 100, 2)))
        print(f'===> per class IoU of {self.cnt} samples at 1s:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_1s[ind_class] * 100, 2)))

        mIoU_2s = self.per_class_iu(self.hist_2s)
        print(f'===> mIoU of {self.cnt} samples at 2s: ' + str(round(np.nanmean(mIoU_2s[:self.num_classes-1]) * 100, 2)))
        print(f'===> per class IoU of {self.cnt} samples at 2s:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_2s[ind_class] * 100, 2)))
        
        mIoU_3s = self.per_class_iu(self.hist_3s)
        print(f'===> mIoU of {self.cnt} samples at 3s: ' + str(round(np.nanmean(mIoU_3s[:self.num_classes-1]) * 100, 2)))
        print(f'===> per class IoU of {self.cnt} samples at 3s:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_3s[ind_class] * 100, 2)))

        mIoU_res_0s = round(np.nanmean(mIoU_0s[:self.num_classes-1]) * 100, 2)
        miou_res_1s = round(np.nanmean(mIoU_1s[:self.num_classes-1]) * 100, 2)
        miou_res_2s = round(np.nanmean(mIoU_2s[:self.num_classes-1]) * 100, 2)
        miou_res_3s = round(np.nanmean(mIoU_3s[:self.num_classes-1]) * 100, 2)

        miou_res_list.append(mIoU_res_0s)
        miou_res_list.append(miou_res_1s)
        miou_res_list.append(miou_res_2s)
        miou_res_list.append(miou_res_3s)

        return mIoU_1s, miou_res_list
    
    def count_iou(self):
        ind_class = -1
        iou_res_list = []

        IoU_0s = self.per_class_iu(self.occ_hist_0s)
        print(f'===> {self.occ_names[ind_class]} - IoU at 1s = ' + str(round(IoU_0s[ind_class] * 100, 2)))
        IoU_1s = self.per_class_iu(self.occ_hist_1s)
        print(f'===> {self.occ_names[ind_class]} - IoU at 1s = ' + str(round(IoU_1s[ind_class] * 100, 2)))
        IoU_2s = self.per_class_iu(self.occ_hist_2s)
        print(f'===> {self.occ_names[ind_class]} - IoU at 2s = ' + str(round(IoU_2s[ind_class] * 100, 2)))
        IoU_3s = self.per_class_iu(self.occ_hist_3s)
        print(f'===> {self.occ_names[ind_class]} - IoU at 3s = ' + str(round(IoU_3s[ind_class] * 100, 2)))

        iou_res_0s = round(IoU_0s[-1] * 100, 2)
        iou_res_1s = round(IoU_1s[-1] * 100, 2)
        iou_res_2s = round(IoU_2s[-1] * 100, 2)
        iou_res_3s = round(IoU_3s[-1] * 100, 2)

        iou_res_list.append(iou_res_0s)
        iou_res_list.append(iou_res_1s)
        iou_res_list.append(iou_res_2s)
        iou_res_list.append(iou_res_3s)
        
        return iou_res_list


class NuPlan_Metric_mIoU_Temporal():
    def __init__(self,
                 save_dir='.',
                 num_classes=12,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.class_names = ['vehicle','place_holder1','place_holder2','place_holder3',
                            'czone_sign','bicycle','generic_object','pedestrian',
                            'traffic_cone','barrier','background','free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.occ_names = ['free', 'occupied']
        self.occ_classes = 2
        # self.occ_hist = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_0s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_1s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_2s = np.zeros((self.occ_classes, self.occ_classes))
        self.occ_hist_3s = np.zeros((self.occ_classes, self.occ_classes))

        self.point_cloud_range = [-50., -50., -4., 50., 50., 4.]
        self.occupancy_size = [0.5, 0.5, 0.5]
        self.voxel_size = 0.5
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        # self.hist = np.zeros((self.num_classes, self.num_classes))
        self.hist_0s = np.zeros((self.num_classes, self.num_classes))
        self.hist_1s = np.zeros((self.num_classes, self.num_classes))
        self.hist_2s = np.zeros((self.num_classes, self.num_classes))
        self.hist_3s = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-10
        free voxel class: 11

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        return round(np.nanmean(mIoUs) * 100, 2), hist
    
    def compute_IoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)

        return round(np.nanmean(mIoUs) * 100, 2), hist


    def add_batch(self,semantics_pred,semantics_gt_temp,mask_lidar_temp,mask_camera_temp):
        self.cnt += 1

        for idx in semantics_gt_temp.keys():
            semantics_gt = semantics_gt_temp[idx]
            mask_lidar = mask_lidar_temp[idx]
            mask_camera = mask_camera_temp[idx]

            curr_semantics_pred = semantics_pred[idx // 2]

            if self.use_image_mask:
                masked_semantics_gt = semantics_gt[mask_camera]
                masked_semantics_pred = curr_semantics_pred[mask_camera]
            elif self.use_lidar_mask:
                masked_semantics_gt = semantics_gt[mask_lidar]
                masked_semantics_pred = curr_semantics_pred[mask_lidar]
            else:
                masked_semantics_gt = semantics_gt
                masked_semantics_pred = curr_semantics_pred
            
            _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
            
            masked_occ_pred = np.zeros_like(masked_semantics_pred)
            masked_occ_pred[masked_semantics_pred != 11] = 1
            masked_occ_gt = np.zeros_like(masked_semantics_gt)
            masked_occ_gt[masked_semantics_gt != 11] = 1
            _, _occ_hist = self.compute_IoU(masked_occ_pred, masked_occ_gt, self.occ_classes)

            if idx == 0:
                self.hist_0s += _hist
                self.occ_hist_0s += _occ_hist
            elif idx == 2:
                self.hist_1s += _hist
                self.occ_hist_1s += _occ_hist
            elif idx == 4:
                self.hist_2s += _hist
                self.occ_hist_2s += _occ_hist
            elif idx == 6:
                self.hist_3s += _hist
                self.occ_hist_3s += _occ_hist

    def count_miou(self):
        miou_res_list = []
        
        mIoU_1s = self.per_class_iu(self.hist_1s)
        print(f'===> mIoU of {self.cnt} samples at 1s: ' + str(round(np.nanmean(mIoU_1s[:self.num_classes-1]) * 100, 2)))
        print(f'===> per class IoU of {self.cnt} samples at 1s:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_1s[ind_class] * 100, 2)))

        mIoU_2s = self.per_class_iu(self.hist_2s)
        print(f'===> mIoU of {self.cnt} samples at 2s: ' + str(round(np.nanmean(mIoU_2s[:self.num_classes-1]) * 100, 2)))
        print(f'===> per class IoU of {self.cnt} samples at 2s:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_2s[ind_class] * 100, 2)))
        
        mIoU_3s = self.per_class_iu(self.hist_3s)
        print(f'===> mIoU of {self.cnt} samples at 3s: ' + str(round(np.nanmean(mIoU_3s[:self.num_classes-1]) * 100, 2)))
        print(f'===> per class IoU of {self.cnt} samples at 3s:')
        for ind_class in range(self.num_classes):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU_3s[ind_class] * 100, 2)))

        miou_res_1s = round(np.nanmean(mIoU_1s[:self.num_classes-1]) * 100, 2)
        miou_res_2s = round(np.nanmean(mIoU_2s[:self.num_classes-1]) * 100, 2)
        miou_res_3s = round(np.nanmean(mIoU_3s[:self.num_classes-1]) * 100, 2)

        miou_res_list.append(miou_res_1s)
        miou_res_list.append(miou_res_2s)
        miou_res_list.append(miou_res_3s)

        return mIoU_1s, miou_res_list
    
    def count_iou(self):
        ind_class = -1
        iou_res_list = []

        IoU_1s = self.per_class_iu(self.occ_hist_1s)
        print(f'===> {self.occ_names[ind_class]} - IoU at 1s = ' + str(round(IoU_1s[ind_class] * 100, 2)))
        IoU_2s = self.per_class_iu(self.occ_hist_2s)
        print(f'===> {self.occ_names[ind_class]} - IoU at 2s = ' + str(round(IoU_2s[ind_class] * 100, 2)))
        IoU_3s = self.per_class_iu(self.occ_hist_3s)
        print(f'===> {self.occ_names[ind_class]} - IoU at 3s = ' + str(round(IoU_3s[ind_class] * 100, 2)))
        
        iou_res_1s = round(IoU_1s[-1] * 100, 2)
        iou_res_2s = round(IoU_2s[-1] * 100, 2)
        iou_res_3s = round(IoU_3s[-1] * 100, 2)

        iou_res_list.append(iou_res_1s)
        iou_res_list.append(iou_res_2s)
        iou_res_list.append(iou_res_3s)
        
        return iou_res_list