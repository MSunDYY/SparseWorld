# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore
from .ray import generate_rays, generate_rays_dense

from PIL import Image
from torchvision import transforms

import random

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

nusc_class_nums = torch.Tensor([
    2854504, 7291443, 141614, 4239939, 32248552, 
    1583610, 364372, 2346381, 582961, 4829021, 
    14073691, 191019309, 6249651, 55095657, 
    58484771, 193834360, 131378779
])
dynamic_class = [0, 1, 3, 4, 5, 7, 9, 10]


def generate_dense_coors(h, w):
    pixel_origin = np.array([0., 0.])
    pixel_size = np.array([1., 1.])

    image_size = (h, w)
    image_bnds = np.zeros((2, 2))
    image_bnds[:, 0] = pixel_origin
    image_bnds[:, 1] = pixel_origin + np.array(image_size)

    image_dim = np.ceil(image_bnds[:, 1] - image_bnds[:, 0]).copy(order='C').astype('int')
    xv, yv = np.meshgrid(range(image_dim[0]), range(image_dim[1]), indexing='ij')
    ref_2d = np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1)], axis=0).astype(np.float64).T

    coors = ref_2d
    return coors

def load_depth(img_file_path, gt_path):
    file_name = os.path.split(img_file_path)[-1]
    cam_depth = np.fromfile(os.path.join(gt_path, f'{file_name}.bin'),
        dtype=np.float32,
        count=-1).reshape(-1, 3)
    
    coords = cam_depth[:, :2].astype(np.int16)
    depth_label = cam_depth[:,2]
    return coords, depth_label

def load_seg_label(img_file_path, gt_path, img_size=[900,1600], mode='lidarseg'):
    if mode=='lidarseg':  # proj lidarseg to img
        coor, seg_label = load_depth(img_file_path, gt_path)
        seg_map = np.zeros(img_size)
        seg_map[coor[:, 1],coor[:, 0]] = seg_label
    else:
        file_name = os.path.join(gt_path, f'{os.path.split(img_file_path)[-1]}.npy')
        seg_map = np.load(file_name)
    return seg_map

def load_img(img_file_path):
    img = Image.open(img_file_path).convert("RGB")
    img = np.array(img, dtype=np.float32, copy=False) / 255.0
    return img

def array_to_img(x):
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    img = Image.fromarray(x.astype('uint8'), 'RGB')
    return img

def get_sensor_transforms(cam_info, cam_name):
    w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
    # sweep sensor to sweep ego
    sensor2ego_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    sensor2ego_tran = torch.Tensor(
        cam_info['cams'][cam_name]['sensor2ego_translation'])
    sensor2ego = sensor2ego_rot.new_zeros((4, 4))
    sensor2ego[3, 3] = 1
    sensor2ego[:3, :3] = sensor2ego_rot
    sensor2ego[:3, -1] = sensor2ego_tran
    # sweep ego to global
    w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
    ego2global_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    ego2global_tran = torch.Tensor(
        cam_info['cams'][cam_name]['ego2global_translation'])
    ego2global = ego2global_rot.new_zeros((4, 4))
    ego2global[3, 3] = 1
    ego2global[:3, :3] = ego2global_rot
    ego2global[:3, -1] = ego2global_tran

    return sensor2ego, ego2global


@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def __init__(self, 
                use_rays=False,
                if_dense=False,
                semantic_gt_path=None,
                depth_gt_path=None,
                aux_frames=[-1,1],
                max_ray_nums=0,
                wrs_use_batch=False,
                **kwargs):
        super().__init__(**kwargs)
        self.use_rays = use_rays
        self.if_dense = if_dense
        self.semantic_gt_path = semantic_gt_path
        self.depth_gt_path = depth_gt_path
        self.aux_frames = aux_frames
        self.max_ray_nums = max_ray_nums

        if wrs_use_batch:   # compute with batch data
            self.WRS_balance_weight = None
        else:               # compute with total dataset
            self.WRS_balance_weight = torch.exp(0.005 * (nusc_class_nums.max() / nusc_class_nums - 1))

        self.dynamic_class = torch.tensor(dynamic_class)

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    
    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        # idx = 0
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)
        # return 1
    
    # def load_annotations(self, ann_file):
    #     """Load annotations from ann_file.

    #     Args:
    #         ann_file (str): Path of the annotation file.

    #     Returns:
    #         list[dict]: List of annotations sorted by timestamps.
    #     """
    #     data = mmcv.load(ann_file, file_format='pkl')
    #     data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    #     data_infos = data_infos[::self.load_interval]
    #     self.metadata = data['metadata']
    #     self.version = self.metadata['version']

    #     split_root = 'dataset/splits-nuscenes/'
    #     scenes = []
    #     with open(split_root + f'train-450scenes.txt', 'r', encoding='utf-8') as f:
    #         for ann in f.readlines():
    #             ann = ann.strip('\n')
    #             scenes.append(ann)
    #     self.scenes = scenes

    #     split_data_infos = []
    #     for data_info in data_infos:
    #         if data_info['scene_name'] in self.scenes:
    #             split_data_infos.append(data_info)

    #     return split_data_infos

    def get_rays(self, index):
        info = self.data_infos[index]

        sensor2egos = []
        ego2globals = []
        intrins = []
        coors = []
        label_depths = []
        label_segs = []
        label_imgs = []
        img_paths = []
        img_tensors = []
        time_ids = {}
        idx = 0

        for time_id in [0] + self.aux_frames:
            time_ids[time_id] = []
            select_id = max(index + time_id, 0)
            if select_id>=len(self.data_infos) or self.data_infos[select_id]['scene_token'] != info['scene_token']:
                select_id = index  # out of sequence
            info = self.data_infos[select_id]

            for cam_name in info['cams'].keys():
                intrin = torch.Tensor(info['cams'][cam_name]['cam_intrinsic'])
                sensor2ego, ego2global = get_sensor_transforms(info, cam_name)
                img_file_path = info['cams'][cam_name]['data_path']

                # load seg/depth GT of rays
                seg_map = load_seg_label(img_file_path, self.semantic_gt_path)
                coor, label_depth = load_depth(img_file_path, self.depth_gt_path)
                label_seg = seg_map[coor[:,1], coor[:,0]]

                # load RGB GT of rays
                img = load_img(img_file_path)
                # normalization
                img_tensor = self.normalize_rgb(img)
                img_numpy = img_tensor.permute(1, 2, 0).detach().numpy()
                label_img = img_numpy[coor[:,1], coor[:, 0], ...]

                sensor2egos.append(sensor2ego)
                ego2globals.append(ego2global)
                intrins.append(intrin)
                coors.append(torch.Tensor(coor))
                label_depths.append(torch.Tensor(label_depth))
                label_segs.append(torch.Tensor(label_seg))
                label_imgs.append(torch.Tensor(label_img))
                img_paths.append(img_file_path)
                img_tensors.append(img_tensor)
                time_ids[time_id].append(idx)
                idx += 1
        
        T, N = len(self.aux_frames)+1, len(info['cams'].keys())
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        sensor2egos = sensor2egos.view(T, N, 4, 4)
        ego2globals = ego2globals.view(T, N, 4, 4)

        # calculate the transformation from adjacent_sensor to key_ego
        keyego2global = ego2globals[0, :,  ...].unsqueeze(0)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()
        sensor2keyegos = sensor2keyegos.view(T*N, 4, 4)

        img_tensors = torch.stack(img_tensors)

        # generate rays for all frames
        rays = generate_rays(
            coors, label_depths, label_segs, label_imgs, sensor2keyegos, intrins,
            max_ray_nums=self.max_ray_nums, 
            time_ids=time_ids, 
            dynamic_class=self.dynamic_class, 
            balance_weight=self.WRS_balance_weight)
        return rays
    
    def get_rays_dense(self, index):
        info = self.data_infos[index]

        sensor2egos = []
        ego2globals = []
        intrins = []
        coors = []
        label_imgs = []
        img_paths = []
        img_tensors = []
        time_ids = {}
        idx = 0

        for time_id in [0] + self.aux_frames:
            time_ids[time_id] = []
            select_id = max(index + time_id, 0)
            if select_id>=len(self.data_infos) or self.data_infos[select_id]['scene_token'] != info['scene_token']:
                select_id = index  # out of sequence
            info = self.data_infos[select_id]

            for cam_name in info['cams'].keys():
                intrin = torch.Tensor(info['cams'][cam_name]['cam_intrinsic'])
                sensor2ego, ego2global = get_sensor_transforms(info, cam_name)
                img_file_path = info['cams'][cam_name]['data_path']

                # load RGB GT of rays
                img = load_img(img_file_path)
                # normalization
                img_tensor = self.normalize_rgb(img)
                img_numpy = img_tensor.permute(1, 2, 0).detach().numpy()

                # dense coords
                h, w, c = img_numpy.shape
                coor = generate_dense_coors(w, h).astype(np.int32)
                
                _full = [i for i in range(coor.shape[0])]
                _portion = random.sample(_full, 4000)
                coor = coor[list(_portion)]

                label_img = img_numpy[coor[:, 1], coor[:, 0], ...]

                sensor2egos.append(sensor2ego)
                ego2globals.append(ego2global)
                intrins.append(intrin)
                coors.append(torch.Tensor(coor))
                label_imgs.append(torch.Tensor(label_img))
                img_paths.append(img_file_path)
                img_tensors.append(img_tensor)
                time_ids[time_id].append(idx)
                idx += 1
        
        T, N = len(self.aux_frames)+1, len(info['cams'].keys())
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        sensor2egos = sensor2egos.view(T, N, 4, 4)
        ego2globals = ego2globals.view(T, N, 4, 4)

        # calculate the transformation from adjacent_sensor to key_ego
        keyego2global = ego2globals[0, :,  ...].unsqueeze(0)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()
        sensor2keyegos = sensor2keyegos.view(T*N, 4, 4)

        img_tensors = torch.stack(img_tensors)

        # generate rays for all frames
        rays = generate_rays_dense(
            coors, label_imgs, sensor2keyegos, intrins,
            max_ray_nums=self.max_ray_nums, 
            time_ids=time_ids)
        return rays

    def get_data_info(self, index):
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        input_dict['with_gt'] = self.data_infos[index]['with_gt'] if 'with_gt' in self.data_infos[index] else True
        if 'occ_path' in self.data_infos[index]:
            input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        # generate rays for rendering supervision
        if self.use_rays:
            if self.if_dense:
                rays_info = self.get_rays_dense(index)
            else:
                rays_info = self.get_rays(index)
            input_dict['rays'] = rays_info
        else:
            input_dict['rays'] = torch.zeros((1))
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            # occ_pred = occ_pred
            # print(occ_pred.shape, gt_semantics.shape, mask_camera.shape)
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

        occ_names, IoU, _, IoU_res = self.occ_eval_metrics.count_iou()
        class_names, mIoU, _, mIoU_res = self.occ_eval_metrics.count_miou()
        # return self.occ_eval_metrics.count_iou()
        res = {
            "IoU": IoU_res,
            "mIoU": mIoU_res,
            "classes": len(mIoU) - 1,
        }
        return res
