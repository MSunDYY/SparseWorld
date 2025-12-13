# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmcv
import numba as nb
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES

@PIPELINES.register_module()
class LoadOccGTFromFile4DTraj(object):
    def __init__(self, 
            mask_path='data/nuscenes/mask_camera_count.npy', 
            thresh=0.2,
            ):
        self.mask_path = mask_path
        self.thresh = thresh

    def __call__(self, results):
        if results['with_gt'] and 'occ_gt_path' in results:
            # assert 'occ_gt_path' in results
            occ_gt_path = results['occ_gt_path']
            occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

            occ_labels = np.load(occ_gt_path)
            semantics = occ_labels['semantics']
            mask_lidar = occ_labels['mask_lidar']
            mask_camera = occ_labels['mask_camera']
        else:
            semantics = np.zeros((200,200,16),dtype=np.uint8)
            mask = np.load(self.mask_path) > self.thresh
            mask = mask.astype(np.uint8)
            mask_lidar = mask
            mask_camera = mask

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        intervals = [1, 2, 3, 4, 5, 6]
        results['temporal_semantics'] = dict()
        for interval in intervals:
            occ_gt_path = results['temporal_occ_gt_path'][interval]
            occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

            occ_labels = np.load(occ_gt_path)
            semantics = occ_labels['semantics']
            mask_lidar = occ_labels['mask_lidar']
            mask_camera = occ_labels['mask_camera']

            results['temporal_semantics'][interval] = dict()
            results['temporal_semantics'][interval]['voxel_semantics'] = semantics
            results['temporal_semantics'][interval]['mask_lidar'] = mask_lidar
            results['temporal_semantics'][interval]['mask_camera'] = mask_camera

        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth4DTraj(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes, gt_labels = torch.Tensor(np.array(gt_boxes)), torch.tensor(np.array(gt_labels))
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)
        
        if 'temporal_img_inputs' in results:
            intervals = [1, 2, 3, 4, 5, 6]
            for interval in intervals:
                interval_imgs, interval_rots, interval_trans, interval_intrins = results['temporal_img_inputs'][interval][:4]
                interval_post_rots, interval_post_trans = results['temporal_img_inputs'][interval][4:]
                results['temporal_img_inputs'][interval] = (interval_imgs, interval_rots, interval_trans, interval_intrins, interval_post_rots, interval_post_trans, bda_rot)

        if 'voxel_semantics' in results:
            if flip_dx:
                results['voxel_semantics'] = results['voxel_semantics'][::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][::-1,...].copy()
            if flip_dy:
                results['voxel_semantics'] = results['voxel_semantics'][:,::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][:,::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][:,::-1,...].copy()
        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy

        intervals = [1, 2, 3, 4, 5, 6]
        for interval in intervals:
            if flip_dx:
                results['temporal_semantics'][interval]['voxel_semantics'] = results['temporal_semantics'][interval]['voxel_semantics'][::-1,...].copy()
                results['temporal_semantics'][interval]['mask_lidar'] = results['temporal_semantics'][interval]['mask_lidar'][::-1,...].copy()
                results['temporal_semantics'][interval]['mask_camera'] = results['temporal_semantics'][interval]['mask_camera'][::-1,...].copy()
            if flip_dy:
                results['temporal_semantics'][interval]['voxel_semantics'] = results['temporal_semantics'][interval]['voxel_semantics'][:,::-1,...].copy()
                results['temporal_semantics'][interval]['mask_lidar'] = results['temporal_semantics'][interval]['mask_lidar'][:,::-1,...].copy()
                results['temporal_semantics'][interval]['mask_camera'] = results['temporal_semantics'][interval]['mask_camera'][:,::-1,...].copy()

        return results


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.
    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.
    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ( (depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map), torch.Tensor(cam_depth[valid_mask])


@PIPELINES.register_module()
class PrepareImageInputs4DTraj(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        load_depth=False, 
        depth_gt_path=None,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.load_depth = load_depth
        self.depth_gt_path = depth_gt_path

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
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

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        gt_depths = list()
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # import ipdb;ipdb.set_trace()
            def load_depth(img_file_path, gt_path):
                file_name = os.path.split(img_file_path)[-1]
                point_depth = np.fromfile(os.path.join(gt_path, f'{file_name}.bin'),
                    dtype=np.float32,
                    count=-1).reshape(-1, 3)
                
                point_depth_aug_map, point_depth_aug = depth_transform(
                    point_depth, resize, self.data_config['input_size'],
                    crop, flip, rotate)
                return point_depth_aug_map

            if self.load_depth:
                img_file_path = results['curr']['cams'][cam_name]['data_path']
                gt_depths.append(load_depth(img_file_path, self.depth_gt_path))
                if self.sequential:
                    assert 'adjacent' in results
                    for adj_info in results['adjacent']:
                        filename_adj = adj_info['cams'][cam_name]['data_path']
                        gt_depths.append(load_depth(filename_adj, self.depth_gt_path))
            else:
                gt_depths.append(torch.zeros(1))
            
            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))
    
            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        imgs = torch.stack(imgs)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        gt_depths = torch.stack(gt_depths)
        results['canvas'] = canvas
        # import ipdb;ipdb.set_trace()
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans), gt_depths
    
    def sample_augmentation_temporal(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        resize = float(fW) / float(W)
        if scale is not None:
            resize += scale
        else:
            resize += self.data_config.get('resize_test', 0.0)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False if flip is None else flip
        rotate = 0
        return resize, resize_dims, crop, flip, rotate
    
    def get_inputs_temporal(self, results, flip=None, scale=None):
        temporal_img_inputs = dict()

        intervals = [1, 2, 3, 4, 5, 6]
        for interval in intervals:
            info_dict = results['temporal_ann_infos'][interval]

            imgs = []
            sensor2egos = []
            ego2globals = []
            intrins = []
            post_rots = []
            post_trans = []
            cam_names = self.choose_cams()
            # results['cam_names'] = cam_names
            canvas = []
            for cam_name in cam_names:
                cam_data = info_dict['curr']['cams'][cam_name]
                filename = cam_data['data_path']
                img = Image.open(filename)
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                intrin = torch.Tensor(cam_data['cam_intrinsic'])

                sensor2ego, ego2global = \
                    self.get_sensor_transforms(info_dict['curr'], cam_name)
                # image view augmentation (resize, crop, horizontal flip, rotate)
                img_augs = self.sample_augmentation_temporal(
                    H=img.height, W=img.width, flip=flip, scale=scale)
                resize, resize_dims, crop, flip, rotate = img_augs
                img, post_rot2, post_tran2 = \
                    self.img_transform(img, post_rot,
                                      post_tran,
                                      resize=resize,
                                      resize_dims=resize_dims,
                                      crop=crop,
                                      flip=flip,
                                      rotate=rotate)

                # for convenience, make augmentation matrices 3x3
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2
                
                canvas.append(np.array(img))
                imgs.append(self.normalize_img(img))
        
                if self.sequential:
                    assert 'adjacent' in info_dict
                    for adj_info in info_dict['adjacent']:
                        filename_adj = adj_info['cams'][cam_name]['data_path']
                        img_adjacent = Image.open(filename_adj)
                        img_adjacent = self.img_transform_core(
                            img_adjacent,
                            resize_dims=resize_dims,
                            crop=crop,
                            flip=flip,
                            rotate=rotate)
                        imgs.append(self.normalize_img(img_adjacent))
                intrins.append(intrin)
                sensor2egos.append(sensor2ego)
                ego2globals.append(ego2global)
                post_rots.append(post_rot)
                post_trans.append(post_tran)

            if self.sequential:
                for adj_info in info_dict['adjacent']:
                    post_trans.extend(post_trans[:len(cam_names)])
                    post_rots.extend(post_rots[:len(cam_names)])
                    intrins.extend(intrins[:len(cam_names)])

                    # align
                    for cam_name in cam_names:
                        sensor2ego, ego2global = \
                            self.get_sensor_transforms(adj_info, cam_name)
                        sensor2egos.append(sensor2ego)
                        ego2globals.append(ego2global)

            imgs = torch.stack(imgs)

            sensor2egos = torch.stack(sensor2egos)
            ego2globals = torch.stack(ego2globals)
            intrins = torch.stack(intrins)
            post_rots = torch.stack(post_rots)
            post_trans = torch.stack(post_trans)
        
            temporal_img_inputs[interval] = (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)
        return temporal_img_inputs

    def __call__(self, results):
        img_inputs, gt_depths  = self.get_inputs(results)
        results['img_inputs'] = img_inputs
        results['gt_depths'] = gt_depths
        temporal_inputs = self.get_inputs_temporal(results)
        results['temporal_img_inputs'] = temporal_inputs
        return results