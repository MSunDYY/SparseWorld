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
from .occ_metrics import Metric_mIoU_Temporal, Metric_FScore
from .ray import generate_rays, generate_rays_dense

from PIL import Image
from torchvision import transforms

import random
import pickle

from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes

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


def count_layers(obj):
    if isinstance(obj, list):
        return 1 + max(count_layers(x) for x in obj)
    else:
        return 0

def train_token():
    with open('./admlp/fengze_nuscenes_infos_train.pkl','rb')as f:
        res=[]
        data=pickle.load(f)['infos']
        for ww in data:
            res.append(ww['token'])
        return res

def test_token():
    with open('./admlp/fengze_nuscenes_infos_val.pkl','rb')as f:
        res=[]
        data=pickle.load(f)['infos']
        for ww in data:
            res.append(ww['token'])
        return res


@DATASETS.register_module()
class NuScenesDatasetOccpancy4DTraj(NuScenesDataset):
    def __init__(self, 
                use_rays=False,
                if_dense=False,
                semantic_gt_path=None,
                depth_gt_path=None,
                ego_gt_path=None,
                is_train=True,
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

        self.tokens = train_token() if not self.test_mode else test_token()
        if ego_gt_path is not None:
            self.ad_info = pickle.load(open(ego_gt_path, 'rb'))
        else:
            self.ad_info = pickle.load(open('admlp/stp3_val/data_nuscene.pkl', 'rb'))
        
        if not self.test_mode:
            self.traj_info = pickle.load(open('occworld/nuscenes_infos_train_temporal_v3_scene.pkl', 'rb'))['infos']
        else:
            self.traj_info = pickle.load(open('occworld/nuscenes_infos_val_temporal_v3_scene.pkl', 'rb'))['infos']

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

        self.use_valid_flag = True
        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR
    
    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        # idx = 0
        nusc_idx = self.temp2nusc_map[idx]
        if self.test_mode:
            return self.prepare_test_data(nusc_idx)
        while True:
            data = self.prepare_train_data(nusc_idx)
            if data is None:
                nusc_idx = self._rand_another(nusc_idx)
                continue
            return data
    
    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.temp2nusc_map)
        # return 1
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        infos = []

        # data_infos = data_infos[:39] #for debug
        self.metadata = data['metadata']
        self.version = self.metadata['version']

        self.temp2nusc_map = []
        for idx, data_info in enumerate(data_infos):
            scene_name = data_info['scene_name']
            scene_path = f'./data/nuscenes/gts/{scene_name}'
            scene_len = len(os.listdir(scene_path))
            frame_idx = data_info['frame_idx']
            if frame_idx + 12 >= scene_len:
                pass
            else:
                self.temp2nusc_map.append(idx + 5)  # fair comparison with OccWorld

        return data_infos

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
        intervals = [1, 2, 3, 4, 5, 6]

        input_dict = super(NuScenesDatasetOccpancy4DTraj, self).get_data_info(index)
        input_dict['with_gt'] = self.data_infos[index]['with_gt'] if 'with_gt' in self.data_infos[index] else True
        if 'occ_path' in self.data_infos[index]:
            input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
            input_dict['temporal_occ_gt_path'] = dict()
            for interval in intervals:
                input_dict['temporal_occ_gt_path'][interval] = self.data_infos[index + interval]['occ_path']
        input_dict['temporal_ann_infos'] = dict()
        for interval in intervals:
            future_ann_infos = super(NuScenesDatasetOccpancy4DTraj, self).get_data_info(index + interval)
            input_dict['temporal_ann_infos'][interval] = future_ann_infos
        
        ego_infos = self.traj_info[input_dict['scene_name']][input_dict['curr']['frame_idx']]
        ego_traj_infos = ego_infos['gt_ego_fut_trajs']
        input_dict['temporal_trajs'] = torch.from_numpy(ego_traj_infos)

        if self.use_valid_flag:
            mask = ego_infos['valid_flag']
        else:
            mask = ego_infos['num_lidar_pts'] > 0
        gt_bboxes_3d = ego_infos['gt_boxes'][mask]
        # gt_names_3d = ego_infos['gt_names'][mask]
        if self.with_velocity:
            gt_velocity = ego_infos['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        if self.with_attr:
            gt_fut_trajs = ego_infos['gt_agent_fut_trajs'][mask]
            gt_fut_masks = ego_infos['gt_agent_fut_masks'][mask]
            gt_fut_goal = ego_infos['gt_agent_fut_goal'][mask]
            gt_lcf_feat = ego_infos['gt_agent_lcf_feat'][mask]
            gt_fut_yaw = ego_infos['gt_agent_fut_yaw'][mask]
            attr_labels = np.concatenate(
                [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
            ).astype(np.float32)
        # gt_bboxes_3d = LiDARInstance3DBoxes(
        #     gt_bboxes_3d,
        #     box_dim=gt_bboxes_3d.shape[-1],
        #     origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        input_dict['temporal_agent_boxes'] = torch.tensor(gt_bboxes_3d)
        input_dict['temporal_agent_feats'] = torch.tensor(attr_labels)

        # generate rays for rendering supervision
        if self.use_rays:
            if self.if_dense:
                rays_info = self.get_rays_dense(index)
            else:
                rays_info = self.get_rays(index)
            input_dict['rays'] = rays_info
            
            input_dict['temporal_rays'] = dict()
            if not self.if_dense:
                for interval in intervals:
                    temp_rays_info = self.get_rays(index + interval)
                    input_dict['temporal_rays'][interval] = temp_rays_info
        else:
            input_dict['rays'] = torch.zeros((1))
            input_dict['temporal_rays'] = dict()

        ego_intervals = [0, 1, 2, 3, 4, 5]
        input_dict['temporal_ego_states'] = dict()
        input_dict['temporal_ego2global'] = dict()
        input_dict['temporal2ego'] = dict()
        curr_scene = self.data_infos[index]['scene_name']
        for ego_interval in ego_intervals:
            info = self.data_infos[index + ego_interval]
            # info = self.data_infos[index]
            token = info['token']
            assert token in self.ad_info.keys()
            # assert token in self.tokens
            assert info['scene_name'] == curr_scene
            assert info['frame_idx'] >= 4

            cur_info = []
            key = list(self.ad_info[token])
            key.sort()
            for k in key:
                if k=='gt':continue
                ele = self.ad_info[token][k]
                if count_layers(ele) == 2:
                    cur_info += ele
                else:
                    cur_info.append(ele)
            cur_info = torch.tensor(cur_info).to(torch.float32).flatten().unsqueeze(-1).permute(1, 0)
            input_dict['temporal_ego_states'][ego_interval] = cur_info
            # viewpad = np.eye(4)
            # viewpad[:3,:3] = Quaternion(input_dict['temporal_ann_infos'][ego_interval+1]['ego2global_rotation']).rotation_matrix
            # viewpad[:3,3] = np.array(input_dict['temporal_ann_infos'][ego_interval+1]['ego2global_translation'])
            input_dict['temporal2ego'][ego_interval] = np.matmul(np.linalg.inv(input_dict['ego2global']),
                                                                 input_dict['temporal_ann_infos'][ego_interval+1]['ego2global']).astype(np.float32)
            input_dict['temporal_ego2global'][ego_interval] = input_dict['temporal_ann_infos'][ego_interval+1]['ego2global']
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU_Temporal(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=False)

        print('\nStarting Evaluation...')
        if len(occ_results)==2:
            occ_results,pred_trajs = occ_results[0],occ_results[1]
        else:
            occ_results = occ_results[0]
            pred_trajs = None

        out_pkl = {}
        for index, occ_pred in enumerate(tqdm(occ_results)):
            gt_semantics_temp = {}
            mask_lidar_temp = {}
            mask_camera_temp = {}

            interval_list = [2, 4, 6]

            # index = 0
            nusc_index = self.temp2nusc_map[index]
            curr_info = self.data_infos[nusc_index]
            try:
                curr_occ_gt = np.load(os.path.join(curr_info['occ_path'],'labels.npz'))
                curr_gt_semantics = curr_occ_gt['semantics']
                curr_mask_lidar = curr_occ_gt['mask_lidar'].astype(bool)
                curr_mask_camera = curr_occ_gt['mask_camera'].astype(bool)
            except:
                time.sleep(1)
                curr_occ_gt = np.load(os.path.join(curr_info['occ_path'], 'labels.npz'))
                curr_gt_semantics = curr_occ_gt['semantics']
                curr_mask_lidar = curr_occ_gt['mask_lidar'].astype(bool)
                curr_mask_camera = curr_occ_gt['mask_camera'].astype(bool)

            if pred_trajs is not None:
                out_pkl[curr_info['token']] = np.cumsum(pred_trajs[index].cpu().numpy(),axis=1)
            gt_semantics_temp[0] = curr_gt_semantics
            mask_lidar_temp[0] = curr_mask_lidar
            mask_camera_temp[0] = curr_mask_camera

            for interval in interval_list:

                info = self.data_infos[nusc_index + interval]
                assert info['scene_name'] == curr_info['scene_name']

                try:
                    occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
                    gt_semantics = occ_gt['semantics']
                    mask_lidar = occ_gt['mask_lidar'].astype(bool)
                    mask_camera = occ_gt['mask_camera'].astype(bool)

                    gt_semantics_temp[interval] = gt_semantics
                    mask_lidar_temp[interval] = mask_lidar
                    mask_camera_temp[interval] = mask_camera
                except:
                    time.sleep(1)
                    occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                    gt_semantics = occ_gt['semantics']
                    mask_lidar = occ_gt['mask_lidar'].astype(bool)
                    mask_camera = occ_gt['mask_camera'].astype(bool)

                    gt_semantics_temp[interval] = gt_semantics
                    mask_lidar_temp[interval] = mask_lidar
                    mask_camera_temp[interval] = mask_camera
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics_temp, mask_lidar_temp, mask_camera_temp)

        if pred_trajs is not None:
            with open('admlp/output_data.pkl', 'wb') as f:
                pickle.dump(out_pkl, file=f)
        iou_res_list = self.occ_eval_metrics.count_iou()
        mIoU_1s, miou_res_list = self.occ_eval_metrics.count_miou()

        res = {
            "IoU": iou_res_list,
            "mIoU": miou_res_list,
            "classes": len(mIoU_1s) - 1,
        }
        return res