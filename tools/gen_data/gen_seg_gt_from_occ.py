
import os
import cv2
import copy
import enum
import mmcv
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
np.random.seed(0)

from PIL import Image

colors = np.array(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],  # 1 barrier              orangey
        [255, 192, 203, 255],  # 2 bicycle              pink
        [255, 255, 0, 255],  # 3 bus                  yellow
        [0, 150, 245, 255],  # 4 car                  blue
        [0, 255, 255, 255],  # 5 construction_vehicle cyan
        [200, 180, 0, 255],  # 6 motorcycle           dark orange
        [255, 0, 0, 255],  # 7 pedestrian           red
        [255, 240, 150, 255],  # 8 traffic_cone         light yellow
        [135, 60, 0, 255],  # 9 trailer              brown
        [160, 32, 240, 255],  # 10 truck                purple
        [255, 0, 255, 255],  # 11 driveable_surface    dark pink
        # [175,   0,  75, 255],       # 12 other_flat           dark red
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # 13 sidewalk             dard purple
        [150, 240, 80, 255],  # 14 terrain              light green
        [230, 230, 250, 255],  # 15 manmade              white
        [0, 175, 0, 255],  # 16 vegetation           green
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)

# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    im,
    lidar2ego_translation,
    lidar2ego_rotation,
    ego2global_translation,
    ego2global_rotation,
    sensor2ego_translation, 
    sensor2ego_rotation,
    cam_ego2global_translation,
    cam_ego2global_rotation,
    cam_intrinsic,
    min_dist: float = 0.0,
    ):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    pc = LidarPointCloud(pc.T)
    # pc.rotate(Quaternion(lidar2ego_rotation).rotation_matrix)
    # pc.translate(np.array(lidar2ego_translation))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(ego2global_rotation).rotation_matrix)
    pc.translate(np.array(ego2global_translation))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego2global_translation))
    pc.rotate(Quaternion(cam_ego2global_rotation).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(sensor2ego_translation))
    pc.rotate(Quaternion(sensor2ego_rotation).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    labels = pc.points[3, :]
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         cam_intrinsic,
                         normalize=True)
    
    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    # mask = np.logical_and(mask, labels != 17)
    points = points[:, mask]
    coloring = coloring[mask]
    labels = labels[mask].astype(np.int32)
    return points, coloring, labels


save_folder = os.path.join('./data/', 'seg_gt_occ') 
info_path_train = './data/nuscenes/bevdetv2-nuscenes_infos_train.pkl'
info_path_val = './data/nuscenes/bevdetv2-nuscenes_infos_val.pkl'


visual=False
lidar_key = 'LIDAR_TOP'
cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT'
]



colors = np.random.randint(0, 255, size=(40, 3))
def get_voxel_coords(arr):
    x, y, z = arr.shape
    coords = np.indices((x, y, z)).transpose(1, 2, 3, 0)
    return coords

def draw_points(img, pts_img, label):
    for i in range(pts_img.shape[1]):
        x, y = pts_img[:, i]
        color = colors[label[i]]
        cv2.circle(img, (x, y), 3, color.tolist(), -1)
    return img

pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

from nuscenes import NuScenes
# nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

label_name = {
    0: 'noise', 
    1: 'animal', 
    2: 'human.pedestrian.adult', 
    3: 'human.pedestrian.child', 
    4: 'human.pedestrian.construction_worker', 
    5: 'human.pedestrian.personal_mobility', 
    6: 'human.pedestrian.police_officer', 
    7: 'human.pedestrian.stroller', 
    8: 'human.pedestrian.wheelchair', 
    9: 'movable_object.barrier', 
    10: 'movable_object.debris', 
    11: 'movable_object.pushable_pullable', 
    12: 'movable_object.trafficcone', 
    13: 'static_object.bicycle_rack', 
    14: 'vehicle.bicycle', 
    15: 'vehicle.bus.bendy', 
    16: 'vehicle.bus.rigid', 
    17: 'vehicle.car', 
    18: 'vehicle.construction', 
    19: 'vehicle.emergency.ambulance', 
    20: 'vehicle.emergency.police', 
    21: 'vehicle.motorcycle', 
    22: 'vehicle.trailer', 
    23: 'vehicle.truck', 
    24: 'flat.driveable_surface', 
    25: 'flat.other', 
    26: 'flat.sidewalk', 
    27: 'flat.terrain', 
    28: 'static.manmade', 
    29: 'static.other', 
    30: 'static.vegetation', 
    31: 'vehicle.ego'}

label_map = {
    'animal':0, 
    'human.pedestrian.personal_mobility':0, 
    'human.pedestrian.stroller':0, 
    'human.pedestrian.wheelchair':0, 
    'movable_object.debris':0,
    'movable_object.pushable_pullable':0, 
    'static_object.bicycle_rack':0, 
    'vehicle.emergency.ambulance':0, 
    'vehicle.emergency.police':0, 
    'noise':0, 
    'static.other':0, 
    'vehicle.ego':0,
    'movable_object.barrier':1, 
    'vehicle.bicycle':2,
    'vehicle.bus.bendy':3,
    'vehicle.bus.rigid':3,
    'vehicle.car':4,
    'vehicle.construction':5,
    'vehicle.motorcycle':6,
    'human.pedestrian.adult':7,
    'human.pedestrian.child':7,
    'human.pedestrian.construction_worker':7,
    'human.pedestrian.police_officer':7,
    'movable_object.trafficcone':8,
    'vehicle.trailer':9,
    'vehicle.truck':10,
    'flat.driveable_surface':11,
    'flat.other':12,
    'flat.sidewalk':13,
    'flat.terrain':14,
    'static.manmade':15,
    'static.vegetation':16
}

label_merged_map = {}
for idx in label_name:
    name = label_name[idx]
    idx_merged = label_map[name]
    label_merged_map[idx] = idx_merged

label_merged_map = {0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0, 12: 8, 13: 0, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9, 23: 10, 24: 11, 25: 12, 26:
13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0}

def voxel2pts(gt_volume):
    if len(gt_volume.shape) == 4:
        b, h, w, z = gt_volume.shape
        gt_volume = gt_volume.reshape(h, w, z)
    voxel_origin = np.array([-40., -40., -1.])
    voxel_center = np.array([0.2, 0.2, 0.2])
    voxel_size = np.array([0.4, 0.4, 0.4])

    scene_size = (80., 80., 6.4)
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = voxel_origin
    vol_bnds[:, 1] = voxel_origin + np.array(scene_size)

    vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order='C').astype('int')
    xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
    ref_3d = np.concatenate([(xv.reshape(1, -1)+0.5)*voxel_size[0], (yv.reshape(1, -1)+0.5)*voxel_size[1], (zv.reshape(1, -1)+0.5)*voxel_size[2]], axis=0).astype(np.float64).T

    pts = ref_3d + voxel_origin
    gt_volume = gt_volume.reshape(-1, 1)
    pts = np.hstack((pts, gt_volume))

    return pts

def vis_pts(gt):
    # print(gt.shape)
    cvted_gt = np.zeros((gt.shape[0], 5), dtype=np.int32)
    for idx in range(gt.shape[0]):
        cvted_gt[idx][:2] = gt[idx][:2]
        cvted_gt[idx][2:] = colors[int(gt[idx][2])][:3]
    return cvted_gt

def vis_image(img, gt, cam_key):
    img[gt[:, 1], gt[:, 0]] = gt[:, 2:]
    tmp = Image.fromarray(img.astype('uint8'))
    tmp.save(f'./vis-{cam_key}.jpg')

def worker(info):
    # print(info.keys())
    # print(info['occ_path'])
    occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
    gt_semantics = occ_gt['semantics']
    # print(gt_semantics.shape)
    # np.save('./tmp.npy', gt_semantics)

    gt_occ = voxel2pts(gt_semantics)
    gt_mask = gt_semantics != 17
    gt_mask = gt_mask.reshape(-1)
    gt_pts = gt_occ[gt_mask]
    # np.save('./gt.npy', gt_pts)

    lidar2ego_translation = info['lidar2ego_translation']
    lidar2ego_rotation = info['lidar2ego_rotation']
    ego2global_translation = info['ego2global_translation']
    ego2global_rotation = info['ego2global_rotation']
    for i, cam_key in enumerate(cam_keys):
        print(cam_key)
        file_name = os.path.split(info['cams'][cam_key]['data_path'])[-1]
        sensor2ego_translation = info['cams'][cam_key]['sensor2ego_translation']
        sensor2ego_rotation = info['cams'][cam_key]['sensor2ego_rotation']
        cam_ego2global_translation = info['cams'][cam_key]['ego2global_translation']
        cam_ego2global_rotation = info['cams'][cam_key]['ego2global_rotation']
        cam_intrinsic = info['cams'][cam_key]['cam_intrinsic']
        img = mmcv.imread(
            os.path.join(info['cams'][cam_key]['data_path']))
        # from PIL import Image
        # x = Image.fromarray(mmcv.bgr2rgb(img).astype('uint8'))
        # x.save('./tmp2-cam.jpg')
        pts_img, depth, label = map_pointcloud_to_image(
            gt_pts.copy(), img, 
            copy.deepcopy(lidar2ego_translation), 
            copy.deepcopy(lidar2ego_rotation), 
            copy.deepcopy(ego2global_translation),
            copy.deepcopy(ego2global_rotation),
            copy.deepcopy(sensor2ego_translation), 
            copy.deepcopy(sensor2ego_rotation), 
            copy.deepcopy(cam_ego2global_translation), 
            copy.deepcopy(cam_ego2global_rotation),
            copy.deepcopy(cam_intrinsic))
        
        projected_pts = np.concatenate([pts_img[:2, :].T, label[:,None]],
                       axis=1).astype(np.float32)
        # print(np.unique(projected_pts[:, 2]))
        # projected_pts.flatten().tofile(os.path.join(save_folder, f'{file_name}.bin'))
        # print(projected_pts.shape)
        rgb_pts = vis_pts(projected_pts).astype(np.int32)
        # print(rgb_pts.shape)
        # print(rgb_pts)
        vis_image(mmcv.bgr2rgb(img).astype('uint8'), rgb_pts, cam_key)

        # break
        

if __name__ == '__main__':
    print('Save to %s'%save_folder)
    mmcv.mkdir_or_exist(save_folder)

    infos = mmcv.load(info_path_train)['infos']
    for info in infos:
        worker(info)
        break
    # print(info_path_train, len(infos))
    # with Pool(8) as p:
    #     p.map(worker, infos)

    # infos = mmcv.load(info_path_val)['infos']
    # print(info_path_val, len(infos))
    # with Pool(8) as p:
    #     p.map(worker, infos)


        
    
