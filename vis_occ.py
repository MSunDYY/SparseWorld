import os

import mmcv
import open3d as o3d
import numpy as np
import torch
import pickle
import math
from typing import Tuple, List, Dict, Iterable
import argparse
import cv2

NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1
FREE_LABEL = 17
BINARY_OBSERVED = 1
BINARY_NOT_OBSERVED = 0

VOXEL_SIZE = [0.4, 0.4, 0.4]
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
SPTIAL_SHAPE = [200, 200, 16]
TGT_VOXEL_SIZE = [0.4, 0.4, 0.4]
TGT_POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]

name = ['other','barrier','bicycle','bus','car','cons','motorcycle','pedestrian','traffic cone','trailer','truck',
        'driver surface','other flat','sidewalk','terrain','manmade','vegetation']

colormap_to_list = [
        [0,   0,   0, 255],  # 0 undefined
        [112, 128, 144, 255],  # 1 barrier  orange
        [220, 20, 60, 255],    # 2 bicycle  Blue
        [255, 127, 80, 255],   # 3 bus  Darkslategrey
        [255, 158, 0, 255],  # 4 car  Crimson
        [233, 150, 70, 255],   # 5 cons. Veh  Orangered
        [255, 61, 99, 255],  # 6 motorcycle  Darkorange
        [0, 0, 230, 255], # 7 pedestrian  Darksalmon
        [47, 79, 79, 255],  # 8 traffic cone  Red
        [255, 140, 0, 255],# 9 trailer  Slategrey
        [255, 99, 71, 255],# 10 truck Burlywood
        [0, 207, 191, 255],    # 11 drive sur  Green
        [175, 0, 75, 255],  # 12 other lat  nuTonomy green
        [75, 0, 75, 255],  # 13 sidewalk
        [112, 180, 60, 255],    # 14 terrain
        [222, 184, 135, 255],    # 15 manmade
        [0, 175, 0, 255],   # 16 vegeyation
]

colormap_to_colors = np.array(colormap_to_list,dtype=np.float32)
colormap_to_gray_colors = np.ones(17*3,dtype=np.float32).reshape(17,3)*180
colormap_to_gray_colors[2:11] *= 0.5
colormap_to_gray_colors = np.concatenate([colormap_to_gray_colors,np.array([[20,20,200]],dtype=np.float32)],axis=0)

def voxel2points(voxel, occ_show, voxelSize):
    """
    Args:
        voxel: (Dx, Dy, Dz)
        occ_show: (Dx, Dy, Dz)
        voxelSize: (dx, dy, dz)

    Returns:
        points: (N, 3) 3: (x, y, z)
        voxel: (N, ) cls_id
        occIdx: (x_idx, y_idx, z_idx)
    """
    occIdx = torch.where(occ_show)
    points = torch.cat((occIdx[0][:, None] * voxelSize[0] + POINT_CLOUD_RANGE[0], \
                        occIdx[1][:, None] * voxelSize[1] + POINT_CLOUD_RANGE[1], \
                        occIdx[2][:, None] * voxelSize[2] + POINT_CLOUD_RANGE[2]),
                       dim=1)      # (N, 3) 3: (x, y, z)
    return points, voxel[occIdx], occIdx

def rotz_angle(points):
    rotation_matrix = np.array([
        [0, 1, 0],  # x' = y
        [-1, 0, 0],  # y' = -x
        [0, 0, 1]  # z' = z (z坐标不变)
    ])
    points_z = points[:,2:3] + 2
    points_angle = points[:,6:7] + np.pi/2
    # 对每个点进行旋转
    rotated_points = np.dot(points[:, :2], rotation_matrix[:2, :2].T)

    # 将旋转后的结果和z坐标合并
    rotated_points = np.concatenate([rotated_points, points_z,points[:,3:6],points_angle,points[:,7:]],axis=-1)

    return rotated_points

def voxel_profile(voxel, voxel_size):
    """
    Args:
        voxel: (N, 3)  3:(x, y, z)
        voxel_size: (vx, vy, vz)

    Returns:
        box: (N, 7) (x, y, z - dz/2, vx, vy, vz, 0)
    """
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)     # (x, y, z - dz/2)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)


def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def my_compute_box_3d(center, size, heading_angle):
    """
    Args:
        center: (N, 3)  3: (x, y, z - dz/2)
        size: (N, 3)    3: (vx, vy, vz)
        heading_angle: (N, 1)
    Returns:
        corners_3d: (N, 8, 3)
    """
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d


def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, bbox3d=None, voxelize=False,
                     bbox_corners=None, linesets=None, vis=None, offset=[0,0,0], large_voxel=True, voxel_size=0.4):
    """
    :param points: (N, 3)  3:(x, y, z)
    :param colors: false 不显示点云颜色
    :param points_colors: (N, 4）
    :param bbox3d: voxel grid (N, 7) 7: (center, wlh, yaw=0)
    :param voxelize: false 不显示voxel边界
    :param bbox_corners: (N, 8, 3)  voxel grid 角点坐标, 用于绘制voxel grid 边界.
    :param linesets: 用于绘制voxel grid 边界.
    :return:
    """
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
    if isinstance(offset, list) or isinstance(offset, tuple):
        offset = np.array(offset)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points+offset)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0])

    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    if large_voxel:
        vis.add_geometry(voxelGrid)
    else:
        vis.add_geometry(pcd)
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # vis.add_geometry(axis)

    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3))+offset)
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        vis.add_geometry(line_sets)

    vis.add_geometry(mesh_frame)

    # ego_pcd = o3d.geometry.PointCloud()
    # ego_points = generate_the_ego_car()
    # ego_pcd.points = o3d.utility.Vector3dVector(ego_points)
    # vis.add_geometry(ego_pcd)

    return vis


def show_occ(occ_state, occ_show, voxel_size, vis=None, offset=[0, 0, 0],gt_boxes = None, gt_labels = None,voxelize=True,show_occ = False):
    """
    Args:
        occ_state: (Dx, Dy, Dz), cls_id
        occ_show: (Dx, Dy, Dz), bool
        voxel_size: [0.4, 0.4, 0.4]
        vis: Visualizer
        offset: checkpoint

    Returns:

    """
    if isinstance(occ_state,np.ndarray):
        occ_state = torch.tensor(occ_state)
    if isinstance(occ_show,np.ndarray):
        occ_show = torch.tensor(occ_show)
    if show_occ:
        mask = torch.rand(8,4)>0.9
        occ_state[96:104,98:102,2:5][mask] = 18
        colors = colormap_to_gray_colors / 255
        occ_show[96:104,98:102,2:5] = True
    else:
        colors = colormap_to_colors / 255

    pcd, labels, occIdx = voxel2points(occ_state, occ_show, voxel_size)
    # pcd: (N, 3)  3: (x, y, z)
    # labels: (N, )  cls_id
    _labels = labels % len(colors)
    pcds_colors = colors[_labels]   # (N, 4)

    bboxes = voxel_profile(pcd, voxel_size)    # (N, 7)   7: (x, y, z - dz/2, dx, dy, dz, 0)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])      # (N, 8, 3)

    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)     # (N, 12, 2)
    # (N, 12, 2) + (N, 1, 1) --> (N, 12, 2)   此时edges中记录的是bboxes_corners的整体id: (0, N*8).
    edges = edges + bases_[:, None, None]

    vis = show_point_cloud(
        points=pcd.numpy(),
        colors=True,
        points_colors=pcds_colors,
        voxelize=voxelize,
        bbox3d=bboxes.numpy(),
        bbox_corners=bboxes_corners.numpy(),
        linesets=edges.numpy(),
        vis=vis,
        offset=offset,
        large_voxel=True,
        voxel_size=0.4
    )
    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0.5, 1), gt_labels)

    vis.run()
    vis.destroy_window()



def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size=[0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    return ego_point_xyz


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    colors = colormap_to_colors/255
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if score is not None:
            color = np.array([0, score[i], 0], dtype=object)

        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(colors[ref_labels[i],:3])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        '--res',default='work_dirs/flashocc-r50/results/', help='Path to the predicted result')
    parser.add_argument(
        '--canva-size', type=int, default=1000, help='Size of canva in pixel')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=6000,
        help='Number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=4,
        help='Trade-off between image-view and bev in size of '
        'the visualized canvas')
    parser.add_argument(
        '--version',
        type=str,
        default='val',
        help='Version of nuScenes dataset')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument(
        '--root_path',
        type=str,
        default='../data/nuscenes',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./vis',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='image',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=10, help='Frame rate of video')
    parser.add_argument(
        '--video-prefix', type=str, default='vis', help='name of video')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # load predicted results
    results_dir = args.res

    # load dataset information
    info_path = \
        args.root_path + '/bevdetv2-nuscenes_infos_%s.pkl' % args.version
    dataset = pickle.load(open(info_path, 'rb'))
    # prepare save path and medium
    vis_dir = args.save_path
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    print('saving visualized result to %s' % vis_dir)
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    if args.format == 'video':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vout = cv2.VideoWriter(
            os.path.join(vis_dir, '%s.mp4' % args.video_prefix), fourcc,
            args.fps, (int(1600 / scale_factor * 3),
                       int(900 / scale_factor * 2 + canva_size)))

    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    print('start visualizing results')

    for cnt, info in enumerate(dataset['infos'][4:min(args.vis_frames, len(dataset['infos'])):10]):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, min(args.vis_frames, len(dataset['infos']))))

        # 创建一个可视化对象

        vis = o3d.visualization.VisualizerWithKeyCallback()

        # 添加坐标系到可视化窗口
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        # 处理场景和数据加载
        scene_name = info['scene_name']
        sample_token = info['token']
        pred_occ_path = os.path.join(results_dir, scene_name, sample_token, 'pred.npz')
        gt_occ_path = info['occ_path']
        pred_occ = np.load(pred_occ_path)['pred']
        gt_data = np.load(os.path.join(gt_occ_path, 'labels.npz'))
        voxel_label = gt_data['semantics']
        lidar_mask = gt_data['mask_lidar']
        camera_mask = gt_data['mask_camera']
        num = {i:(voxel_label[camera_mask!=0]==i).sum() for i in np.unique(voxel_label[camera_mask!=0])[:-1]}
        num = sorted(num.items(), key=lambda x: x[1])

        title = [f'{name[key]}:{value}' for key,value in num]

        vis.create_window(window_name='  '.join(title))
        # 加载图像视图
        imgs = [cv2.imread(info['cams'][view]['data_path']) for view in views]

        # 处理体素展示
        voxel_show = np.logical_and(pred_occ != FREE_LABEL, camera_mask)
        voxel_size = VOXEL_SIZE
        vis = show_occ(torch.from_numpy(pred_occ), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                       offset=[0, pred_occ.shape[0] * voxel_size[0] * 1.2 * 0.5, 0])

        if not args.draw_gt:
            voxel_show = np.logical_and(voxel_label != FREE_LABEL, camera_mask)
            vis = show_occ(torch.from_numpy(voxel_label), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                           offset=[0, voxel_label.shape[0] * voxel_size[0] * 1.2 * -0.5, 0])

        # 设置视角和渲染选项
        view_control = vis.get_view_control()
        view_control.set_lookat([-0.185, 0.513, 3.485])
        view_control.set_front([-0.974, -0.055, 0.221])
        view_control.set_up([0.221, 0.014, 0.975])
        view_control.set_zoom(0.08)

        # opt = vis.get_render_option()
        # opt.background_color = np.array([1, 1, 1])
        # opt.line_width = 5

        vis.poll_events()
        vis.update_renderer()

        # 显示并捕获屏幕
        try:
            vis.run()  # 手动退出后

            # occ_canvas = np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255
            # occ_canvas = occ_canvas.astype(np.uint8)[..., [2, 1, 0]]  # 转换为 BGR 格式
            # occ_canvas_resize = cv2.resize(occ_canvas, (canva_size, canva_size), interpolation=cv2.INTER_CUBIC)
            #
            # # 清理几何体并准备组合图像
            # vis.clear_geometries()
            # big_img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3), dtype=np.uint8)
            # big_img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
            # big_img[900 + canva_size * scale_factor:, :, :] = np.concatenate(
            #     [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]], axis=1)
            # big_img = cv2.resize(big_img, (int(1600 / scale_factor * 3), int(900 / scale_factor * 2 + canva_size)))
            #
            # # 合并 occ_canvas 和 big_img
            # w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
            # big_img[int(900 / scale_factor):int(900 / scale_factor) + canva_size, w_begin:w_begin + canva_size,
            # :] = occ_canvas_resize
            # # vis.destroy_window()
            # # 保存结果
            # if args.format == 'image':
            #     out_dir = os.path.join(vis_dir, f'{scene_name}', f'{sample_token}')
            #     mmcv.mkdir_or_exist(out_dir)
            #     cv2.imwrite(os.path.join(out_dir, 'occ.png'), occ_canvas)
            #     cv2.imwrite(os.path.join(out_dir, 'overall.png'), big_img)
            # elif args.format == 'video':
            #     cv2.putText(big_img, f'{cnt:{cnt}}', (5, 15), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
            #                 fontScale=0.5)
            #     cv2.putText(big_img, f'{scene_name}', (5, 35), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
            #                 fontScale=0.5)
            #     cv2.putText(big_img, f'{sample_token[:5]}', (5, 55), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
            #                 fontScale=0.5)
            #     vout.write(big_img)
        except:
            vis.destroy_window()
        # 销毁窗口


    if args.format == 'video':
        vout.release()



if __name__ == '__main__':
    main()