"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
import time

box_colormap = [
    [0, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
    [0,0,0.5],
    [0,0.5,0.5],
    [0.5,0.5,0]

]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num + 1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes_frames(frames, file_names, draw_origin=True, auto=True, color=None, frame_rate=100):
    def create_vis():
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.zeros(3)
        opt = vis.get_render_option()
        # 设置背景色（这里为白色）
        opt.background_color = np.array([0, 0, 0])

        return vis

    vis = create_vis()

    pts = open3d.geometry.PointCloud()

    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)
    if auto:
        for frame in frames:
            pts.points = open3d.utility.Vector3dVector(frame[:, :3])
            if color == None:
                pts.colors = open3d.utility.Vector3dVector(np.ones((frame.shape[0], 3)))

            vis.add_geometry(pts)
            vis.update_renderer()
            vis.poll_events()
            time.sleep(1 / frame_rate)
            vis.remove_geometry(pts)

    else:

        for frame in frames:
            pts.points = open3d.utility.Vector3dVector(frame[:, :3])
            if (color == None):
                pts.colors = open3d.utility.Vector3dVector(np.ones((frame.shape[0], 3)))
            vis.add_geometry(pts)
            vis.update_renderer()
            vis.poll_events()
            vis.run()
            vis.destroy_window()
            del vis
            vis = create_vis()
            vis.add_geometry(axis_pcd)

        vis.destroy_window()


def draw_pcd(path):
    pcd = open3d.io.read_point_cloud(path)

    # 设置点云颜色 只能是0 1 如[1,0,0]代表红色为既r
    pcd.paint_uniform_color([1, 1, 1])
    # 创建窗口对象
    vis = open3d.visualization.Visualizer()
    # 创建窗口，设置窗口标题
    vis.create_window(window_name="point_cloud")
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    # 设置点云渲染参数
    opt = vis.get_render_option()
    # 设置背景色（这里为白色）
    opt.background_color = np.array([0, 0, 0])
    # 设置渲染点的大小
    opt.point_size = 1.0
    ctrl = vis.get_view_control()
    ctrl.set_lookat(np.array([1, 0, 0], dtype=np.float64))
    ctrl.set_zoom(5)

    vis.add_geometry(pcd)
    # 添加点云
    vis.run()
    vis.destroy_window()


def draw_scenes_pcd(data_path):
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # 创建一个点云数据（示例数据）
    pcd = open3d.io.read_point_cloud(data_path)
    pointcloud = open3d.geometry.PointCloud()

    # 添加点云数据到可视化对象
    vis.add_geometry(pointcloud)
    pcd = np.asarray(pcd.points)
    pointcloud.points = open3d.utility.Vector3dVector(pcd)
    # 创建一个窗口并启动可视化循环
    vis.update_geometry(pointcloud)
    # 获取视图控制对象
    view_control = vis.get_view_control()
    view_control.change_field_of_view(step=0.2)
    # 设置视点（前方 x 轴）、目标（原点）和上方向（上面 z 轴）
    # view_control.set_lookat([1, 0, 0], [0, 0, 0], [0, 0, 1])
    view_control.set_front(np.array([-1, 0, 0], dtype=np.float64))
    # 启动可视化循环

    pointcloud.colors = open3d.utility.Vector3dVector(np.ones((pcd.shape[0], 3)))
    vis.run()
    # 关闭窗口并清理资源
    vis.destroy_window()


def draw_scenes(points, file_name=None, gt_boxes=None, gt_labels=None, ref_boxes=None, ref_labels=None, ref_scores=None,
                point_colors=None,draw_origin=True,window_name='open3d',background_color = np.zeros(3)):
    print('points num:',points.shape[0])
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    if isinstance(gt_labels,torch.Tensor):
        gt_labels=gt_labels.to('cpu').numpy()
    if isinstance(point_colors,torch.Tensor):
        point_colors = point_colors.to('cpu').numpy()
    if isinstance(ref_scores,torch.Tensor):
        ref_scores = ref_scores.to('cpu').numpy()
    if file_name is not None:
        print('file_name:', file_name, 'The num of points is:', points.shape[0])

    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = background_color

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()

    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    view_control = vis.get_view_control()
    view_control.set_lookat(np.array([0, 0, 0]))
    view_control.set_up((0, 1, 1))
    view_control.set_front((0, 0, 0))
    view_control.rotate(0, 0)
    view_control.change_field_of_view(step=20)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0.5, 1), gt_labels)

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


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
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if score is not None:
            color = np.array([0,score[i],0],dtype=object)
        
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
