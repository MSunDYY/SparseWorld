from open3d_vis_utils import draw_scenes
import numpy as np
points = np.load('/home/yons/points.npy')
draw_scenes(points)