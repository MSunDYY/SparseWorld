# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from torch.utils.checkpoint import checkpoint as cp
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp import autocast
from mmdet3d.models import builder

# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221, 2307405309])

nusc_class_names = [
    "empty", # 0
    "barrier", # 1
    "bicycle", # 2 
    "bus", # 3 
    "car", # 4
    "construction", # 5
    "motorcycle", # 6
    "pedestrian", # 7
    "trafficcone", # 8
    "trailer", # 9
    "truck", # 10
    "driveable_surface", # 11
    "other", # 12
    "sidewalk", # 13
    "terrain", # 14
    "mannade", # 15 
    "vegetation", # 16
]

@HEADS.register_module()
class OccHead(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channel,
        num_level=1,
        soft_weights=False,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        point_cloud_range=[-40., -40., -1., 40., 40., 5.4],
        final_occ_size=[200, 200, 16],
        empty_idx=17,
        balance_cls_weight=True,
        with_cp=False,
        use_deblock=False,
    ):
        super(OccHead, self).__init__()

        self.fp16_enabled=False
      
        if type(in_channels) is not list:
            in_channels = [in_channels]
        self.with_cp = with_cp
        self.use_deblock = use_deblock
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        
        self.point_cloud_range = torch.tensor(np.array(point_cloud_range)).float()

        # voxel-level prediction
        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] // 2
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
                        out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True))
            self.occ_convs.append(occ_conv)

        self.occ_pred_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=mid_channel, 
                        out_channels=mid_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, mid_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=mid_channel//2, 
                        out_channels=out_channel, kernel_size=1, stride=1, padding=0))

        self.soft_weights = soft_weights
        self.num_point_sampling_feat = self.num_level + 1 * self.use_deblock
        if self.soft_weights:
            soft_in_channel = mid_channel
            self.voxel_soft_weights = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=soft_in_channel, 
                        out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, soft_in_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=soft_in_channel//2, 
                        out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))

        if self.use_deblock:
            upsample_cfg=dict(type='deconv3d', bias=False)
            upsample_layer = build_conv_layer(
                    upsample_cfg,
                    in_channels=self.in_channels[0],
                    out_channels=self.in_channels[0]//2,
                    kernel_size=2,
                    stride=2,
                    padding=0)

            self.deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, self.in_channels[0]//2)[1],
                                    nn.ReLU(inplace=True))

        self.class_names = nusc_class_names    
        self.empty_idx = empty_idx
    
    @force_fp32(apply_to=('voxel_feats')) 
    def forward_coarse_voxel(self, voxel_feats):
        output_occs = []
        output = {}

        if self.use_deblock:
            if self.with_cp and voxel_feats[0].requires_grad:
                x0 = cp(self.deblock, voxel_feats[0])
            else:
                x0 = self.deblock(voxel_feats[0])
            output_occs.append(x0)
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            if self.with_cp  and feats.requires_grad:
                x = cp(occ_conv, feats)
            else:
                x = occ_conv(feats)
            output_occs.append(x)

        if self.soft_weights:
            voxel_soft_weights = self.voxel_soft_weights(output_occs[0])
            voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
        else:
            voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat

        out_voxel_feats = 0
        _, _, H, W, D= output_occs[0].shape
        for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
            feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
            out_voxel_feats += feats * weights.unsqueeze(1)
        output['out_voxel_feats'] = [out_voxel_feats]
        if self.with_cp and  out_voxel_feats.requires_grad:
            out_voxel = cp(self.occ_pred_conv, out_voxel_feats)
        else:
            out_voxel = self.occ_pred_conv(out_voxel_feats)

        output['occ'] = [out_voxel]

        return output
     
    @force_fp32()
    def forward(self, voxel_feats, **kwargs):
        
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level
        
        output = self.forward_coarse_voxel(voxel_feats)
        out_voxel_feats = output['out_voxel_feats'][0]
        coarse_occ = output['occ'][0]

        res = {
            'output_voxels': output['occ'],
        }


        return res


class DownScaleModule3DCustom(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.downscale1 = nn.Conv3d(self.in_dim, self.in_dim*2, 2, stride=2)
        self.downscale2 = nn.Conv3d(self.in_dim*2, self.in_dim*4, 2, stride=2)
        self.downscale3 = nn.Conv3d(self.in_dim*4, self.in_dim*4, 2, stride=2)
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def forward(self, feats):
        b_, H, W, Z, dim_ = feats.size()
        feats = feats.permute(0, 4, 1, 2, 3).contiguous()

        downscaled_2x_feats = self.downscale1(feats)
        downscaled_4x_feats = self.downscale2(downscaled_2x_feats)
        downscaled_8x_feats = self.downscale3(downscaled_4x_feats)

        out_feats = self.pooling(downscaled_8x_feats)
        
        out_feats = out_feats.permute(0, 2, 3, 4, 1).contiguous()

        return out_feats