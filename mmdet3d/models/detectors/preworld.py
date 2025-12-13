# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet_occ import BEVStereo4DOCC
import torch.nn.functional as F
import torch
import cv2
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from .. import builder
import time
from .preworld_temporal_traj import Scatter
from .loss import CE_ssc_loss, sem_scal_loss, geo_scal_loss, l1_loss
from .lovasz_softmax import lovasz_softmax
from IPython import embed

# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221, 2307405309])

@DETECTORS.register_module()
class PreWorld(BEVStereo4DOCC):
    def __init__(self,
                 out_dim=32,
                 dataset_type='Nuscenes',
                 num_classes=18,
                 dense_nerf_head=None,
                 nerf_head=None,
                 occupancy_head=None,
                 test_threshold=8.5,
                 use_lss_depth_loss=True,
                 use_3d_loss=True,
                 if_pretrain=False,
                 if_render=True,
                 if_post_finetune=False,
                 weight_voxel_ce=0.0,
                 weight_voxel_sem_scal=0.0,
                 weight_voxel_geo_scal=0.0,
                 weight_voxel_lovasz=0.0,
                 empty_idx=17,
                 use_focal_loss=True,
                 balance_cls_weight=True,
                 final_softplus=True,
                 **kwargs):
        super(PreWorld, self).__init__(use_predicter=False, **kwargs)
        self.dataset_type = dataset_type
        self.out_dim = out_dim
        self.use_3d_loss = use_3d_loss
        self.test_threshold = test_threshold
        self.use_lss_depth_loss = use_lss_depth_loss
        self.balance_cls_weight = balance_cls_weight
        self.final_softplus = final_softplus
        self.if_pretrain = if_pretrain
        self.if_render = if_render
        self.if_post_finetune = if_post_finetune
        self.empty_idx = empty_idx

        if self.balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            if self.dataset_type == 'NuPlan':
                self.class_weights = torch.from_numpy(1 / np.log(nuplan_class_frequencies[:17] + 0.001)).float()
                self.class_weights[1:4] = 0.
                self.class_weights[11:] = 0.
            self.semantic_loss = nn.CrossEntropyLoss(
                    weight=self.class_weights, reduction="mean"
                )
        else:
            self.semantic_loss = nn.CrossEntropyLoss(reduction="mean")

        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        self.out_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))

        if self.final_softplus:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2),
                nn.Softplus(),
            )
        else:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2),
            )

        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2, num_classes-1),
        )

        self.color_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2, 3),
        )
        
        self.nerf_head = builder.build_head(nerf_head)

        self.occupancy_head = builder.build_head(occupancy_head)
        self.weight_voxel_ce = weight_voxel_ce
        self.weight_voxel_sem_scal = weight_voxel_sem_scal
        self.weight_voxel_geo_scal = weight_voxel_geo_scal
        self.weight_voxel_lovasz = weight_voxel_lovasz

        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss'))
    

    def loss_sup(self, pred, tag, cls_num):
        sup_pred = torch.ones_like(pred).to(pred.device)
        loss_sup = nn.CrossEntropyLoss(reduction="mean")(pred.reshape(-1, cls_num), sup_pred.reshape(-1, cls_num))

        loss_ = dict()
        loss_[f'loss_sup_{tag}'] = loss_sup * 0.

        return loss_
    

    def loss_sup_occupancy(self, output_voxels, target_voxels):
        loss_dict = dict()
        cls_weights = torch.cat([self.class_weights, torch.tensor([0])])
        loss_dict['loss_sup_voxel'] = CE_ssc_loss(output_voxels, target_voxels, cls_weights.type_as(output_voxels), 255) * 0.

        return loss_dict

    
    def loss_voxel(self, output_voxels, target_voxels, camera_mask=None):
        output_voxels[torch.isnan(output_voxels)] = 0
        output_voxels[torch.isinf(output_voxels)] = 0
        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        if self.use_focal_loss:
            cls_weights = torch.cat([self.class_weights, torch.tensor([0])])
            loss_dict['loss_voxel_ce'] = self.weight_voxel_ce * self.focal_loss(output_voxels, target_voxels, cls_weights.type_as(output_voxels), 255, camera_mask=camera_mask)
        else:
            cls_weights = torch.cat([self.class_weights, torch.tensor([0])])
            loss_dict['loss_voxel_ce'] = self.weight_voxel_ce * CE_ssc_loss(output_voxels, target_voxels, cls_weights.type_as(output_voxels), 255)
        
        loss_dict['loss_voxel_sem'] = self.weight_voxel_sem_scal * sem_scal_loss(output_voxels, target_voxels, 255, camera_mask=camera_mask)
        loss_dict['loss_voxel_geo'] = self.weight_voxel_geo_scal * geo_scal_loss(output_voxels, target_voxels, 255, non_empty_idx=self.empty_idx, camera_mask=camera_mask)
        loss_dict['loss_voxel_lovasz'] = self.weight_voxel_lovasz * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=self.empty_idx, camera_mask=camera_mask)

        return loss_dict

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        # extract volumn feature
        img = [i.cuda() for i in img]
        kwargs = Scatter(kwargs)
        img_inputs = self.prepare_inputs(img, stereo=True)
        img_feats, _ = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc

        # TODO: save feature
        
        if not self.if_post_finetune:  # using density head & semantic head
            # predict SDF
            density_prob = self.density_mlp(voxel_feats)
            density = density_prob[...,0]
            semantic = self.semantic_mlp(voxel_feats)

            # SDF --> Occupancy
            no_empty_mask = density > self.test_threshold
            # no_empty_mask = density_prob.argmax(-1) == 0
            semantic_res = semantic.argmax(-1)

            B, H, W, Z, C = voxel_feats.shape
            occ = torch.ones((B,H,W,Z), dtype=semantic_res.dtype).to(semantic_res.device)
            occ = occ * (self.num_classes-1)
            occ[no_empty_mask] = semantic_res[no_empty_mask]

            occ = occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)

            geo_occ = torch.ones((B,H,W,Z), dtype=semantic_res.dtype).to(semantic_res.device)
            geo_occ = geo_occ * (self.num_classes-1)
            geo_occ[no_empty_mask] = 0
            geo_occ = geo_occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        
        else:  # using occpancy head
            voxel_feat = voxel_feats[0, ...]
            voxel_feat = voxel_feat.permute(3, 0, 1, 2).unsqueeze(0)
            occ_pred_result = self.occupancy_head([voxel_feat])

            occ_pred = occ_pred_result['output_voxels'][0].squeeze(0)
            occ_pred = occ_pred.permute(1, 2, 3, 0)
            occ_pred = occ_pred.argmax(-1)

            geo_occ = torch.ones_like(occ_pred, dtype=occ_pred.dtype).to(occ_pred.device)

            if self.dataset_type == 'NuPlan':
                geo_occ = geo_occ * self.empty_idx

                nonempty_mask = occ_pred < self.empty_idx
                empty_mask = occ_pred >= self.empty_idx
                occ_pred[empty_mask] = self.empty_idx
                occ = occ_pred.cpu().numpy().astype(np.uint8)
            else:
                geo_occ = geo_occ * (self.num_classes-1)

                nonempty_mask = occ_pred != 17
                occ = occ_pred.cpu().numpy().astype(np.uint8)

            geo_occ[nonempty_mask] = 0
            geo_occ = geo_occ.cpu().numpy().astype(np.uint8)

        return {
            'semantic_occ': [occ],
            'geo_occ': [geo_occ],
        }
          

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # extract volumn feature
        img_inputs = self.prepare_inputs(img_inputs, stereo=True)
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
                bda, curr2adjsensor = img_inputs
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc

        occ_preds = []
        for batch_idx in range(voxel_feats.shape[0]):
            voxel_feat = voxel_feats[batch_idx, ...]
            voxel_feat = voxel_feat.permute(3, 0, 1, 2).unsqueeze(0)
            occ_pred_result = self.occupancy_head([voxel_feat])
            occ_preds.append(occ_pred_result['output_voxels'][0].squeeze(0))
        
        occ_preds = torch.stack(occ_preds, dim=0)

        # predict SDF
        density_prob = self.density_mlp(voxel_feats)
        density = density_prob[..., 0]
        semantic = self.semantic_mlp(voxel_feats)
        color = self.color_mlp(voxel_feats)

        # compute loss
        losses = dict()

        # 3D occupancy loss
        if self.if_post_finetune:
            voxel_semantics = kwargs['voxel_semantics']

            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            if kwargs['mask_camera'] is not None:
                camera_mask = kwargs['mask_camera']
                camera_mask = camera_mask.type(torch.bool)
                loss_voxel = self.loss_voxel(
                    occ_preds,
                    voxel_semantics,
                    camera_mask=None  # w/o camera mask
                )
            else:
                loss_voxel = self.loss_voxel(
                    occ_preds,
                    voxel_semantics,
                    camera_mask=None  # w/o camera mask
                )
            losses.update(loss_voxel)
        else:
            voxel_semantics = kwargs['voxel_semantics']

            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            loss_sup_voxel = self.loss_sup_occupancy(occ_preds, voxel_semantics)
            losses.update(loss_sup_voxel)
        
        # 2D rendering loss
        if self.if_render:
            loss_rendering = self.nerf_head(density, semantic, color, if_pretrain=self.if_pretrain, dataset_type=self.dataset_type, rays=kwargs['rays'], bda=bda)
            losses.update(loss_rendering)
        else:
            loss_sup_semantic = self.loss_sup(semantic, 'semantic', self.num_classes-1)
            loss_sup_color = self.loss_sup(color, 'color', 3)
            loss_sup_density = self.loss_sup(density, 'density', 1)

            losses.update(loss_sup_semantic)
            losses.update(loss_sup_color)
            losses.update(loss_sup_density)
        
        # lss-depth loss (BEVStereo's feature)
        if self.use_lss_depth_loss:
            loss_depth = self.img_view_transformer.get_depth_loss(kwargs['gt_depth'], depth)
            losses['loss_lss_depth'] = loss_depth
        
        if self.if_pretrain:
            loss_sup_semantic = self.loss_sup(semantic, 'semantic', self.num_classes-1)
            losses.update(loss_sup_semantic)

        return losses

