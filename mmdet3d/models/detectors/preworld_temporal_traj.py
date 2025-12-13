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

from .loss import CE_ssc_loss, sem_scal_loss, geo_scal_loss, l1_loss, l2_loss
from .lovasz_softmax import lovasz_softmax
from IPython import embed

from mmdet3d.models.heads import DownScaleModule3DCustom
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes

# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221, 2307405309])
def Scatter(src_dict):
    for key, value in src_dict.items():
        if isinstance(value, torch.Tensor):
            src_dict[key] = value.cuda()
        if isinstance(value, dict):
            src_dict[key] = Scatter(value)
        if isinstance(value,list):
            if isinstance(value[0],dict):
                src_dict[key] = [Scatter(v) for v in value]
            if isinstance(value[0],torch.Tensor):
                src_dict[key] = [v.cuda() for v in value]
    return src_dict

@DETECTORS.register_module()
class PreWorld4DTraj(BEVStereo4DOCC):
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
        super(PreWorld4DTraj, self).__init__(use_predicter=False, **kwargs)
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
        
        self.velocity_dim = 3
        self.past_frame = 5
        self.plan_head = nn.Sequential(
            nn.Linear(self.velocity_dim * (self.past_frame + 2), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.out_dim)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(self.out_dim*2, self.out_dim*4),
            nn.Softplus(),
            nn.Linear(self.out_dim*4, self.out_dim),
        )

        self.downscale = DownScaleModule3DCustom(in_dim=self.out_dim)

        self.ego_fusion_head = nn.Sequential(
            nn.Linear(self.out_dim*5, self.out_dim*8),
            nn.Softplus(),
            nn.Linear(self.out_dim*8, self.out_dim*4),
            nn.Softplus(),
            nn.Linear(self.out_dim*4, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2, self.out_dim),
        )

        self.traj_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2, 2),
        )
        self.l2_loss = l2_loss()

        self.box_mode_3d = Box3DMode.LIDAR
        self.planning_metric = None

    def set_epoch(self, epoch): 
        self.curr_epoch = epoch

    def loss_sup(self, pred, tag, cls_num, ego_interval):
        sup_pred = torch.ones_like(pred).to(pred.device)
        loss_sup = nn.CrossEntropyLoss(reduction="mean")(pred.reshape(-1, cls_num), sup_pred.reshape(-1, cls_num))

        loss_ = dict()
        loss_[f'loss_sup_{tag}_{str(ego_interval)}s'] = loss_sup * 0.

        return loss_
    

    def loss_sup_occupancy(self, output_voxels, target_voxels, ego_interval):
        loss_dict = dict()
        cls_weights = torch.cat([self.class_weights, torch.tensor([0])])
        loss_dict[f'loss_sup_voxel_{str(ego_interval)}s'] = CE_ssc_loss(output_voxels, target_voxels, cls_weights.type_as(output_voxels), 255) * 0.

        return loss_dict

    
    def loss_voxel(self, output_voxels, target_voxels, ego_interval, camera_mask=None):
        output_voxels[torch.isnan(output_voxels)] = 0
        output_voxels[torch.isinf(output_voxels)] = 0
        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        if self.use_focal_loss:
            cls_weights = torch.cat([self.class_weights, torch.tensor([0])])
            loss_voxel_ce = self.weight_voxel_ce * self.focal_loss(output_voxels, target_voxels, cls_weights.type_as(output_voxels), 255, camera_mask=camera_mask)
        else:
            cls_weights = torch.cat([self.class_weights, torch.tensor([0])])
            loss_voxel_ce = self.weight_voxel_ce * CE_ssc_loss(output_voxels, target_voxels, cls_weights.type_as(output_voxels), 255)
        
        loss_voxel_sem = self.weight_voxel_sem_scal * sem_scal_loss(output_voxels, target_voxels, 255, camera_mask=camera_mask)
        loss_voxel_geo = self.weight_voxel_geo_scal * geo_scal_loss(output_voxels, target_voxels, 255, non_empty_idx=self.empty_idx, camera_mask=camera_mask)
        loss_voxel_lovasz = self.weight_voxel_lovasz * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=self.empty_idx, camera_mask=camera_mask)

        # total_loss = loss_voxel_ce + loss_voxel_sem + loss_voxel_geo + loss_voxel_lovasz
        loss_dict[f'loss_voxel_ce_{str(ego_interval)}s'] = loss_voxel_ce
        loss_dict[f'loss_voxel_sem_{str(ego_interval)}s'] = loss_voxel_sem
        loss_dict[f'loss_voxel_geo_{str(ego_interval)}s'] = loss_voxel_geo
        loss_dict[f'loss_voxel_lovasz_{str(ego_interval)}s'] = loss_voxel_lovasz
        # loss_dict[f'loss_voxel_{str(ego_interval)}s'] = total_loss

        return loss_dict

    def loss_traj(self, pred_traj, gt_traj, ego_interval):
        loss_dict = dict()
        loss_dict[f'loss_traj_{str(ego_interval)}s'] = self.l2_loss(pred_traj, gt_traj)

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
        # img_feats = [torch.zeros([1,32,16,200,200]).cuda()]
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc

        res_dict = dict()
        
        if not self.if_post_finetune:  # using density head & semantic head
            temporal_ego_states = kwargs['temporal_ego_states'][0]
            _, x_, y_, z_, _ = voxel_feats.shape
            
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

            res_dict.update({
                f'semantic_occ_0s': [occ],
                f'geo_occ_0s': [geo_occ],
            })

            ego_intervals = [0, 1, 2, 3, 4, 5]
            for ego_interval in ego_intervals:
                ego_states = temporal_ego_states[0]

                bs, _, dim_ = ego_states.shape
                ego_states = ego_states.reshape((bs, dim_))
                ego_feats = self.plan_head(ego_states)
                ego_feats = ego_feats.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                ego_feats = ego_feats.repeat_interleave(z_, dim=3).repeat_interleave(y_, dim=2).repeat_interleave(x_, dim=1)

                updated_voxel_feats = torch.cat([voxel_feats, ego_feats], dim=-1)
                res_voxel_feats = self.fusion_head(updated_voxel_feats)

                fused_voxel_feats = res_voxel_feats + voxel_feats  # residual connection

                # predict SDF
                density_prob = self.density_mlp(fused_voxel_feats)
                density = density_prob[...,0]
                semantic = self.semantic_mlp(fused_voxel_feats)

                # SDF --> Occupancy
                no_empty_mask = density > self.test_threshold
                # no_empty_mask = density_prob.argmax(-1) == 0
                semantic_res = semantic.argmax(-1)

                B, H, W, Z, C = fused_voxel_feats.shape
                occ = torch.ones((B,H,W,Z), dtype=semantic_res.dtype).to(semantic_res.device)
                occ = occ * (self.num_classes-1)
                occ[no_empty_mask] = semantic_res[no_empty_mask]

                occ = occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)

                geo_occ = torch.ones((B,H,W,Z), dtype=semantic_res.dtype).to(semantic_res.device)
                geo_occ = geo_occ * (self.num_classes-1)
                geo_occ[no_empty_mask] = 0
                geo_occ = geo_occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)

                future_interval = ego_interval + 2

                res_dict.update({
                    f'semantic_occ_{int(future_interval)}s': [occ],
                    f'geo_occ_{int(future_interval)}s': [geo_occ],
                })

                voxel_feats = fused_voxel_feats.clone()  # recursive
        
        else:  # using occpancy head
            temporal_ego_states = kwargs['temporal_ego_states'][0]
            _, x_, y_, z_, _ = voxel_feats.shape

            voxel_feat = voxel_feats[0, ...]
            voxel_feat = voxel_feat.permute(3, 0, 1, 2).unsqueeze(0)
            occ_pred_result = self.occupancy_head([voxel_feat])

            occ_pred = occ_pred_result['output_voxels'][0].squeeze(0)
            occ_pred = occ_pred.permute(1, 2, 3, 0)
            occ_pred = occ_pred.argmax(-1)

            geo_occ = torch.ones_like(occ_pred, dtype=occ_pred.dtype).to(occ_pred.device)
            geo_occ = geo_occ * (self.num_classes-1)

            nonempty_mask = occ_pred != 17
            occ = occ_pred.cpu().numpy().astype(np.uint8)

            geo_occ[nonempty_mask] = 0
            geo_occ = geo_occ.cpu().numpy().astype(np.uint8)

            res_dict.update({
                f'semantic_occ_0s': [occ],
                f'geo_occ_0s': [geo_occ],
            })

            ego_intervals = [0, 1, 2, 3, 4, 5,6 ]
            pred_traj_list = []
            for ego_interval in ego_intervals:
                ego_states = temporal_ego_states[0]

                bs, _, dim_ = ego_states.shape
                ego_states = ego_states.reshape((bs, dim_))
                ego_feats = self.plan_head(ego_states)
                identity = ego_feats
                ego_feats = ego_feats.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                ego_feats = ego_feats.repeat_interleave(z_, dim=3).repeat_interleave(y_, dim=2).repeat_interleave(x_, dim=1)

                updated_voxel_feats = torch.cat([voxel_feats, ego_feats], dim=-1)
                res_voxel_feats = self.fusion_head(updated_voxel_feats)

                fused_voxel_feats = res_voxel_feats + voxel_feats  # residual connection

                downscaled_voxel_feats = self.downscale(fused_voxel_feats)
                downscaled_voxel_feats = downscaled_voxel_feats.squeeze(1).squeeze(1).squeeze(1)
                updated_ego_feats = torch.cat([identity, downscaled_voxel_feats], dim=-1)
                res_ego_feats = self.ego_fusion_head(updated_ego_feats)
                fused_ego_feats = identity + res_ego_feats
                pred_traj = self.traj_head(fused_ego_feats)
                pred_traj_list.append(pred_traj)

                fused_voxel_feat = fused_voxel_feats[0, ...]
                fused_voxel_feat = fused_voxel_feat.permute(3, 0, 1, 2).unsqueeze(0)
                occ_pred_result = self.occupancy_head([fused_voxel_feat])

                occ_pred = occ_pred_result['output_voxels'][0].squeeze(0)
                occ_pred = occ_pred.permute(1, 2, 3, 0)
                occ_pred = occ_pred.argmax(-1)

                geo_occ = torch.ones_like(occ_pred, dtype=occ_pred.dtype).to(occ_pred.device)
                geo_occ = geo_occ * (self.num_classes-1)

                nonempty_mask = occ_pred != 17
                occ = occ_pred.cpu().numpy().astype(np.uint8)

                geo_occ[nonempty_mask] = 0
                geo_occ = geo_occ.cpu().numpy().astype(np.uint8)

                future_interval = ego_interval + 1

                res_dict.update({
                    f'semantic_occ_{int(future_interval)}s': [occ],
                    f'geo_occ_{int(future_interval)}s': [geo_occ],
                })

                voxel_feats = fused_voxel_feats.clone()  # recursive
        res_dict['pred_traj'] = torch.stack(pred_traj_list,1)
        return res_dict

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
        _, x_, y_, z_, _ = voxel_feats.shape

        # compute loss
        losses = dict()

        # lss-depth loss (BEVStereo's feature)
        if self.use_lss_depth_loss:
            loss_depth = self.img_view_transformer.get_depth_loss(kwargs['gt_depth'], depth)
            losses['loss_lss_depth'] = loss_depth
        
        curr_interval = 0
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

        # 3D occ loss
        if self.if_post_finetune:
            voxel_semantics = kwargs['voxel_semantics']
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            loss_voxel = self.loss_voxel(
                occ_preds,
                voxel_semantics,
                curr_interval,
                camera_mask=None  # w/o camera mask
            )
            losses.update(loss_voxel)
        else:
            voxel_semantics = kwargs['voxel_semantics']
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            loss_sup_voxel = self.loss_sup_occupancy(occ_preds, voxel_semantics, curr_interval)
            losses.update(loss_sup_voxel)
        
        # 2D rendering loss
        if self.if_render:
            loss_rendering = self.nerf_head(density, semantic, color, if_pretrain=self.if_pretrain, if_temporal=True, dataset_type=self.dataset_type, rays=kwargs['rays'], bda=bda, interval=curr_interval)
            losses.update(loss_rendering)
        else:
            loss_sup_semantic = self.loss_sup(semantic, 'semantic', self.num_classes-1, curr_interval)
            loss_sup_color = self.loss_sup(color, 'color', 3, curr_interval)
            loss_sup_density = self.loss_sup(density, 'density', 1, curr_interval)

            losses.update(loss_sup_semantic)
            losses.update(loss_sup_color)
            losses.update(loss_sup_density)

        # TODO: add temporal information and ego status here
        if self.if_render:
            if self.curr_epoch <= 2:
                future_intervals = [0, 1]
            else:
                future_intervals = range(0, min(self.curr_epoch - 1, 6))
        else:
            if self.curr_epoch <= 4:
                future_intervals = [0, 1]
            else:
                future_intervals = range(0, min(int((self.curr_epoch - 3) // 2) + 1, 6))
        
        for ego_interval in future_intervals:
            ego_states = kwargs['temporal_ego_states'][0]
            bs, _, dim_ = ego_states.shape
            ego_states = ego_states.reshape((bs, dim_))
            ego_feats = self.plan_head(ego_states)
            identity = ego_feats
            ego_feats = ego_feats.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            ego_feats = ego_feats.repeat_interleave(z_, dim=3).repeat_interleave(y_, dim=2).repeat_interleave(x_, dim=1)

            updated_voxel_feats = torch.cat([voxel_feats, ego_feats], dim=-1)
            res_voxel_feats = self.fusion_head(updated_voxel_feats)
            fused_voxel_feats = res_voxel_feats + voxel_feats  # residual connection

            downscaled_voxel_feats = self.downscale(fused_voxel_feats)
            downscaled_voxel_feats = downscaled_voxel_feats.squeeze(1).squeeze(1).squeeze(1)
            updated_ego_feats = torch.cat([identity, downscaled_voxel_feats], dim=-1)
            res_ego_feats = self.ego_fusion_head(updated_ego_feats)
            fused_ego_feats = identity + res_ego_feats

            pred_traj = self.traj_head(fused_ego_feats)

            occ_preds = []
            for batch_idx in range(fused_voxel_feats.shape[0]):
                fused_voxel_feat = fused_voxel_feats[batch_idx, ...]
                fused_voxel_feat = fused_voxel_feat.permute(3, 0, 1, 2).unsqueeze(0)
                occ_pred_result = self.occupancy_head([fused_voxel_feat])
                occ_preds.append(occ_pred_result['output_voxels'][0].squeeze(0))
            
            occ_preds = torch.stack(occ_preds, dim=0)

            # predict SDF
            density_prob = self.density_mlp(fused_voxel_feats)
            density = density_prob[..., 0]
            semantic = self.semantic_mlp(fused_voxel_feats)
            color = self.color_mlp(fused_voxel_feats)

            future_interval = ego_interval + 1

            # 3D occ loss
            if self.if_post_finetune:
                # voxel_semantics = kwargs['voxel_semantics']
                voxel_semantics = kwargs['temporal_semantics'][future_interval]['voxel_semantics']

                assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
                loss_voxel = self.loss_voxel(
                    occ_preds,
                    voxel_semantics,
                    future_interval,
                    camera_mask=None  # w/o camera mask
                )
                losses.update(loss_voxel)
            else:
                voxel_semantics = kwargs['voxel_semantics']

                assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
                loss_sup_voxel = self.loss_sup_occupancy(occ_preds, voxel_semantics, future_interval)
                losses.update(loss_sup_voxel)
            
            # 2D rendering loss
            if self.if_render:
                loss_rendering = self.nerf_head(density, semantic, color, if_pretrain=self.if_pretrain, if_temporal=True, dataset_type=self.dataset_type, rays=kwargs['temporal_rays'][future_interval], bda=bda, interval=future_interval)
                losses.update(loss_rendering)
            else:
                loss_sup_semantic = self.loss_sup(semantic, 'semantic', self.num_classes-1, future_interval)
                loss_sup_color = self.loss_sup(color, 'color', 3, future_interval)
                loss_sup_density = self.loss_sup(density, 'density', 1, future_interval)

                losses.update(loss_sup_semantic)
                losses.update(loss_sup_color)
                losses.update(loss_sup_density)
            
            # traj loss
            gt_trajs = kwargs['temporal_trajs'][:, future_interval - 1, :]

            loss_traj = self.loss_traj(pred_traj, gt_trajs, future_interval)
            losses.update(loss_traj)
            
            voxel_feats = fused_voxel_feats.clone()  # recursive

        return losses

