# Copyright (c) Phigent Robotics. All rights reserved.
from mmdet3d.models.detectors.bevdet_occ import BEVStereo4DOCC
from .opus import OPUS
import torch.nn.functional as F
import torch
import time
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn.bricks.transformer import MultiheadAttention
from torch import nn
import numpy as np
from mmdet3d.models import builder
from .opus_transformer import OPUSSelfAttention, OPUSCrossAttention
from mmcv.cnn import bias_init_with_prob
from mmdet3d.models.detectors.loss import CE_ssc_loss, sem_scal_loss, geo_scal_loss, l1_loss, l2_loss
from mmdet3d.models.detectors.lovasz_softmax import lovasz_softmax
from IPython import embed
from mmdet3d.models.sparsedetectors.bbox.utils import decode_points, encode_points, trans_coords,get_matched_inds
from mmdet3d.models.heads import DownScaleModule3DCustom
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
device = torch.device('cuda')
# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947,
                                   497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031,
                                   141625221, 2307405309])
import time
# from ptflops import get_model_complexity_info
from thop import profile

def Scatter(src_dict):
    for key, value in src_dict.items():
        if isinstance(value, torch.Tensor):
            src_dict[key] = value.cuda()
        if isinstance(value, dict):
            src_dict[key] = Scatter(value)
        if isinstance(value, list):
            if isinstance(value[0], dict):
                src_dict[key] = [Scatter(v) for v in value]
            if isinstance(value[0], torch.Tensor):
                src_dict[key] = [v.cuda() for v in value]
    return src_dict


@DETECTORS.register_module()
class SparseWorld4DTraj(OPUS):
    def __init__(self,
                 out_dim=32,
                 dataset_type='Nuscenes',
                 num_classes=18,
                 test_threshold=8.5,
                 drop_out=0.1,
                 use_3d_loss=True,
                 if_pretrain=False,
                 if_render=True,
                 if_post_finetune=False,
                 finetune_epoch = 0,
                 num_out_query=600,
                 empty_idx=17,
                 use_focal_loss=True,
                 balance_cls_weight=True,
                 final_softplus=True,
                 **kwargs):
        super(SparseWorld4DTraj, self).__init__(**kwargs)
        self.dataset_type = dataset_type
        self.out_dim = out_dim
        self.use_3d_loss = use_3d_loss
        self.test_threshold = test_threshold
        self.num_refines = self.pts_bbox_head.transformer.num_refines[-1]
        self.balance_cls_weight = balance_cls_weight
        self.final_softplus = final_softplus
        # self.if_pretrain = if_pretrain
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

        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss'))

        self.velocity_dim = 3
        self.past_frame = 5
        self.pc_range = self.pts_bbox_head.pc_range

        self.plan_head = nn.Sequential(
            nn.Linear(self.velocity_dim * (self.past_frame + 2), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.out_dim)
        )
        self.ego_cross_attn = OPUSCrossAttention(self.out_dim, 8, drop_out, self.pts_bbox_head.pc_range)

        self.position_encoder = nn.Sequential(
            nn.Linear(4 * self.num_refines, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
        )

        self.reg_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 3)
        )

        self.vel_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 2)
        )

        self.cls_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 17)
        )
        self.points_scale_branch = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3),
        )

        self.traj_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, 2),
        )
        self.l2_loss = l2_loss()

        self.box_mode_3d = Box3DMode.LIDAR
        self.planning_metric = None
        self.finetune_epoch = finetune_epoch

        self.pred_num = torch.zeros(18).cuda()

        self.gt_traj = list()
        self.tau = list()

    def init_weights(self):
        self.pts_bbox_head.init_weights()
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def set_epoch(self, epoch):
        self.curr_epoch = epoch
        if epoch<self.finetune_epoch:
            self.pretrain = True
            self.pts_bbox_head.pretrain = True
            if getattr(self.pts_bbox_head, 'num_stamps_all', None) is not None:
                self.pts_bbox_head.num_stamps_all[:] = 1  # avoid diving 0
        else:
            self.pretrain = False
            self.pts_bbox_head.pretrain = False
            num_stamps = self.pts_bbox_head.num_stamps_all / torch.sum(self.pts_bbox_head.num_stamps_all, dim=-1,
                                                                       keepdim=True)
            self.pts_bbox_head.ind_stamps_all = get_matched_inds(num_stamps, [self.num_query] + self.num_fu_query)

            self.pts_bbox_head.reset_mask()


    def trans_points(self, points_proposal, points_delta, trans_matrix):

        inv_trans_matrix = torch.linalg.inv(trans_matrix.cpu()).cuda()
        points_proposal = decode_points(points_proposal, self.pc_range)
        # points_proposal = points_proposal.mean(dim=2, keepdim=True) fengze
        new_points = torch.matmul(points_proposal, trans_matrix[..., :3, :3].transpose(1, 2)) + trans_matrix[..., None,
                                                                                                :3, 3]
        new_points = new_points + points_delta
        new_points = torch.matmul(new_points, inv_trans_matrix[..., :3, :3].transpose(1, 2)) + inv_trans_matrix[...,
                                                                                               None, :3, 3]

        return encode_points(new_points, self.pc_range)

    def refine_points(self, points_proposal, points_delta):
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def loss_traj(self, pred_traj, gt_traj, ego_interval):
        loss_dict = dict()
        loss_dict[f'loss_traj_{str(ego_interval)}s'] = self.l2_loss(pred_traj, gt_traj)

        return loss_dict

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        img = [img] if img is None else img

        result = self.simple_test(img_metas[0], img[0], **kwargs)

        

        return result


    def forward_backbone(self,img,img_metas,**kwargs):

        B = img.shape[0]
        ego_states = kwargs['temporal_ego_states'][0]
        bs, _, dim_ = ego_states.shape
        ego_states = ego_states.view((bs, 1, dim_))
        ego_feat = self.plan_head(ego_states)
        points_scale = self.points_scale_branch(ego_feat)
        points_scale = torch.tanh(points_scale)
        self.pts_bbox_head.points_scale = (points_scale + 1) / 2 * (1.5 - 0.8) + 0.8

        if self.training:
            img_feats = self.extract_feat(img, img_metas)
            outs = self.pts_bbox_head(img_feats, img_metas)
        else:
            outs = self.simple_test_online(img_metas,img)

        ind_stamps_all = self.pts_bbox_head.ind_stamps_all
        query_feat = outs['query_feat']
        query_pos = outs['all_refine_pts'][-1]
        query_cls = outs['all_cls_scores'][-1]

        curr_query_feat = query_feat[:, ind_stamps_all == 0]
        curr_query_pos = query_pos[:, ind_stamps_all == 0].detach()
        curr_query_timestamp = query_pos.new_zeros(B,self.num_query,self.num_refines,1)
        curr_query_cls = query_cls[:,ind_stamps_all==0]
        outputs = dict(cls_score = curr_query_cls,
                       refine_pts = curr_query_pos,
                       outs = outs)

        forecast_points_list = list()
        forecast_semantics_list = list()
        pred_trajs_list = list()
        forecast_points_mask_list = list()
        if self.training:
            num_fu_frames = max(1,min(self.curr_epoch - self.finetune_epoch+1, self.num_fu_frames))
        else:
            num_fu_frames = self.num_fu_frames

        for interval in range(num_fu_frames):
            # fu_query_feat = outs['fu_query_feat'].reshape(B,self.num_fu_frames,self.num_fu_query,self.out_dim)
            fused_ego_feat,_ = self.ego_cross_attn(ego_feat.new_ones(B, 1, 3)*0.5, ego_feat, curr_query_pos.detach(),
                                                    curr_query_feat.detach(), )
            pred_traj = self.traj_head(fused_ego_feat)
            pred_trajs_list.append(pred_traj)

            curr_query_feat = torch.cat([curr_query_feat, query_feat[:, ind_stamps_all == interval + 1]], dim=1)
            curr_query_pos = torch.cat([curr_query_pos, query_pos[:, ind_stamps_all == interval + 1]],
                                          dim=1).detach()
            if interval<6:
                curr_query_timestamp = torch.cat([curr_query_timestamp,curr_query_pos.new_ones(B,self.num_fu_query[interval],self.num_refines,1)*0.5],dim=1)

            pos_embedding = self.position_encoder(torch.cat([curr_query_pos,curr_query_timestamp],dim=-1).flatten(2,3))
            curr_query_feat = curr_query_feat + fused_ego_feat + pos_embedding

            reg_offset = self.reg_branch(curr_query_feat).unflatten(-1, (-1, 3)) * 0.5
            cls_score = self.cls_branch(curr_query_feat).unflatten(-1, (-1, 17))
            vel_offset = self.vel_branch(curr_query_feat).unflatten(-1, (-1, 2))
            #
            pred_labels = cls_score.argmax(-1)
            pred_moving_mask = torch.logical_and(pred_labels >= 2, pred_labels <= 10).unsqueeze(-1)
            reg_offset = torch.cat(
                [reg_offset[..., :2] + vel_offset * pred_moving_mask, reg_offset[..., 2:]], dim=-1)
            reg_offset = reg_offset.flatten(2, 3)

            # reg_offset = self.reg_branch(curr_query_feat) * 0.5
            curr_query_pos = self.refine_points(curr_query_pos, reg_offset)
            forecast_semantics_list.append(cls_score)
            forecast_points_list.append(curr_query_pos)
            if self.training:
                ego2lidar =torch.tensor(np.stack([meta['ego2lidar'] for meta in img_metas]) ,device=device,dtype=torch.float32)
                gt_traj = kwargs['temporal_trajs'][:, interval:interval + 1, :]
                pred_traj_expand = torch.cat([-gt_traj, torch.zeros_like(pred_traj[:, :, :1])], dim=-1)

                gt_points = self.trans_points(curr_query_pos.flatten(1, 2), pred_traj_expand, ego2lidar).reshape(
                    curr_query_pos.shape)
                forecast_points_mask_list.append(gt_points[..., 0] >= 0)

        if not self.pretrain and len(pred_trajs_list)<self.num_fu_frames :
            fused_ego_feat,_ = self.ego_cross_attn(ego_feat.new_zeros(B, 1, 3), ego_feat, curr_query_pos,
                                                 curr_query_feat)
            pred_traj = self.traj_head(fused_ego_feat)
            pred_trajs_list.append(pred_traj)

        outputs.update(
                        dict(forecast_semantics_list = forecast_semantics_list,
                       forecast_points_list = forecast_points_list,
                       pred_trajs_list = pred_trajs_list,
                       forecast_points_mask_list = forecast_points_mask_list))
        return outputs

    def simple_test(self,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""

        for key in kwargs.keys():
            kwargs[key] = kwargs[key][0]

        outputs = self.forward_backbone(img, img_metas, **kwargs)
        cls_score, curr_query_pos, outs = outputs['cls_score'],outputs['refine_pts'],outputs['outs']

        pred_dict = dict(cls_scores=outs['all_cls_scores'][-1][:,self.pts_bbox_head.ind_stamps_all==0], refine_pts=outs['all_refine_pts'][-1][:,self.pts_bbox_head.ind_stamps_all==0])
        occ_pred = self.pts_bbox_head.get_occ(pred_dict)[0]
        # self.pred_num += torch.bincount(occ_pred.flatten())
        geo_pred = torch.ones_like(occ_pred) * 17
        geo_pred[occ_pred != 17] = 0
        res_dict = {f'semantic_occ_0s': [occ_pred.cpu().numpy()],
                    f'geo_occ_0s': [geo_pred.cpu().numpy()]}

        forecast_points_list, forecast_semantics_list, pred_trajs_list = \
            outputs['forecast_points_list'],outputs['forecast_semantics_list'],outputs['pred_trajs_list']

        for interval in range(self.num_fu_frames):
            input_dict = dict(cls_scores=forecast_semantics_list[interval],
                              refine_pts=forecast_points_list[interval])
            occ_forecast = self.pts_bbox_head.get_occ(input_dict)[0]  # eval for single batch
            geo_forecast = torch.ones_like(occ_forecast) * 17
            geo_forecast[occ_forecast != 17] = 0
            # pred_traj_list.append(pred_traj)
            res_dict.update({
                f'semantic_occ_{int(interval + 1)}s': [occ_forecast.cpu().numpy()],
                f'geo_occ_{int(interval + 1)}s': [geo_forecast.cpu().numpy()],
            })

        res_dict['pred_traj'] = torch.cat(pred_trajs_list, 1)
        return res_dict

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      **kwargs):

        temporal_semantics = kwargs['temporal_semantics']
        B = img.shape[0]
        temporal2ego = kwargs['temporal2ego']
        outputs = self.forward_backbone(img,img_metas,**kwargs)
        cls_score,refine_pts,outs = outputs['cls_score'],outputs['refine_pts'],outputs['outs']

        losses = dict()
        ind_stamps_all = self.pts_bbox_head.ind_stamps_all
        if self.pretrain:
            loss_inputs = [voxel_semantics, temporal_semantics, temporal2ego, outs]
            losses.update(self.pts_bbox_head.loss_pretrain(*loss_inputs))
        else:
            # outs_inits = dict(init_points = outs['init_points'],all_cls_scores = [], all_refine_pts = [])
            loss_inputs = [voxel_semantics, temporal_semantics, temporal2ego, outs]
            losses.update(self.pts_bbox_head.loss_pretrain(*loss_inputs))
            outs['init_points'] = None
            for i in range(len(outs['all_cls_scores'])):
                outs['all_cls_scores'][i] = outs['all_cls_scores'][i][:,ind_stamps_all==0]
                outs['all_refine_pts'][i] = outs['all_refine_pts'][i][:,ind_stamps_all==0]
            loss_inputs = [voxel_semantics,outs,]
            losses.update(self.pts_bbox_head.loss(*loss_inputs))

        forecast_points_list = outputs['forecast_points_list']
        forecast_semantics_list = outputs['forecast_semantics_list']
        pred_trajs_list = outputs['pred_trajs_list']
        forecast_points_mask_list = outputs['forecast_points_mask_list']

        voxel_semantics_temporal = [sem['voxel_semantics'] for sem in kwargs['temporal_semantics'].values()]

        num_fu_frames = len(forecast_semantics_list)
        losses.update(
            self.pts_bbox_head.loss_future(voxel_semantics_temporal[:num_fu_frames],
                                           forecast_points_list,forecast_semantics_list,forecast_points_mask_list))
        for interval,pred_traj in enumerate(pred_trajs_list):

            loss_traj = self.loss_traj(pred_traj.squeeze(1), kwargs['temporal_trajs'][:, interval, :], interval + 1)
            losses.update(loss_traj)

        return losses