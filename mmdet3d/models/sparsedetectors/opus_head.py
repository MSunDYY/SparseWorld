import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn, Voxelization
from mmdet.core import multi_apply
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmdet3d.models.sparsedetectors.bbox.utils import decode_points,dist_loss_weight,get_matched_inds
import torch

colors = torch.tensor([
    [1.00, 0.00, 0.00],   # 红 Red
    [1.00, 0.65, 0.00],   # 橙 Orange
    [1.00, 1.00, 0.00],   # 黄 Yellow
    [0.00, 0.50, 0.00],   # 绿 Green
    [0.00, 1.00, 1.00],   # 青 Cyan
    [0.00, 0.00, 1.00],   # 蓝 Blue
    [0.50, 0.00, 0.50],   # 紫 Purple
])


@HEADS.register_module()
class OPUSHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 num_fu_query,
                 num_fu_frames,
                 transformer=None,
                 pc_range=[],
                 empty_label=17,
                 voxel_size=[],
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 is_pretrain=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_pts=dict(type='L1Loss'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.num_fu_query = num_fu_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.num_fu_frames = num_fu_frames
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        self.loss_cls = build_loss(loss_cls)
        self.loss_pts = build_loss(loss_pts)
        self.loss_stamp = torch.nn.BCEWithLogitsLoss()
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.voxel_generator = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=10,
            max_voxels=self.num_query * self.num_refines[-1],
            deterministic=False
        )

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        self.pretrain = is_pretrain
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)

        self._init_layers()
        self.scene_size_new = torch.tensor([160, 120, 9.6]).cuda()
        self.pc_range_new = torch.tensor([-80, -60, -3, 80, 60, 6.6]).cuda()
        self.voxel_num_new = torch.tensor([400,300,24]).cuda()
        self.register_buffer("num_stamps_all", torch.ones(self.num_query+sum(self.num_fu_query),self.num_fu_frames+1).long())


    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query + sum(self.num_fu_query), 3)
        nn.init.uniform_(self.init_points.weight[:,0], 0, 1.1)
        nn.init.uniform_(self.init_points.weight[:,1:],0,1)

    def init_weights(self):
        self.transformer.init_weights()

    def forward(self, mlvl_feats, img_metas):

        B, Q = mlvl_feats[0].shape[0], self.num_query + sum(self.num_fu_query)
        # self.init_points.weight.data.clamp_(0.0, 1.0)
        init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)

        if getattr(self,'points_scale',None) is not None:
            init_points = init_points * self.points_scale.unsqueeze(1)
            # from open3d_vis_utils import draw_scenes
            # draw_scenes(init_points.reshape(-1,3))
        query_feat = init_points.new_zeros(B, Q, self.embed_dims)
        if getattr(self,'ind_stamps_all',None) is None:
            num_stamps_all = self.num_stamps_all.float()
            num_stamps_all = num_stamps_all / torch.sum(num_stamps_all, -1, keepdim=True)
            ind_stamps_all = get_matched_inds(num_stamps_all, [self.num_query] + self.num_fu_query)
            self.ind_stamps_all = ind_stamps_all
            if not self.training:
                self.reset_mask()

        # fu_query_feat = fu_init_points.new_zeros(B, fu_Q, self.embed_dims)
        if self.pretrain:
            num_stamps_all = self.num_stamps_all.float()
            num_stamps_all = num_stamps_all / torch.sum(num_stamps_all, -1, keepdim=True)
            ind_stamps_all = get_matched_inds(num_stamps_all, [self.num_query] + self.num_fu_query)
            self.ind_stamps_all = ind_stamps_all

        query_feat, cls_scores, refine_pts = self.transformer(
            init_points,
            query_feat,
            mlvl_feats,
            img_metas=img_metas,

        )

        return dict(init_points=init_points,
                    all_cls_scores=cls_scores,
                    all_refine_pts=refine_pts,
                    query_feat=query_feat,
                    )

    def get_dis_weight(self, pts):
        max_dist = torch.sqrt(
            self.scene_size[0] ** 2 + self.scene_size[1] ** 2)
        centers = (self.pc_range[:3] + self.pc_range[3:]) / 2
        dist = (pts - centers[None, ...])[..., :2]
        dist = torch.norm(dist, dim=-1)
        return dist / max_dist + 1

    def discretize(self, pts, clip=True, decode=False):
        loc = torch.floor((pts - self.pc_range[:3]) / self.voxel_size)
        if clip:
            loc[..., 0] = loc[..., 0].clamp(0, self.voxel_num[0] - 1)
            loc[..., 1] = loc[..., 1].clamp(0, self.voxel_num[1] - 1)
            loc[..., 2] = loc[..., 2].clamp(0, self.voxel_num[2] - 1)

        return loc.long() if not decode else \
            (loc + 0.5) * self.voxel_size + self.pc_range[:3]
    def reset_mask(self):
        init_points = self.init_points.weight.data
        ind_mask = init_points.new_zeros(init_points.shape[0], init_points.shape[0])
        for i in range(self.num_fu_frames):
            row_idx = (self.ind_stamps_all == i).nonzero(as_tuple=True)[0]
            col_idx = (self.ind_stamps_all > i).nonzero(as_tuple=True)[0]
            grid_row, grid_col = torch.meshgrid(row_idx, col_idx, indexing='ij')
            ind_mask[grid_row, grid_col] = -1e5
        for decoder_layer in self.transformer.decoder.decoder_layers:
            decoder_layer.self_attn.ind_mask = ind_mask
    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points, gt_labels):
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()

        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])

        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()
        gt_paired_pts = refine_pts[gt_paired_idx]
        pred_paired_pts = gt_points[pred_paired_idx]

        # cls assignment
        refine_pts_labels = gt_labels[pred_paired_idx]
        cls_weights = self.train_cfg.get('cls_weights', [1] * self.num_classes)
        cls_weights = refine_pts.new_tensor(cls_weights)
        label_weights = cls_weights * \
                        self.get_dis_weight(pred_paired_pts)[..., None]

        # gt side assignment
        empty_dist_thr = self.train_cfg.get('empty_dist_thr', 0.2)
        empty_weights = self.train_cfg.get('empty_weights', 3)

        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)
        mask = (dist > empty_dist_thr)
        gt_pts_weights[mask] = empty_weights
        if True:
            dis = torch.norm(pred_paired_pts - refine_pts,dim=-1,p=2,keepdim=True)
            label_weights = label_weights * torch.clamp(1/dis,max=1,min=0.4)
            gt_pts_weights[(gt_labels>=15) | (gt_labels==11)] *=0.5


        return (refine_pts_labels, gt_paired_idx, pred_paired_idx, label_weights,
                gt_pts_weights)
    def gather(self,tensor,sampled_inds):
        B,N = sampled_inds.shape[:2]
        batch_indx = torch.arange(tensor.shape[0],device=sampled_inds.device).unsqueeze(1).repeat(1,N)
        return tensor[batch_indx,sampled_inds]

    def get_targets(self):
        # To instantiate the abstract method
        pass

    def loss_single_mask(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_labels_list,
                    gt_stamps_list,
                    temporal_reweight=False):

        num_imgs,num_query,num_pts = cls_scores.shape[:3]  # B
        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)

        if temporal_reweight:
            for i in range(num_imgs):
                gt_points = gt_points_list[i]
                gt_stamps = gt_stamps_list[i]
                mask = gt_stamps[:,0]==1
                mask[gt_points[:,0]>40] = True
                mask[(gt_points[:,0]>35) * (gt_stamps[:,2]==1)] = True
                mask[(gt_points[:,0]>30) * (gt_stamps[:,1] == 1)] = True
                gt_points_list[i] = gt_points[mask]
                gt_stamps_list[i] = gt_stamps[mask]
                gt_labels_list[i] = gt_labels_list[i][mask]

        refine_pts = decode_points(refine_pts, self.pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights) = multi_apply(
            self._get_target_single, refine_pts_list, gt_points_list,
             gt_labels_list)

        gt_paired_pts, pred_paired_pts, gt_weights,pred_paired_stamps = [], [], [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])

            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])
            pred_paired_stamps.append(gt_stamps_list[i][pred_paired_idx_list[i]])

            fore_mask = (labels_list[i].reshape(num_query,-1,1)<2) | (labels_list[i].reshape(num_query,-1,1)>10)
            self.num_stamps_all += (gt_stamps_list[i][pred_paired_idx_list[i]].reshape(num_query,num_pts,self.num_fu_frames+1) * fore_mask).sum(1)

            # gt_pts_weights[i] = gt_pts_weights[i] * mask[gt_paired_idx_list[i]].squeeze(-1) * dist_mask[i]


        # concatenate all results from different samples
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)


        # cls_weights = cls_weights * mask
        # calculate loss cls
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 weight=cls_weights,
                                 avg_factor=cls_scores.shape[0])
        # calculate loss pts

        loss_pts = self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])

        if False:
            dis = torch.norm(pred_pts-pred_paired_pts,dim=-1,p=2,keepdim=True)
            pred_weights = torch.clamp(dis,max=1.5,min=0.4)

        loss_pts += self.loss_pts(pred_pts,
                                  pred_paired_pts,
                                  avg_factor=pred_pts.shape[0])


        # loss_stamp = F.binary_cross_entropy_with_logits(pred_stamps[moving_mask],
        #                              pred_paired_stamps[moving_mask].float(),
        #                              reduction='none',
        #                              ).sum() / (moving_mask.sum() * (self.num_fu_frames+1))
        return loss_cls, loss_pts



    def loss_single(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_labels_list):
        num_imgs = cls_scores.size(0)  # B
        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights) = multi_apply(
            self._get_target_single, refine_pts_list, gt_points_list,
             gt_labels_list)

        gt_paired_pts, pred_paired_pts = [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # concatenate all results from different samples
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)


        # calculate loss cls
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 weight=cls_weights,
                                 avg_factor=cls_scores.shape[0])
        # calculate loss pts
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts,
                                  pred_paired_pts,
                                  avg_factor=pred_pts.shape[0])

        return loss_cls, loss_pts

    def loss_single_rangemask(self,
                    cls_scores,
                    refine_pts,
                    points_mask,
                    gt_points_list,
                    gt_labels_list):
        num_imgs = cls_scores.size(0)  # B

        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        if points_mask is None:
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            refine_pts_list = [refine_pts[i] for i in range(num_imgs)]
        else:
            points_mask = points_mask.reshape(num_imgs, -1)
            points_mask = points_mask.reshape(-1,1)
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            refine_pts_list = [refine_pts[i] for i in range(num_imgs)]
        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights) = multi_apply(
            self._get_target_single, refine_pts_list, gt_points_list,
             gt_labels_list)

        gt_paired_pts, pred_paired_pts = [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # concatenate all results from different samples
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

        # calculate loss cls
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 weight=cls_weights *points_mask,
                                 avg_factor=cls_scores.shape[0])
        # calculate loss pts
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts,
                                  pred_paired_pts,
                                  weight=points_mask,
                                  avg_factor=pred_pts.shape[0])

        return loss_cls, loss_pts



    def loss_future(self, voxel_semantics, all_refine_pts, all_cls_scores,points_mask):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17

        gt_points_list, gt_labels_list = [],  []
        for voxel_sem in voxel_semantics:
            gt_points,  gt_labels = \
                self.get_sparse_voxels(voxel_sem)
            gt_points_list.append(gt_points)

            gt_labels_list.append(gt_labels)

        # all_cls_scores = torch.unbind(all_cls_scores,0)
        all_gt_points_list = gt_points_list
        all_gt_labels_list = gt_labels_list

        losses_cls, losses_pts = multi_apply(
            self.loss_single_rangemask, all_cls_scores, all_refine_pts,points_mask,
            all_gt_points_list, all_gt_labels_list)

        loss_dict = dict()
        # loss of init_points

        # loss from other decoder layers
        time_stamp = 1
        for loss_cls_i, loss_pts_i in zip(losses_cls, losses_pts):
            loss_dict[f'fu{time_stamp}.loss_cls'] = loss_cls_i * (not self.pretrain)
            loss_dict[f'fu{time_stamp}.loss_pts'] = loss_pts_i * (not self.pretrain)
            time_stamp += 1
        return loss_dict

    def loss_pretrain(self,voxel_semantic,temporal_semantics,temporal2ego, pred_dicts):
        B = voxel_semantic.shape[0]
        loss = dict()

        init_points_all = pred_dicts['init_points']

        pred_dict = {'init_points':init_points_all,
                     'all_cls_scores': list(),
                     'all_refine_pts': list(),
                     'all_pred_stamps': list()}
        num_refine = len(pred_dicts['all_cls_scores']) if self.pretrain else 1
        for cls_scores, refine_pts in zip(pred_dicts['all_cls_scores'][:num_refine], pred_dicts['all_refine_pts'][:num_refine]):
            pred_dict['all_cls_scores'].append(cls_scores)
            pred_dict['all_refine_pts'].append(refine_pts)

        voxel_semantic_stack,voxel_stamp_stack = self.get_sparse_voxels_stack(voxel_semantic,temporal_semantics,temporal2ego)


        loss.update(self.loss_stack(voxel_semantic_stack,pred_dict,time_stamp = 0,voxel_stamp=voxel_stamp_stack))
        
        return loss


    def loss_stack(self, voxel_semantics, preds_dicts, time_stamp=0,temporal2ego=None,voxel_stamp=None):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores']  # 6 ,B,2k4,32,17
        all_refine_pts = preds_dicts['all_refine_pts']

        num_dec_layers = len(all_cls_scores)

        gt_points_list, gt_labels_list, gt_stamps_list = \
            self.get_sparse_voxels_new(voxel_semantics,voxel_stamp)
        # gt_points_list,  gt_temporal_list = \
        #     self.get_sparse_voxels_new(voxel_stamp)
        if temporal2ego is not None:
            B = len(gt_points_list)
            gt_points_list = [torch.matmul(gt_points_list[b],temporal2ego[b,:3,:3].transpose(1,0)) + temporal2ego[b,None:3,3] for b in range(B)]

        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        # all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_stamps_list = [gt_stamps_list for _ in range(num_dec_layers)]

        if len(all_cls_scores)>0:
            losses_cls, losses_pts = multi_apply(
            self.loss_single_mask, all_cls_scores, all_refine_pts,
            all_gt_points_list, all_gt_labels_list,all_gt_stamps_list,temporal_reweight = True)
        else:
            losses_cls, losses_pts = [],[]
        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)

            _, init_loss_pts = self.loss_single_mask(
                pseudo_scores, init_points, gt_points_list,
                 gt_labels_list,gt_stamps_list,temporal_reweight=True)
            loss_dict[f'init_loss_pts'] = init_loss_pts

        # loss from the last decoder layer

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls, losses_pts):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            num_dec_layer += 1
        return loss_dict
    def loss(self, voxel_semantics, preds_dicts):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores'] # 6 ,B,2k4,32,17
        all_refine_pts = preds_dicts['all_refine_pts']

        num_dec_layers = len(all_cls_scores)
        gt_points_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        # all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_pts = multi_apply(
            self.loss_single, all_cls_scores, all_refine_pts,
            all_gt_points_list,  all_gt_labels_list)

        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)
            _, init_loss_pts = self.loss_single(
                pseudo_scores, init_points, gt_points_list,
                 gt_labels_list)
            loss_dict['init_loss_pts'] = init_loss_pts

        # loss from the last decoder layer


        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls, losses_pts):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            num_dec_layer += 1
        return loss_dict

    def get_occ(self, pred_dicts, expand_range = False,thre1 = 0.1,thre2=0.1):
        if expand_range:
            pc_range = self.pc_range_new
            voxel_num = self.voxel_num_new
        else:
            pc_range = self.pc_range
            voxel_num = self.voxel_num

        cls_scores = pred_dicts['cls_scores'].sigmoid()
        # cls_scores = all_cls_scores[-1].sigmoid()
        refine_pts = pred_dicts['refine_pts']
        if thre1!=None:
            mask = cls_scores.argmax(-1)==15
            dis = torch.norm(refine_pts - torch.mean(refine_pts,dim=2,keepdim=True),dim=-1)
            cls_scores[mask] = cls_scores[mask] * torch.clamp(thre1 /dis[mask],max=1)[:,None]
        if thre2!=None:
            mask = cls_scores.argmax(-1) == 16
            dis = torch.norm(refine_pts - torch.mean(refine_pts, dim=2, keepdim=True), dim=-1)
            cls_scores[mask] = cls_scores[mask] * torch.clamp(thre2 / dis[mask], max=1)[:, None]

        batch_size = refine_pts.shape[0]
        ctr_dist_thr = self.test_cfg.get('ctr_dist_thr', 3.0)
        score_thr = self.test_cfg.get('score_thr', 0.1)
        score_thr = torch.tensor(score_thr,device=cls_scores.device)
        result_list = []
        for i in range(batch_size):
            refine_pts, cls_scores = refine_pts[i], cls_scores[i]
            refine_pts = decode_points(refine_pts, self.pc_range)

            # filter weak points by distance and score
            centers = refine_pts.mean(dim=1, keepdim=True)
            ctr_dists = torch.norm(refine_pts - centers, dim=-1)
            mask_dist = ctr_dists < ctr_dist_thr
            max_score,index = cls_scores.max(-1)
            mask_score = (cls_scores.max(-1)[0] > score_thr[index])
            mask = mask_dist & mask_score
            refine_pts = refine_pts[mask]
            cls_scores = cls_scores[mask]
            if False:
                pts = torch.cat([refine_pts, cls_scores], dim=-1)
                pts_infos, voxels, num_pts = self.voxel_generator(pts)
                voxels = torch.flip(voxels, [1]).long()
                pts, scores = pts_infos[..., :3], pts_infos[..., 3:]
                scores = scores.sum(dim=1) / num_pts[..., None]
            else:
                index = ((refine_pts - pc_range[:3]) // self.voxel_size).long()
                mask = torch.logical_and(index >= 0, index < voxel_num).all(-1)
                voxels, unq_inv, pts_num = torch.unique(index[mask], return_inverse=True, return_counts=True, dim=0)
                scores = torch_scatter.scatter_max(cls_scores[mask], unq_inv, 0)[0]
            occ = scores.new_zeros((voxel_num[0], voxel_num[1],
                                    voxel_num[2], self.num_classes))
            occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = scores
            occ = occ.permute(3, 0, 1, 2).unsqueeze(0)
            if self.test_cfg.get('padding', True):
                # padding
                dilated_occ = F.max_pool3d(occ, 3, stride=1, padding=1)
                eroded_occ = -F.max_pool3d(-dilated_occ, 3, stride=1, padding=1)

                max_score,index = occ.max(1)
                original_mask = (max_score > score_thr[index]) #| (eroded_occ.argmax(1)==15 )
                original_mask = original_mask.expand_as(eroded_occ)
                eroded_occ[original_mask] = occ[original_mask]
            else:
                eroded_occ = occ
                # eroded_occ = occ[0].permute(1,2,3,0)
            eroded_occ = eroded_occ.squeeze(0).permute(1, 2, 3, 0)
            voxels = torch.nonzero((eroded_occ > score_thr).any(dim=-1))
            scores = eroded_occ[voxels[:, 0], voxels[:, 1], voxels[:, 2], :]
            occ_pred = torch.ones(voxel_num.tolist(), device=eroded_occ.device, dtype=torch.long) * 17
            occ_pred[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = eroded_occ[voxels[:, 0], voxels[:, 1],
                                                                     voxels[:, 2], :].argmax(-1)
            result_list.append(occ_pred)

        return torch.stack(result_list, 0)

    @torch.no_grad()
    def get_sparse_voxels_new(self, voxel_semantics,voxel_stamp):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size_new[0] + self.pc_range_new[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size_new[1] + self.pc_range_new[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size_new[2] + self.pc_range_new[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, H, Z)
        coors = torch.stack([xx, yy, zz], dim=-1)  # actual space

        gt_points, gt_masks, gt_labels, gt_stamps = [], [], [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.empty_label
            gt_points.append(coors[mask])
              # camera mask and not empty
            gt_labels.append(voxel_semantics[i][mask])
            gt_stamps.append(voxel_stamp[i][mask])

        return gt_points,  gt_labels, gt_stamps

    @torch.no_grad()
    def get_sparse_voxels(self, voxel_semantics):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, H, Z)
        coors = torch.stack([xx, yy, zz], dim=-1)  # actual space

        gt_points, gt_labels = [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.empty_label
            gt_points.append(coors[mask])
             # camera mask and not empty
            gt_labels.append(voxel_semantics[i][mask])

        return gt_points, gt_labels

    @torch.no_grad()
    def get_sparse_voxels_stack(self,voxel_semantic,temporal_semantics,temporal_pre2curs,mask_moving = True):
        B, W, H, Z = voxel_semantic.shape
        stack_semantics = voxel_semantic.new_ones(B,400,300,24) * 17
        stack_stamp = voxel_semantic.new_zeros(B,400,300,24,self.num_fu_frames+1)
        device = voxel_semantic.device
        new_voxel_num = self.voxel_num_new
        pc_range = self.pc_range_new
        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, W, Z)
        coors = torch.stack([xx, yy, zz], dim=-1)

        for b in range(B):
            for t in [0,6,5,4,3,2,1]:
                if t>0:
                    cur_semantic,fu2cur = (temporal_semantics[t]['voxel_semantics'][b],
                                       temporal_pre2curs[t-1][b])
                else:
                    cur_semantic,fu2cur = voxel_semantic[b],torch.eye(4,device=device)
                cur_points = coors[cur_semantic!=17]
                cur_semantic = cur_semantic[cur_semantic!=17]
                cur_points = torch.matmul(cur_points,fu2cur[:3,:3].transpose(0,1)) + fu2cur[:3,3]
                warped_coords = torch.floor((cur_points - pc_range[None,:3])//self.voxel_size[None,:]).long()
                mask = torch.logical_and(warped_coords>=0,warped_coords<new_voxel_num[None,:]).all(-1)

                stack_stamp[b, warped_coords[mask, 0], warped_coords[mask, 1], warped_coords[mask, 2], t] = 1
                # if mask_moving and t>0:
                #     mask = mask * torch.logical_or(cur_semantic<2,cur_semantic>10)

                stack_semantics[b,warped_coords[mask,0],warped_coords[mask,1],warped_coords[mask,2]] = cur_semantic[mask]

        return stack_semantics,stack_stamp
