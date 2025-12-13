import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import bias_init_with_prob
from mmcv.cnn.bricks.transformer import  FFN
from mmdet.models.utils.builder import TRANSFORMER
from mmdet3d.models.sparsedetectors.bbox.utils import decode_points, encode_points
from .utils import DUMP
from .opus_sampling import sampling_4d
from .checkpoint import checkpoint as cp
from mmdet3d.models.sparsedetectors.csrc.wrapper import MSMV_CUDA
from mmdet3d.models.sparsedetectors.bbox.utils import encode_points,decode_points,MultiheadAttention
import time

@TRANSFORMER.register_module()
class OPUSTransformer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=[1, 2, 4, 8, 16, 32],
                 scales=[1.0],
                 pc_range=[],
                 init_cfg=None,
                 num_query=600,
                 num_fu_query=100):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_refines = num_refines

        self.decoder = OPUSTransformerDecoder(
            embed_dims, num_frames, num_views, num_points, num_layers, num_levels,
            num_classes, num_refines, num_groups, scales, pc_range=pc_range,num_query=num_query,num_fu_query=num_fu_query)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, img_metas):
        query_feat, cls_scores, refine_pts  = self.decoder(
            query_points, query_feat, mlvl_feats, img_metas)

        cls_scores = [torch.nan_to_num(score) for score in cls_scores]
        refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]

        return query_feat, cls_scores, refine_pts


class OPUSTransformerDecoder(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_refines=16,
                 num_groups=4,
                 scales=[1.0],
                 pc_range=[],
                 init_cfg=None,
                 num_query=600,
                 num_fu_query=100):
        super().__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_groups = num_groups

        if len(scales) == 1:
            scales = scales * num_layers
        if not isinstance(num_refines, list):
            num_refines = [num_refines]
        if len(num_refines) == 1:
            num_refines = num_refines * num_layers
        last_refines = [1] + num_refines

        # params are shared across all decoder layers
        self.decoder_layers = ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                OPUSTransformerDecoderLayer(
                    embed_dims, num_frames, num_views, num_points, num_levels, num_classes, 
                    num_groups, num_refines[i], last_refines[i], layer_idx=i, 
                    scale=scales[i], pc_range=pc_range,num_query=num_query,num_fu_query=num_fu_query)
            )

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layers.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, img_metas):
        cls_scores, refine_pts, pred_stamps = [], [], []


        # organize projections matrix and copy to CUDA
        lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
        lidar2img = torch.tensor(lidar2img,device=query_points.device) # [B, N, 4, 4]
        ego2lidar = np.asarray([m['ego2lidar'] for m in img_metas]).astype(np.float32)
        ego2lidar = query_feat.new_tensor(ego2lidar) # [B, 4, 4]
        ego2lidar = ego2lidar.unsqueeze(1).expand_as(lidar2img)  # [B, N, 4, 4]
        occ2img = torch.matmul(lidar2img, ego2lidar)
        # occ2img = torch.rand((2,30,4,4),device=query_points.device)
        # if img_metas[0]['sample_idx'].startswith('4dfab'):
        #     print('e93e98')

        mlvl_feats_reshaped = []
        # group image features in advance for sampling, see `sampling_4d` for more details
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N, T, G, C = self.num_views, self.num_frames, self.num_groups, GC//self.num_groups
            assert T*N == TN
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:  # Our CUDA operator requires channel_last
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
                feat = feat.reshape(B*T*G, N, H, W, C)
            else:  # Torch's grid_sample requires channel_first
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, G, C, N, H, W]
                feat = feat.reshape(B*T*G, C, N, H, W)

            mlvl_feats_reshaped.append(feat.contiguous())
        for i, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i

            query_points = query_points.detach()
            query_feat, cls_score, query_points = decoder_layer(
                query_points, query_feat, mlvl_feats_reshaped, occ2img, img_metas)

            cls_scores.append(cls_score)
            refine_pts.append(query_points)

        return query_feat, cls_scores, refine_pts,


class OPUSTransformerDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=16,
                 last_refines=16,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 layer_idx=0,
                 scale=1.0,
                 pc_range=[],
                 init_cfg=None,
                 num_query=600,
                 num_fu_query=100,
                 ):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.num_points = num_points
        self.num_refines = num_refines
        self.last_refines = last_refines
        self.layer_idx = layer_idx
        self.scale = scale

        self.position_encoder = nn.Sequential(
            nn.Linear(3 * self.last_refines, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = OPUSSelfAttention(
            embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range,layer_idx=layer_idx,num_query=num_query,num_fu_query=num_fu_query)

        self.sampling = OPUSSampling(embed_dims, num_frames=num_frames, num_views=num_views,
                                     num_groups=num_groups, num_points=num_points, 
                                     num_levels=num_levels, pc_range=pc_range)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames,
                                     n_groups=num_groups, out_points=32)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))

        cls_branch.append(nn.Linear(
            self.embed_dims, self.num_classes * self.num_refines))
        self.cls_branch = nn.Sequential(*cls_branch)
        reg_branch = []

        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))

        reg_branch.append(nn.Linear(self.embed_dims, 3 * self.num_refines))
        self.reg_branch = nn.Sequential(*reg_branch)


    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def refine_points(self, points_proposal, points_delta):
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def forward(self, query_points, query_feat, mlvl_feats, occ2img, img_metas):
        """
        query_points: [B, Q, 3] [x, y, z]
        """
        query_pos = self.position_encoder(query_points.flatten(2, 3))
        query_feat = query_feat + query_pos

        sampled_feat = self.sampling(
            query_points, query_feat, mlvl_feats, occ2img, img_metas)
        query_feat = self.norm1(self.mixing(sampled_feat, query_feat))
        query_feat,_ = self.self_attn(query_points, query_feat)

        query_feat = self.norm2(query_feat)
        query_feat = self.norm3(self.ffn(query_feat))

        B, Q = query_points.shape[:2]
        cls_score = self.cls_branch(query_feat)  # [B, Q, P * num_classes]
        reg_offset = self.scale * self.reg_branch(query_feat)  # [B, Q, P * 3]

        cls_score = cls_score.view(B, Q, self.num_refines, self.num_classes)
        refine_pt = self.refine_points(query_points, reg_offset)

        if DUMP.enabled:
            pass # TODO: enable OTR dump

        return query_feat, cls_score, refine_pt,

class OPUSCrossAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)
        self.pos_embed = nn.Linear(3,embed_dims)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_points, query_feat, key_points, key_feat):
        """
        query_points: [B, Q, 6]
        query_feat: [B, Q, C]
        """
        dist = self.calc_points_dists(query_points,key_points)
        dist = torch.exp(dist/40)
        tau = self.gen_tau(key_feat)  # [B, Q, 8]
        tau = F.softplus(tau)
        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]

        attn_mask = dist[:, None, :, :] * tau[:,:,None,:]  # [B, 8, Q, Q]

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        query_pos = self.pos_embed(query_points*0)
        key_pos = self.pos_embed(decode_points(key_points,self.pc_range).mean(2))

        # query_pos = None
        # key_pos = None
        # return self.attention(query_feat, key_feat, attn_mask=attn_mask)
        return self.attention(query_feat,key_feat, attn_mask=attn_mask,query_pos = query_pos,key_pos = key_pos)

    def forward(self, query_points, query_feat, key_points, key_feat):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat,key_points, key_feat,
                      use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat, key_points, key_feat)

    @torch.no_grad()
    def calc_points_dists(self, points1,points2):
        points1 = decode_points(points1, self.pc_range)
        points2 = decode_points(points2, self.pc_range)
        points2 = points2.mean(dim=2)
        dist = torch.norm(points1.unsqueeze(-2) - points2.unsqueeze(-3), dim=-1)
        return -dist



class OPUSSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self, 
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 pc_range=[],
                 init_cfg=None,
                 layer_idx=0,
                 num_query=600,
                 num_fu_query=100,):
        super().__init__(init_cfg)
        self.pc_range = pc_range
        self.layer_idx = layer_idx
        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)
        self.num_query = num_query
        self.num_heads = num_heads
        self.num_fu_query = num_fu_query
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_points, query_feat):
        """
        query_points: [B, Q, 6]
        query_feat: [B, Q, C]
        """
        B = query_feat.shape[0]
        dist = self.calc_points_dists(query_points)
        tau = self.gen_tau(query_feat)  # [B, Q, 8]
        # tau = F.softplus(tau)
        # tau = torch.clamp(tau,min=0)
        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]
        if getattr(self,'ind_mask',None) is not None:
            attn_mask = attn_mask + self.ind_mask[None,None,:,:]
        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        return self.attention(query_feat,query_feat,query_feat, attn_mask=attn_mask)

    def forward(self, query_points, query_feat):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat,
                      use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat)

    @torch.no_grad()
    def calc_points_dists(self, points):
        points = decode_points(points, self.pc_range)
        points = points.mean(dim=2)
        dist = torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)
        return -dist


class OPUSSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self,
                 embed_dims=256,
                 num_frames=4,
                 num_views=6,
                 num_groups=4,
                 num_points=8,
                 num_levels=4,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_views = num_views
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_points, query_feat, mlvl_feats, occ2img, img_metas):
        '''
        query_points: [B, Q, 6]
        query_feat: [B, Q, C]
        '''
        B, Q = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # query points
        query_points = decode_points(query_points, self.pc_range)
        if query_points.shape[2] == 1:
            query_center = query_points
            query_scale = torch.zeros_like(query_center)
        else:
            query_center = query_points.mean(dim=2, keepdim=True)
            query_scale = query_points.std(dim=2, keepdim=True)

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, -1, 3)

        sampling_points = query_center + sampling_offset * query_scale
        sampling_points = sampling_points.view(B, Q, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            occ2img,
            image_h, image_w,
            self.num_views
        )  # [B, Q, G, FP, C]

        return sampled_feats

    def forward(self, query_points, query_feat, mlvl_feats, occ2img, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat, mlvl_feats,
                      occ2img, img_metas, use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat, mlvl_feats,
                                      occ2img, img_metas)


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super().__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)

class DropMultiheadAttention(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=0.,
                 init_cfg=None,
                 batch_first=False,
                 num_out_query=None,
                 **kwargs):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.num_out_query = num_out_query
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = nn.Dropout(dropout_layer)

    def forward(self,
                query,
                key=None,
                value=None,

                attn_mask=None,
                key_padding_mask=None,
                **kwargs):

        identity = query
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out,weight = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)
        if self.num_out_query is None:
            num_out_query = weight.shape[-1]
        else:
            num_out_query = self.num_out_query

        sampled_inds = torch.topk(weight.sum(1),num_out_query,-1)[1]
        # sampled_inds = torch.randperm(weight.shape[1]).cuda()[None,:num_out_query]
        if self.batch_first:
            out = out.transpose(0, 1)
        out = torch.gather(out, 1,sampled_inds[:,:,None].repeat(1,1,self.embed_dims))
        return torch.gather(identity,1,sampled_inds[:,:,None].repeat(1,1,self.embed_dims)) + self.dropout_layer(self.proj_drop(out)), sampled_inds






class DropOPUSSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 num_points = 128,
                 pc_range=[],
                 init_cfg=None,
                 num_out_query = None,
                 freeze = False,
                 **kwargs):
        super().__init__(init_cfg)
        self.pc_range = pc_range
        self.num_points = num_points
        self.position_encoder = nn.Sequential(
            nn.Linear(3 * self.num_points, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

        self.attention = DropMultiheadAttention(embed_dims, num_heads, dropout, batch_first=True,num_out_query = num_out_query,**kwargs)
        self.gen_tau = nn.Linear(embed_dims, num_heads)
        self.ffn = FFN(embed_dims,feedforward_channels=512,ffn_drop=0.1)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_points, query_feat):
        """
        query_points: [B, Q, 6]
        query_feat: [B, Q, C]
        """

        query_pos = self.position_encoder(query_points.flatten(2,3))
        query_feat = query_feat + query_pos

        dist = self.calc_points_dists(query_points)
        tau = self.gen_tau(query_feat)  # [B, Q, 8]

        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        query_feat,sampled_inds = self.attention(query_feat,query_feat,query_feat, attn_mask=attn_mask)
        query_points = torch.gather(query_points,1,sampled_inds[:,:,None,None].repeat(1,1,query_points.shape[-2],3))
        query_feat = self.norm1(query_feat)
        query_feat = self.norm2(self.ffn(query_feat) + query_feat)
        return query_feat, query_points
    def forward(self, query_points, query_feat):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat,
                      use_reentrant=False)
        else:
            return self.inner_forward(query_points, query_feat)

    @torch.no_grad()
    def calc_points_dists(self, points):
        points = decode_points(points, self.pc_range)
        points = points.mean(dim=2)
        dist = torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)
        return -dist

