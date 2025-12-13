import numpy as np
import torch
import torch.nn as nn
import warnings
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
def normalize_bbox(bboxes):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]

    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        out = torch.cat([cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy], dim=-1)
    else:
        out = torch.cat([cx, cy, w, l, cz, h, rot.sin(), rot.cos()], dim=-1)

    return out


def denormalize_bbox(normalized_bboxes):
    rot_sin = normalized_bboxes[..., 6:7]
    rot_cos = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sin, rot_cos)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    w = normalized_bboxes[..., 2:3].exp()
    l = normalized_bboxes[..., 3:4].exp()
    h = normalized_bboxes[..., 5:6].exp()

    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        out = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        out = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)

    return out


def encode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].log()
    rot = bboxes[..., 6:7]

    if pc_range is not None:
        xyz[..., 0] = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        xyz[..., 1] = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        xyz[..., 2] = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])

    if bboxes.shape[-1] > 7:
        vel = bboxes[..., 7:9].clone()
        return torch.cat([xyz, wlh, rot.sin(), rot.cos(), vel], dim=-1)
    else:
        return torch.cat([xyz, wlh, rot.sin(), rot.cos()], dim=-1)


def decode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].exp()
    rot = torch.atan2(bboxes[..., 6:7], bboxes[..., 7:8])

    if pc_range is not None:
        xyz[..., 0] = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        xyz[..., 1] = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        xyz[..., 2] = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    if bboxes.shape[-1] > 8:
        vel = bboxes[..., 8:10].clone()
        return torch.cat([xyz, wlh, rot, vel], dim=-1)
    else:
        return torch.cat([xyz, wlh, rot], dim=-1)


def dist_loss_weight(points,threshold=45, sigma=10):
    weight = torch.ones_like(points[:,0])
    abs_points = torch.abs(points).max(-1)[0]
    mask = abs_points>threshold
    weight[mask] = torch.exp(-(torch.abs(abs_points[mask] - threshold)) / sigma)
    return weight

def encode_points(points, pc_range=None):
    points = points.clone()
    points[..., 0] = (points[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    points[..., 1] = (points[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    points[..., 2] = (points[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
    return points


def reweight(maps):
    max_map,max_index = maps[:,:-1].max(-1)
    sort_index = torch.sort(max_map,descending=True)[1]
    maps = maps[sort_index]
    return maps
def get_matched_inds(maps,num_inds):
    assert maps.shape[0] == sum(num_inds)
    num_points = maps.shape[0]
    inds = torch.ones(maps.shape[0],device=maps.device) * (-1)
    maps = torch.concat([maps,torch.arange(num_points,device=maps.device).unsqueeze(-1)],dim=-1)
    maps = reweight(maps)
    i=0
    while i<num_points:
        ind = torch.argmax(maps[i,:-1],-1).item()
        inds[int(maps[i,-1])] = ind
        # print(ind)
        if (inds == ind).sum().item() == num_inds[ind]:
            maps[i+1:,ind] = 0
            maps[i+1:] = reweight(maps[i+1:])
        i+=1
    return inds
def decode_points(points, pc_range=None):
    points = points.clone()
    points[..., 0] = points[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    points[..., 1] = points[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    points[..., 2] = points[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return points

def trans_coords(ego2global,temporal_ego2global):
    temporal2ego = dict()
    for key,val in temporal_ego2global.items():
        temporal2ego[key] = torch.matmul(torch.linalg.inv(ego2global),val).float()
    return temporal2ego


class MultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super().__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = nn.Dropout(0)
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out,weight = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,)

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out)),weight