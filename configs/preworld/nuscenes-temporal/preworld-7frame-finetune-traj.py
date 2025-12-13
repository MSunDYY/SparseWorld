_base_ = ['./bevstereo-occ-traj.py']

use_checkpoint = False
occ_size = [200, 200, 16]
voxel_out_channel = 32
empty_idx = 17
num_cls = 18
point_cloud_range = [-40., -40., -1., 40., 40., 5.4]

model = dict(
    type='PreWorld4DTraj',
    final_softplus=True,
    use_lss_depth_loss=False,
    use_3d_loss=False,
    if_render=False,
    if_post_finetune=True,
    weight_voxel_ce=1.0,
    weight_voxel_sem_scal=1.0,
    weight_voxel_geo_scal=1.0,
    weight_voxel_lovasz=1.0,
    empty_idx=empty_idx,
    nerf_head=dict(
        type='NerfHead',
        point_cloud_range=[-40., -40., -1., 40., 40., 5.4],
        voxel_size=0.4,
        scene_center=[0, 0, 2.2],
        radius=39,
        use_depth_sup=False,
        weight_depth=0.0,
        weight_semantic=0.0,
        weight_color=0.0,
        weight_entropy_last=0.01,   
        weight_distortion=0.01,
    ),
    occupancy_head=dict(
        type='OccHead',
        with_cp=use_checkpoint,
        use_deblock=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        final_occ_size=occ_size,
        empty_idx=empty_idx,
        num_level=1,
        in_channels=[voxel_out_channel],
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
    )
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)



depth_gt_path = './data/depth_gt'
semantic_gt_path = './data/seg_gt_lidarseg'

data = dict(
    samples_per_gpu=1,  # with 8 GPU, Batch Size=16
    workers_per_gpu=2,
    train=dict(
        use_rays=False,
        if_dense=False,
        depth_gt_path=depth_gt_path,
        semantic_gt_path=semantic_gt_path,
        aux_frames=[-3,-2,-1,1,2,3],
        max_ray_nums=19200,
    )
)


runner = dict(type='EpochBasedRunner', max_epochs=18)

# custom hooks
custom_hooks = [dict(type='CustomSetEpochInfoHook')]

log_config = dict(
    interval=10,
)

# find_unused_parameters = True