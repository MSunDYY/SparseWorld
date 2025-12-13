point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDatasetOccpancy4DTraj'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=4),
    dict(type='LoadOccGTFromFile4DTraj'),
    dict(
        type='RandomTransformImage',
        ida_aug_conf=dict(
            resize_lim=(0.38, 0.55),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect4D',
        keys=[
            'img', 'voxel_semantics', 'mask_lidar', 'mask_camera', 'rays',
            'temporal_semantics', 'temporal_rays', 'temporal_ego_states',
            'temporal_trajs', 'temporal_ego2global'
        ],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'lidar2img', 'img_timestamp', 'ego2lidar', 'ego2global',
                   'sample_idx'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='color'),
    dict(
        type='LoadMultiViewImageFromMultiSweeps',
        sweeps_num=4,
        test_mode=False),
    dict(type='LoadOccGTFromFile4DTraj'),
    dict(
        type='RandomTransformImage',
        ida_aug_conf=dict(
            resize_lim=(0.38, 0.55),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect4D',
                keys=[
                    'img', 'voxel_semantics', 'mask_lidar', 'mask_camera',
                    'temporal_semantics', 'temporal_ego_states',
                    'temporal_trajs', 'temporal_agent_boxes',
                    'temporal_agent_feats'
                ],
                meta_keys=[
                    'filename', 'box_type_3d', 'ori_shape', 'img_shape',
                    'pad_shape', 'sample_idx', 'lidar2img', 'img_timestamp',
                    'ego2lidar', 'gt_boxes', 'gt_labels', 'occ_gt_path'
                ])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type='NuScenesDatasetOccpancy4DTraj',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_train.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles',
                to_float32=False,
                color_type='color'),
            dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=4),
            dict(type='LoadOccGTFromFile4DTraj'),
            dict(
                type='RandomTransformImage',
                ida_aug_conf=dict(
                    resize_lim=(0.38, 0.55),
                    final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=True),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='Collect4D',
                keys=[
                    'img', 'voxel_semantics', 'mask_lidar', 'mask_camera',
                    'rays', 'temporal_semantics', 'temporal_rays',
                    'temporal_ego_states', 'temporal_trajs',
                    'temporal_ego2global'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'lidar2img', 'img_timestamp', 'ego2lidar',
                           'ego2global', 'sample_idx'))
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=True,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(0, 5, 1),
        load_interval=1),
    val=dict(
        type='NuScenesDatasetOccpancy4DTraj',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles',
                to_float32=False,
                color_type='color'),
            dict(
                type='LoadMultiViewImageFromMultiSweeps',
                sweeps_num=4,
                test_mode=False),
            dict(type='LoadOccGTFromFile4DTraj'),
            dict(
                type='RandomTransformImage',
                ida_aug_conf=dict(
                    resize_lim=(0.38, 0.55),
                    final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect4D',
                        keys=[
                            'img', 'voxel_semantics', 'mask_lidar',
                            'mask_camera', 'temporal_semantics',
                            'temporal_ego_states', 'temporal_trajs',
                            'temporal_agent_boxes', 'temporal_agent_feats'
                        ],
                        meta_keys=[
                            'filename', 'box_type_3d', 'ori_shape',
                            'img_shape', 'pad_shape', 'sample_idx',
                            'lidar2img', 'img_timestamp', 'ego2lidar',
                            'gt_boxes', 'gt_labels', 'occ_gt_path'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(0, 5, 1)),
    test=dict(
        type='NuScenesDatasetOccpancy4DTraj',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles',
                to_float32=False,
                color_type='color'),
            dict(
                type='LoadMultiViewImageFromMultiSweeps',
                sweeps_num=4,
                test_mode=False),
            dict(type='LoadOccGTFromFile4DTraj'),
            dict(
                type='RandomTransformImage',
                ida_aug_conf=dict(
                    resize_lim=(0.38, 0.55),
                    final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect4D',
                        keys=[
                            'img', 'voxel_semantics', 'mask_lidar',
                            'mask_camera', 'temporal_semantics',
                            'temporal_ego_states', 'temporal_trajs',
                            'temporal_agent_boxes', 'temporal_agent_feats'
                        ],
                        meta_keys=[
                            'filename', 'box_type_3d', 'ori_shape',
                            'img_shape', 'pad_shape', 'sample_idx',
                            'lidar2img', 'img_timestamp', 'ego2lidar',
                            'gt_boxes', 'gt_labels', 'occ_gt_path'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(0, 5, 1)))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = ''
load_from = 'ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
resume_from = 'work_dirs/sparseworld-7frame-finetune21/epoch_17.pth'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT'
    ],
    Ncams=6,
    input_size=(512, 1408),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
grid_config = dict(
    x=[-40, 40, 0.4],
    y=[-40, 40, 0.4],
    z=[-1, 5.4, 0.4],
    depth=[1.0, 45.0, 0.5])
voxel_size = [0.4, 0.4, 0.4]
numC_Trans = 32
use_checkpoint = False
occ_size = [200, 200, 16]
voxel_out_channel = 32
empty_idx = 17
num_cls = 18
depth_gt_path = './data/depth_gt'
semantic_gt_path = './data/seg_gt_lidarseg'
object_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
occ_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]
embed_dims = 256
num_layers = 6
num_query = 720
num_fu_query = [120, 110, 80, 70, 50, 40]
num_frames = 5
num_fu_frames = 6
num_levels = 4
num_points = 4
num_refines = [1, 4, 16, 32, 64, 128]
multi_adj_frame_id_cfg = (0, 5, 1)
pretrain = False
finetune_epoch = 5
img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=True)
img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=4)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
model = dict(
    type='SparseWorld4DTraj',
    final_softplus=True,
    out_dim=256,
    use_grid_mask=False,
    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=0,
    is_pretrain=False,
    finetune_epoch=5,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=True),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    pts_bbox_head=dict(
        type='OPUSHead',
        num_classes=17,
        in_channels=256,
        num_query=720,
        num_fu_frames=6,
        num_fu_query=[120, 110, 80, 70, 50, 40],
        pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        voxel_size=[0.4, 0.4, 0.4],
        is_pretrain=False,
        transformer=dict(
            type='OPUSTransformer',
            embed_dims=256,
            num_frames=5,
            num_points=4,
            num_layers=6,
            num_levels=4,
            num_classes=17,
            num_refines=[1, 4, 16, 32, 64, 128],
            scales=[0.5],
            pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
            num_query=720,
            num_fu_query=[120, 110, 80, 70, 50, 40]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_pts=dict(type='SmoothL1Loss', beta=0.2, loss_weight=0.5)),
    train_cfg=dict(
        pts=dict(cls_weights=[
            10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1
        ])),
    test_cfg=dict(pts=dict(score_thr=0.4, padding=True)))
bda_aug_conf = dict(
    rot_lim=(-0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
ida_aug_conf = dict(
    resize_lim=(0.38, 0.55),
    final_dim=(256, 704),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True)
share_data_config = dict(
    type='NuScenesDatasetOccpancy4DTraj',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=(0, 5, 1))
test_data_config = dict(
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFiles',
            to_float32=False,
            color_type='color'),
        dict(
            type='LoadMultiViewImageFromMultiSweeps',
            sweeps_num=4,
            test_mode=False),
        dict(type='LoadOccGTFromFile4DTraj'),
        dict(
            type='RandomTransformImage',
            ida_aug_conf=dict(
                resize_lim=(0.38, 0.55),
                final_dim=(256, 704),
                bot_pct_lim=(0.0, 0.0),
                rot_lim=(0.0, 0.0),
                H=900,
                W=1600,
                rand_flip=True),
            training=False),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect4D',
                    keys=[
                        'img', 'voxel_semantics', 'mask_lidar', 'mask_camera',
                        'temporal_semantics', 'temporal_ego_states',
                        'temporal_trajs', 'temporal_agent_boxes',
                        'temporal_agent_feats'
                    ],
                    meta_keys=[
                        'filename', 'box_type_3d', 'ori_shape', 'img_shape',
                        'pad_shape', 'sample_idx', 'lidar2img',
                        'img_timestamp', 'ego2lidar', 'gt_boxes', 'gt_labels',
                        'occ_gt_path'
                    ])
            ])
    ],
    ann_file='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
    type='NuScenesDatasetOccpancy4DTraj',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=(0, 5, 1))
key = 'test'
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=36)
custom_hooks = [dict(type='CustomSetEpochInfoHook')]
revise_keys = [('backbone', 'img_backbone')]
cfg_name = 'sparse-occ-traj-finetune'
gpu_ids = [0]
