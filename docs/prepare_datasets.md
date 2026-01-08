## Prepare nuScenes Dataset
Please organize the dataset and the necessary files according to the following directory structure.

step 1: Download nuScenes V1.0 full dataset data from [HERE](https://www.nuscenes.org/download) on `./data/nuscenes`.

step 2: Download nuScenes-lidarseg data from [HERE](https://www.nuscenes.org/download).

step 3: Download (only) the 'gts' from [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D).

step 4: Download the pkl files from [Huggingface](https://huggingface.co/MSunDYY2001/SparseWorld/tree/main/data/nuscenes).

step 5: Download pre-trained model weights from [Huggingface](https://huggingface.co/MSunDYY2001/SparseWorld/tree/main/ckpts).

step 6 Download pkl files preprocessed by admlp and occworld from [Huggingface](https://huggingface.co/MSunDYY2001/SparseWorld/tree/main), which is consistent with [PreWorld](https://github.com/getterupper/PreWorld).
## Folder structure

```
SparseWorld
├── mmdet3d/
├── tools/
├── env/
├── configs/
├── ckpts/
│   ├── epoch_56.pth
|   ├── cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
├── data/
│   ├── nuscenes/
│   │   ├── gts/  # ln -s occupancy gts to this location
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-trainval/
|   |   ├── bevdetv2-nuscenes_infos_val.pkl    
|   |   ├── bevdetv2-nuscenes_infos_train.pkl  
├── occworld/
│   ├── nuscenes_infos_train_temporal_v3_scene.pkl
│   └── nuscenes_infos_val_temporal_v3_scene.pkl
├── admlp/
│   ├── fengze_nuscenes_infos_val.pkl
│   ├── fengze_nuscenes_infos_train.pkl
│   └── stp3_val
│       ├── data_nuscene.pkl
│       ├── filter_token.pkl
│       ├── stp3_occupancy.pkl
│       └── stp3_traj_gt.pkl

```

Then, create a symbolic link for the planning evaluation.
```
cd AD-MLP/pytorch/admlp
ln -s ../../../admlp/stp3_val stp3_val
cd ../../..
```