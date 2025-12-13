# Prerequisites

**Please ensure you have prepared the environment and the dataset.**

# Train and Test

## Train 

1. Train PreWorld with 8 GPUs on nuScenes dataset (3D Occupancy Prediction):
    ```bash
    bash ./tools/dist_train.sh ./configs/preworld/nuscenes/preworld-7frame-finetune.py 8
    ```

    Or:
    ```bash
    # Pre-training Stage
    bash ./tools/dist_train.sh ./configs/preworld/nuscenes/preworld-7frame-pretrain.py 8
    # Fine-tuning Stage
    bash ./tools/dist_train.sh ./configs/preworld/nuscenes/preworld-7frame-finetune.py 8 --resume-from=./path/to/pre-trained_ckpts.pth  # Please modify 'max_epochs' in the configuration file accordingly.
    ```

2. Train PreWorld with 8 GPUs on nuScenes dataset (4D Occupancy Forecasting):
    ```bash
    bash ./tools/dist_train.sh configs/preworld/nuscenes-temporal/preworld-7frame-finetune-traj.py 8
    ```

    Or:
    ```bash
    # Pre-training Stage
    bash ./tools/dist_train.sh ./configs/preworld/nuscenes-temporal/preworld-7frame-pretrain-traj.py 8
    # Fine-tuning Stage
    bash ./tools/dist_train.sh ./configs/preworld/nuscenes-temporal/preworld-7frame-finetune-traj.py 8 --resume-from=./path/to/pre-trained_ckpts.pth  # Please modify 'max_epochs' in the configuration file accordingly.
    ```

## Evaluation 

Eval PreWorld with 8 GPUs on nuScenes dataset (3D Occupancy Prediction):
```bash
bash ./tools/dist_test.sh ./configs/preworld/nuscenes/preworld-7frame-finetune.py ./path/to/ckpts.pth 8
```

Eval PreWorld with 8 GPUs on nuScenes dataset (4D Occupancy Forecasting):
```bash
bash ./tools/dist_test_temporal.sh ./configs/preworld/nuscenes-temporal/preworld-7frame-finetune-traj.py ./path/to/ckpts.pth 8
```

## Visualization

3D Occupancy Prediction on nuScenes dataset:
```bash
# Dump predictions
bash ./tools/dist_test.sh ./configs/preworld/nuscenes/preworld-7frame-finetune.py ./path/to/ckpts.pth 1 --dump_dir=vis_dirs/output
# Visualization (select scene-id)
python tools/visualization/visual.py vis_dirs/output/scene-xxxx
```
(The pkl file needs to be regenerated for visualization.)