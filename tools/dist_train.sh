#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

PORT=$((RANDOM + 10000))
NCCL_DEBUG=INFO
NCCL_P2P_DISABLE=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --work-dir ./work_dirs/our-nusc-base \
    --launcher pytorch ${@:3}

