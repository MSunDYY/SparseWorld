#!/usr/bin/env bash

GPUS=$1
PORT=$((RANDOM + 10000))
NCCL_DEBUG=INFO
NCCL_P2P_DISABLE=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test_temporal.py \
    --eval segm \
    --launcher pytorch \
    ${@:2}
