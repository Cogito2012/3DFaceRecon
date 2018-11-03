#!/bin/bash

ROOT_PATH=$(cd $(dirname $0); pwd)

export PYTHONPATH=$PYTHONPATH:${ROOT_PATH}/rendering_layer
export PYTHONPATH=$PYTHONPATH:￥{ROOT_PATH}/utils

sudo rm -rf ./output/*

CUDA_VISIABLE_DEVICES=0 python trainval.py \
	--batch_size 4 \
	--phase train \
    --base_lr 0.00001 \
    --display 2


