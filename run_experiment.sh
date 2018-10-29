#!/bin/bash

ROOT_PATH=$(cd $(dirname $0); pwd)

export PYTHONPATH=$PYTHONPATH:${ROOT_PATH}/rendering_layer
export PYTHONPATH=$PYTHONPATH:￥{ROOT_PATH}/utils

CUDA_VISIABLE_DEVICES=0 python trainval.py


