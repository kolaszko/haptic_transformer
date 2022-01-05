#!/bin/sh

DATASET=$1
BATCH=$2
EPOCHS=$3
MODEL=$4

export PYTHONPATH=$PYTHONPATH:/home/mbed/Projects/haptic_transformer

nohup python -u transformer_cv.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type "${MODEL}" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/${DATASET}_split_haptr_12.yaml" >${DATASET}_acc_${MODEL}_light.log &
