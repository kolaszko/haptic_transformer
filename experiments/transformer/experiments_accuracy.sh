#!/bin/sh

DATASET=$1
BATCH=$2
EPOCHS=$3
MODEL=$4

export PYTHONPATH=$PYTHONPATH:/home/mbed/Projects/haptic_transformer

nohup python -u transformer_train.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type "${MODEL}" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/${DATASET}_haptr_12.yaml" >${DATASET}_acc_${MODEL}_light.log &&
  nohup python -u transformer_train.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 4 --nheads 8 --model-type "${MODEL}" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/${DATASET}_haptr_12.yaml" >${DATASET}_acc_${MODEL}_base.log &&
  nohup python -u transformer_train.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 8 --nheads 8 --model-type "${MODEL}" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/${DATASET}_haptr_12.yaml" >${DATASET}_acc_${MODEL}_large.log &
