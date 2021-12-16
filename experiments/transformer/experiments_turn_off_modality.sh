#!/bin/sh

DATASET=$1
BATCH=$2
EPOCHS=$3

export PYTHONPATH=$PYTHONPATH:/home/mbed/Projects/haptic_transformer

nohup python -u transformer_train.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/${DATASET}_haptr_1_only.yaml" >${DATASET}_modoff_light_haptr_1_only.log &&
  nohup python -u transformer_train.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/${DATASET}_haptr_2_only.yaml" >${DATASET}_modoff_light_haptr_2_only.log &
