#!/bin/sh

DATASET=$1
BATCH=$2
EPOCHS=$3

export PYTHONPATH=$PYTHONPATH:/home/mbed/Projects/haptic_transformer

nohup python -u transformer_train.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type "haptr_modatt" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/${DATASET}_haptr_12_split.yaml" >${DATASET}_acc_haptr_light_modatt.log &&
  nohup python -u transformer_train.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 2 --nheads 4 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/${DATASET}_haptr_12.yaml" >${DATASET}_acc_haptr_light.log &&
  nohup python -u transformer_train.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 4 --nheads 8 --model-type "haptr_modatt" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/${DATASET}_haptr_12_split.yaml" >${DATASET}_acc_haptr_base_modatt.log &&
  nohup python -u transformer_train.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 4 --nheads 8 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/${DATASET}_haptr_12.yaml" >${DATASET}_acc_haptr_base.log &&
  nohup python -u transformer_train.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 8 --nheads 8 --model-type "haptr_modatt" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/${DATASET}_haptr_12_split.yaml" >${DATASET}_acc_haptr_large_modatt.log &&
  nohup python -u transformer_train.py --epochs ${EPOCHS} --batch-size ${BATCH} --num-encoder-layers 8 --nheads 8 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/${DATASET}_haptr_12.yaml" >${DATASET}_acc_haptr_large.log &
