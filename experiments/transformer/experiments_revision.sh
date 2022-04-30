#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/home/mbed/Projects/haptic_transformer

nohup python -u transformer_train.py --epochs 1750 --batch-size 128 --num-encoder-layers 2 --nheads 4 --model-type "haptr_modatt" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/put_split_haptr_12.yaml" >haptr_light.log &&
nohup python -u transformer_train.py --epochs 1750 --batch-size 128 --num-encoder-layers 2 --nheads 4 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/qcat_haptr_1_only.yaml" >qcat_1_only.log &&
nohup python -u transformer_train.py --epochs 1750 --batch-size 128 --num-encoder-layers 2 --nheads 4 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/qcat_haptr_2_only.yaml" >qcat_2_only.log &

