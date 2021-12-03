#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/home/mbed/Projects/haptic_transformer

nohup python -u transformer_train.py --num-encoder-layers 2 --nheads 4 --model-type "haptr_modatt" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/put_haptr_ft_split.yaml" >acc_haptr_light_modatt.log &&
  nohup python -u transformer_train.py --num-encoder-layers 2 --nheads 4 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/put_haptr_ft.yaml" >acc_haptr_light.log &&
  nohup python -u transformer_train.py --num-encoder-layers 4 --nheads 8 --model-type "haptr_modatt" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/put_haptr_ft_split.yaml" >acc_haptr_base_modatt.log &&
  nohup python -u transformer_train.py --num-encoder-layers 4 --nheads 8 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/put_haptr_ft.yaml" >acc_haptr_base.log &&
  nohup python -u transformer_train.py --num-encoder-layers 8 --nheads 8 --model-type "haptr_modatt" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/put_haptr_ft_split.yaml" >acc_haptr_large_modatt.log &&
  nohup python -u transformer_train.py --num-encoder-layers 8 --nheads 8 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/put_haptr_ft.yaml" >acc_haptr_large.log &
