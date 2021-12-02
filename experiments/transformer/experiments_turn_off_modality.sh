#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/home/mbed/Projects/haptic_transformer

nohup python -u transformer_train.py --num-encoder-layers 2 --nheads 4 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/put_haptr_ft.yaml" >light_haptr.log &&
  nohup python -u transformer_train.py --num-encoder-layers 2 --nheads 4 --model-type "haptr_modatt" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/put_haptr_ft_split.yaml" >light_haptr_modatt.log &&
  nohup python -u transformer_train.py --num-encoder-layers 2 --nheads 4 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/put_haptr_one_f_only.yaml" >light_haptr_force_only.log &&
  nohup python -u transformer_train.py --num-encoder-layers 2 --nheads 4 --model-type "haptr" --dataset-config-file "/home/mbed/Projects/haptic_transformer/experiments/config/put_haptr_one_t_only.yaml" >light_haptr_torque_only.log &
