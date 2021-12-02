#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/home/mbed/Projects/haptic_transformer

#nohup python -u transformer_train.py --num-encoder-layers 2 --nheads 4 --model-type "haptr_modatt" >haptr_light1_modatt.log &&
  nohup python -u transformer_train.py --num-encoder-layers 2 --nheads 4 --model-type "haptr" >haptr_light1.log &&
  nohup python -u transformer_train.py --num-encoder-layers 4 --nheads 8 --model-type "haptr_modatt" >haptr_base1_modatt.log &&
  nohup python -u transformer_train.py --num-encoder-layers 4 --nheads 8 --model-type "haptr" >haptr_base1.log &&
  nohup python -u transformer_train.py --num-encoder-layers 8 --nheads 8 --model-type "haptr_modatt" >haptr_large1_modatt.log &&
  nohup python -u transformer_train.py --num-encoder-layers 8 --nheads 8 --model-type "haptr" >haptr_large1.log &
