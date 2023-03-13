#!/bin/bash

program=train_mlp
dir=./$program/
mkdir $dir

data_dir=../../data/2023-03-06_data/

num_epoch=100
batch_size=1
learning_rate=0.001
seed=123456789
hidden_dim=512

# Position encoding
load_pe=1
num_eigen=5

# Device
device=cuda
device_idx=0

# Test mode
test_mode=0

for fold in 0 1 2 3 4 5
do
name=${program}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.hidden_dim.${hidden_dim}.load_pe.${load_pe}.num_eigen.${num_eigen}.fold.${fold}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --dir=$dir --data_dir=${data_dir} --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --hidden_dim=$hidden_dim --load_pe=${load_pe} --num_eigen=${num_eigen} --test_mode=$test_mode --fold=$fold --device=$device
done

