#!/bin/bash

program=train_transformer
dir=./$program/
mkdir $dir

data_dir=../../data/2023-03-06_data/

num_epoch=100
batch_size=1
learning_rate=0.001
seed=123456789
hidden_dim=32
heads=4
local_heads=1
depth=1

# Device
device=cuda
device_idx=5

# Test mode
test_mode=0

# Position encoding dimension if used
pe_dim=16

# Position encoding
pe_type=lap

for fold in 0 1 2 3 4 5
do
name=${program}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.hidden_dim.${hidden_dim}.heads.${heads}.local_heads.${local_heads}.depth.${depth}.pe_type.${pe_type}.pe_dim.${pe_dim}.fold.${fold}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --dir=$dir --data_dir=${data_dir} --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --hidden_dim=$hidden_dim --heads=$heads --local_heads=$local_heads --depth=$depth --test_mode=$test_mode --pe_type=$pe_type --pe_dim=$pe_dim --fold=$fold
done
