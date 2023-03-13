#!/bin/bash

program=train_gnn_hetero
dir=./$program/
mkdir $dir

data_dir=../../data/2023-03-06_data/

num_epoch=300
batch_size=1
learning_rate=0.001
seed=123456789
n_layers=6
hidden_dim=32

device=cuda
device_idx=0

# Position encoding
pe=none
pos_dim=5

# Test mode
test_mode=0

# GNN type
gnn_type=gcn

# Virtual node
virtual_node=0

for fold in 0 1 2 3 4 5
do
name=${program}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.n_layers.${n_layers}.hidden_dim.${hidden_dim}.pe.${pe}.pos_dim.${pos_dim}.virtual_node.${virtual_node}.gnn_type.${gnn_type}.fold.${fold}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --dir=$dir --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --n_layers=$n_layers --hidden_dim=$hidden_dim --pe=$pe --pos_dim=$pos_dim --virtual_node=$virtual_node --gnn_type=$gnn_type --test_mode=$test_mode --device=$device --data_dir=$data_dir --fold=$fold
done

