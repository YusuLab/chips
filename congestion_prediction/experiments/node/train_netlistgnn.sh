#!/bin/bash

program=train_netlistgnn

target=demand
#target=capacity
#target=congestion

mkdir $program
cd $program
mkdir $target
cd ..
dir=./$program/$target/

data_dir=/data/zluo/hetero_net/data/2023-03-06_data/

num_epoch=1000
batch_size=1
learning_rate=0.001
seed=123456789

# For NetlistGNN
n_layer=2
node_feats=16 #64
net_feats=16 #128
pin_feats=16
edge_feats=4

# Device
device=cuda
device_idx=4

# Position encoding
pe=lap
pos_dim=10

# Global information
load_global_info=0

# Persistence diagram & Neighbor list
load_pd=1

# Test mode
test_mode=0

# Execution
for graph_index in 0 5 10 16 21 26
do
name=${program}.${target}.n_layer.${n_layer}.node_feats.${node_feats}.net_feats.${net_feats}.pin_feats.${pin_feats}.edge_feats.${edge_feats}.num_epoch.${num_epoch}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.pe.${pe}.pos_dim.${pos_dim}.load_global_info.${load_global_info}.load_pd.${load_pd}.graph_index.${graph_index}
CUDA_VISIBLE_DEVICES=$device_idx python3 $program.py --n_layer=$n_layer --node_feats=$node_feats --net_feats=$net_feats --pin_feats=$pin_feats --edge_feats=$edge_feats --target=$target --dir=$dir --name=$name --num_epoch=$num_epoch --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --pe=$pe --pos_dim=$pos_dim --load_global_info=$load_global_info --load_pd=$load_pd --test_mode=$test_mode --device=$device --data_dir=$data_dir --graph_index=$graph_index
done

