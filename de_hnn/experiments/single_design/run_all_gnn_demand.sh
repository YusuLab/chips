#!/bin/bash

#for n in 3 8 13 19 24 30 36 42 47 52 58 64 70; 
#do 
#	source train_gnn_hetero_net.sh $n 0 gcn demand 4 0 1000
#done

# for n in 3 8 13 19 24 30 36 42 47 52 58 64 70; 
# do 
# 	source train_gnn_hetero_net.sh $n 1 gat demand 3 0 1000
# done

for n in 3 8 13 19 24 30 36 42 47 52 58 64 70; 
do
 	source train_gnn_hetero_net.sh $n 0 hypernodir demand 1 0 1000
done

# for n in 3 8 13 19 24 30 36 42 47;
# do
# 	source train_gnn_hetero_net.sh $n 1 hypernodir demand 3 0 1000
# done

# for n in 3 8 13 19 24 30 36 42 47;
# do
#  	source train_gnn_hetero_net.sh $n 1 hyper demand 3 0 1000
# done

# for n in 3 8 13 19 24 30 36 42 47 52 58 64 70; 
# do
# 	source train_gnn_hetero_net.sh $n 1 hyper demand 3 1 1000
# done

# for n in 3 8 13 19 24 30 36 42 47 52 58 64 70; 
# do 
# 	source train_gnn_hetero_net.sh $n 1 hyper demand 3 2 1000
# done
