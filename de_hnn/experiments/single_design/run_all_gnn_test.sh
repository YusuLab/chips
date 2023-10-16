#!/bin/bash

# for n in 30 #3 8 13 19 24 30 36 42 47; 
# do 
# 	source train_gnn_hetero_net.sh $n 1 sota demand 1 0 1000
# done

#for n in 52 58 64 70; #3 8 13 19 24 30 36 42 47;
#do 
#	source train_gnn_hetero_net.sh $n 1 sota hpwl 1 0 1000
#done


for n in 70;
do 
	source train_gnn_hetero_net.sh $n 0 gcn hpwl 0 0 1000
done
