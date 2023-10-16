#!/bin/bash

# #gcn
#for n in 3 8 13 24 30 36 42 47 52 58 64 70;
#do
#	source train_gnn_hetero_hpwl.sh $n 1 hyper hpwl 1 0 1000
#done

for n in 3 8 13 24 30 36 42 47 52 58 64 70; 
do 
	source train_gnn_hetero_hpwl.sh $n 1 hyper hpwl 1 1 1000
done

for n in 3 8 13 24 30 36 42 47 52 58 64 70; 
do
	source train_gnn_hetero_hpwl.sh $n 1 hyper hpwl 1 2 1000
done

#for n in 3 8 13 24 30 36 42 47 52 58 64 70;
#do
#	source train_gnn_hetero_net.sh $n 0 hyper hpwl 2 0 1000
#done

#for n in 3 8 13 24 30 36 42 47 52 58 64 70;
#do
#	source train_gnn_hetero_net.sh $n 0 hypernodir hpwl 2 0 1000
#done



#
#for n in 3; 
#do
#	source train_gnn_hetero_net.sh $n 0 hypernodir hpwl 2 0 1000
#done 
#
#for n in 3 8 13 30 36 52 58; 
#do
# 	source train_gnn_hetero_net.sh $n 1 hyper hpwl 4 0 1000
#done

#for n in 3 8 13 24 30 36 52 58; 
#do
#	source train_gnn_hetero_net.sh $n 1 hyper hpwl 4 1 1000
#done

#for n in 3 8 13 24 30 36 52 58; 
#do 
#	source train_gnn_hetero_net.sh $n 1 hyper hpwl 4 2 1000
#done


