#!/bin/bash
#gcn
#for n in 58 64 70;
#do 
#	source train_gnn_hetero.sh $n 1 gcn classify 2 0
#done

for n in 3 8 13 24 30 36 42 47 52 58 64 70;
do 
	source train_gnn_hetero.sh $n 0 hyper classify 1 0 
done

for n in 3 8 13 24 30 36 42 47 52 58 64 70;
do 
	source train_gnn_hetero.sh $n 1 hyper classify 1 1
done

for n in 3 8 13 24 30 36 42 47 52 58 64 70;
do
        source train_gnn_hetero.sh $n 1 hyper classify 1 0
done

for n in 3 8 13 24 30 36 42 47 52 58 64 70;
do
        source train_gnn_hetero.sh $n 1 hyper classify 1 2
done

for n in 3 8 13 24 30 36 42 47 52 58 64 70;
do
        source train_gnn_hetero.sh $n 0 hypernodir classify 1 0
done



