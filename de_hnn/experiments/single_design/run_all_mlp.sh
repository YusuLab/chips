#!/bin/bash

for n in 19 52 58;
do
 	source train_mlp_net.sh $n hpwl
done

#for n in 52 58 64 70; #3 8 13 19 24 30 36 42 47 52 58 64 70;
#do
#	source train_mlp.sh $n classify
#done

