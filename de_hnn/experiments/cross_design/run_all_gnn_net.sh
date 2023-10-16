#!/bin/bash
#gcn
#source train_gnn_hetero_net.sh 0 1 gcn demand 2 0

#gat
#source train_gnn_hetero_net.sh 0 1 gat demand 2 0

#hyper pd-0
#source train_gnn_hetero_demand.sh 0 0 hyper demand 1 0

#hyper_no_dir
#source train_gnn_hetero_net.sh 0 1 hypernodir demand 3 0

# # #hyper normal
source train_gnn_hetero_demand.sh 0 1 hyper demand 1 1

#source train_gnn_hetero_net.sh 0 1 hyper demand 2 2

#source train_gnn_hetero_net.sh 0 0 hyper hpwl 2 0

#source train_gnn_hetero_net.sh 0 1 hyper hpwl 2 2



#gcn
#source train_gnn_hetero_demand.sh 0 1 gcn hpwl 4 0

#gat
#source train_gnn_hetero_demand.sh 0 1 sota hpwl 2 0

#hyper pd-0
#source train_gnn_hetero_net.sh 0 0 hyper demand 3 0

#hyper_no_dir
#source train_gnn_hetero_net.sh 0 1 hypernodir demand 1 1

# # #hyper normal
#source train_gnn_hetero_net.sh 0 1 hyper hpwl 2 1

 #hyper VN
#source train_gnn_hetero_demand.sh 0 0 hyper hpwl 6 1


#source train_gnn_hetero_demand.sh 0 1 hyper hpwl 6 2
