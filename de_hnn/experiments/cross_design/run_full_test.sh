#!/bin/bash

#source {train_script_file}.sh {number_of_layer} {use_pd_or_not} {target} {device_number} {use_vn}

source train_gnn_hetero_net.sh 3 1 gcn demand 0 0
source train_gnn_hetero_net.sh 3 1 allset demand 0 0 
source train_gnn_hetero_net.sh 3 1 hyper demand 0 1
