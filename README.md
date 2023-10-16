# DE-HNN: An effective neural model for Circuit Netlist representation

This is the github repository for the codes for paper: "DE-HNN: An effective neural model for Circuit Netlist representation"

## How to generate data

Raw data is available at [url]. After download the raw data, please extract and put the raw data directory to "/de_hnn/data/" directory. 

After that, please run following command to generate the full processed dataset. The whole process can last for hours. 

```commandline
source run_all_python_scripts.sh
``` 

## How to train 

After the data is created, the experiments can be run following the instrcutions below. 

### Simple Test

First, enter directory "/de_hnn/experiments/cross_design/".

A simple test on cross-design full-DE-HNN can be run using the following command:

```commandline
source run_simple_test.sh
```
A training process will be initiated immediately. This is a simple test just to test if codes are ready to use. 


A more complete test run with full number of epoches can be run using the following command:

```commandline
source run_full_test.sh
```

The whole process might take 30 minutes to hours depending on devices, which will produce the full results for full-DE-HNN for cross-design.


### For full single-design experiments

All files are ready in directory "/de_gnn/experiments/single_design/".

Please run the commands in this format "source run_all_{model}.sh" depending on which model you want to run. 


### For full cross-design experiments

All files are ready in directory "/de_gnn/experiments/cross_design/".

Please run the commands in this format "source run_all_{model}.sh" depending on which model you want to run.


Thank you!
