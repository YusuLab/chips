#!/bin/bash

for fold in 0 5 10 15 21 26
do
python3 create_split.py $fold
done