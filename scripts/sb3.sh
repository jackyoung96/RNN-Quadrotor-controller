#!/bin/bash
for var in 0 1 2 3 4
do
    python train_drone_sb3.py --tb_log --mode $1 --gpu $2
done