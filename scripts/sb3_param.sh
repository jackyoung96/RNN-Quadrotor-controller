#!/bin/bash
for var in 0 1 2 3 4
do
    python train_drone_sb3.py --tb_log --mode $1 --gpu $2 --param --rew_angvel $3 --rew_angvel_xy $4 --rew_angvel_z $5 $6
done