#!/bin/bash
for var in 0 1
do
    python train_drone_sb3.py --tb_log --mode PPO --gpu $1 --rew_angvel_z 0.05 --dyn $2
done