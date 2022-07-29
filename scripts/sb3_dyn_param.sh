#!/bin/bash
python train_drone_sb3.py --tb_log --mode SAC --gpu $1 --rew_angvel_z 0.05 --dyn $2 --param
# python train_drone_sb3.py --tb_log --mode SAC --gpu $1 --rew_angvel_z 0.05 --dyn $2