#!/bin/bash

for seed in $(($1 + 0)) $(($1 + 1)) $(($1 + 2))
do
    python train_drone_sac.py --rew_angvel_z 1.0  --seed $seed --rnn GRUparam --gpu $2 --tb_log $3
    python train_drone_sac.py --rew_angvel 0.05 --seed $seed --rnn GRUparam --gpu $2 --tb_log $3
    python train_drone_sac.py --rew_angvel 0.05 --rew_angvel_z 0.05 --seed $seed --rnn GRUparam --gpu $2 --tb_log $3
done