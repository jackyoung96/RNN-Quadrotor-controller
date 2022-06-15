#!/bin/bash
for var in 0 1 2 3 4
do
    python train_drone_td3.py --rnn RNN2 --tb_log --seed $var --gpu $1
done