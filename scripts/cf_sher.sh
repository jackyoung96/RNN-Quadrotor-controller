#!/bin/bash
for var in 0 1
do
    python train_drone_td3.py --rnn RNNHER --single_pos --tb_log --seed $var --gpu $1 
    # python train_drone_td3.py --rnn RNNsHER --maintain_length 30 --tb_log --seed $var --gpu $1
done