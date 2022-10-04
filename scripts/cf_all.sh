#!/bin/bash
var=5
python train_drone_td3.py --rnn None --tb_log --seed $var --gpu $1
python train_drone_td3.py --rnn LSTMsHER --tb_log --seed $var --gpu $1
python train_drone_td3.py --rnn LSTMHER --tb_log --seed $var --gpu $1
python train_drone_td3.py --rnn LSTM2 --tb_log --seed $var --gpu $1