python train_td3.py --env Pendulum-v0 --gpu 2 --rnn RNN2 --tb_log --hparam --lr_scheduler
python train_td3.py --env Pendulum-v0 --gpu 2 --rnn LSTM2 --tb_log --hparam --lr_scheduler
python train_td3.py --env Pendulum-v0 --gpu 2 --rnn GRU2 --tb_log --hparam --lr_scheduler