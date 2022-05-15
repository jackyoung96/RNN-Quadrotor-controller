python train_td3.py --env Pendulum-v0 --gpu 3 --rnn RNN3 --tb_log --hparam --lr_scheduler
python train_td3.py --env Pendulum-v0 --gpu 3 --rnn LSTM3 --tb_log --hparam --lr_scheduler
python train_td3.py --env Pendulum-v0 --gpu 3 --rnn GRU3 --tb_log --hparam --lr_scheduler