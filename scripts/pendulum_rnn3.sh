python train_td3.py --env Pendulum-v0 --gpu 3 --randomize --rnn RNN3 --tb_log --hparam --lr_scheduler
python train_td3.py --env Pendulum-v0 --gpu 3 --randomize --rnn LSTM3 --tb_log --hparam --lr_scheduler
python train_td3.py --env Pendulum-v0 --gpu 3 --randomize --rnn GRU3 --tb_log --hparam --lr_scheduler