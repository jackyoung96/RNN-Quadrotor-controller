python train_td3.py --env Pendulum-v0 --gpu 2 --randomize --rnn RNN2 --tb_log --hparam --lr_scheduler
python train_td3.py --env Pendulum-v0 --gpu 2 --randomize --rnn LSTM2 --tb_log --hparam --lr_scheduler
python train_td3.py --env Pendulum-v0 --gpu 2 --randomize --rnn GRU2 --tb_log --hparam --lr_scheduler