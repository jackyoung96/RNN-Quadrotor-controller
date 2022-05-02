
## generating gif
> ```python train_td3.py --test --hidden_dim 256 --gpu 0 --env Pendulum-v0 --rnn LSTM --path save/TD3/randomize/LSTM/Pendulum-v0/best```
> ```python train_td3.py --test --hidden_dim 128 --gpu 0 --env Pendulum-v0 --rnn RNN --path save/TD3/randomize/fastRNN/Pendulum-v0/22Apr28161113/best```

## Hyperparameter search
> ```python train_td3.py --env Pendulum-v0 --randomize --rnn GRU --tb_log --gpu 3 --hparam```