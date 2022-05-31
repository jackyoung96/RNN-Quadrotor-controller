
## generating gif
> ```python train_td3.py --test --hidden_dim 256 --gpu 0 --env Pendulum-v0 --rnn LSTM --path save/TD3/randomize/LSTM/Pendulum-v0/best```
> ```python train_td3.py --test --hidden_dim 128 --gpu 0 --env Pendulum-v0 --rnn RNN --path save/TD3/randomize/fastRNN/Pendulum-v0/22Apr28161113/best```
> ```python train_td3.py --test --hidden_dim 128 --gpu 2 --env takeoff-aviary-v0 --rnn RNN --path save/TD3/randomize/RNN/takeoff-aviary-v0/22May03042352/best```


## Hyperparameter search
> ```python train_td3.py --env Pendulum-v0 --randomize --rnn GRU --tb_log --gpu 3 --hparam```


## drone run command
> ```python train_drone_td3.py --rnn=RNNHER --policy_actf tanh --reward_norm  --her_gamma 0.0 --angvel_goal --her_length 100 --tb_log --gpu 2```
> ```python train_drone_td3.py --rnn=RNNHER --policy_actf tanh  --her_gamma 0.0 --angvel_goal --her_length 100 --tb_log --gpu 1```
> ```python train_drone_td3.py --rnn=RNNHER --policy_actf tanh --reward_norm  --her_gamma 1.0 --her_length 100 --tb_log --gpu 0```

> ```python train_drone_td3.py --rnn=RNNHER --policy_actf tanh --reward_norm  --her_gamma 0.0 --her_length 100 --tb_log --gpu 3```
> ```python train_drone_td3.py --rnn=RNNHER --policy_actf tanh --reward_norm  --her_gamma 0.0 --positive_rew --her_length 100 --tb_log --gpu 2```