
## generating gif
> ```python train_td3.py --test --hidden_dim 256 --gpu 0 --env Pendulum-v0 --rnn LSTM --path save/TD3/randomize/LSTM/Pendulum-v0/best```
> ```python train_td3.py --test --hidden_dim 128 --gpu 0 --env Pendulum-v0 --rnn RNN --path save/TD3/randomize/fastRNN/Pendulum-v0/22Apr28161113/best```
> ```python train_td3.py --test --hidden_dim 128 --gpu 2 --env takeoff-aviary-v0 --rnn RNN --path save/TD3/randomize/RNN/takeoff-aviary-v0/22May03042352/best```


## Hyperparameter search
> ```python train_td3.py --env Pendulum-v0 --randomize --rnn GRU --tb_log --gpu 3 --hparam```

## pendulum run command

python train_pendulum_td3.py --policy_actf tanh --reward_norm --gpu 0 --rnn None --hparam 


## drone run command

### FF + reward normalize

> python train_drone_td3.py --policy_actf tanh --tb_log --reward_norm --gpu 1 --rnn None

### HER + no her + reward normalize

> python train_drone_td3.py --policy_actf tanh --her_length 100 --tb_log --reward_norm --her_gamma 1.0 --gpu 0 --rnn RNNHER 

### HER

> python train_drone_td3.py --policy_actf tanh --her_gamma 0.0 --tb_log --gpu 2 --rnn RNNHER

### HERbhv
> python train_drone_td3.py --policy_actf tanh --her_gamma 0.0 --her_length 100 --tb_log --gpu 0 --rnn RNNbhvHER --behavior_path artifacts/agent-22Jun01050026:v15/iter0055000

## Test drone in real world

> roslaunch sim2real sim2real.launch
> rostopic echo /crazyflie/log1


controller.c, estimate.c -> Default controller, Default estimate modifying -> flashing