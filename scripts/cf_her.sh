python train_drone_td3.py --rnn RNNHER --tb_log --her_gamma 0 --seed 0 --gpu $1
python train_drone_td3.py --rnn LSTMHER --tb_log --her_gamma 0 --seed 0 --gpu $1

python train_drone_td3.py --rnn RNNHER --tb_log --her_gamma 0 --seed 1 --gpu $1
python train_drone_td3.py --rnn LSTMHER --tb_log --her_gamma 0 --seed 1 --gpu $1

python train_drone_td3.py --rnn RNNHER --tb_log --her_gamma 0 --seed 2 --gpu $1
python train_drone_td3.py --rnn LSTMHER --tb_log --her_gamma 0 --seed 2 --gpu $1

python train_drone_td3.py --rnn RNNHER --tb_log --her_gamma 0 --seed 3 --gpu $1
python train_drone_td3.py --rnn LSTMHER --tb_log --her_gamma 0 --seed 3 --gpu $1

python train_drone_td3.py --rnn RNNHER --tb_log --her_gamma 0 --seed 4 --gpu $1
python train_drone_td3.py --rnn LSTMHER --tb_log --her_gamma 0 --seed 4 --gpu $1