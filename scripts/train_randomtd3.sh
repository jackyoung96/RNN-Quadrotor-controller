python train_td3.py --env Pendulum-v0 --tb_log --gpu $1 --randomize
python train_td3.py --env Pendulum-v0 --tb_log --gpu $1 --lstm --randomize
# python train_td3.py --env takeoff-aviary-v0 --tb_log --gpu $1 --randomize
# python train_td3.py --env takeoff-aviary-v0 --tb_log --gpu $1 --lstm --randomize