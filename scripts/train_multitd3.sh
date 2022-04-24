# python train_td3.py --env Pendulum-v0 --tb_log --gpu $1 --multitask
# python train_td3.py --env Pendulum-v0 --tb_log --gpu $1 --lstm --multitask
python train_td3.py --env takeoff-aviary-v0 --tb_log --gpu $1 --multitask
python train_td3.py --env takeoff-aviary-v0 --tb_log --gpu $1 --lstm --multitask