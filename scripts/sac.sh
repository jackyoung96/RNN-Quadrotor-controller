for var in 0 1 2 3 4
do
    python train_drone_sac.py --rew_angvel_z 0.05 --seed $var --rnn $1 --gpu $2 --tb_log $3
done