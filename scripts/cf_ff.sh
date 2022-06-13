for s in {0..4}
do
    python train_drone_td3.py --rnn None --tb_log --reward_norm --seed $s --gpu $1
done