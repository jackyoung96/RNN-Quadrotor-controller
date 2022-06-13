for s in {0..4}
do
    python train_drone_td3.py --rnn LSTMHER --tb_log --seed $s --gpu $1
done