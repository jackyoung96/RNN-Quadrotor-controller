None_norand="models/SAC_22Aug28025017/iter0020000"
# None_rand="models/SAC_22Aug26205040/iter0020000"
# None_rand="models/SAC_22Sep06183520/iter0040000"
None_rand="models/SAC_22Sep08220553/iter0040000"
RNN="models/SAC_22Aug24212908/iter0020000"
RNNpolicy="models/SAC_22Aug24213629/iter0020000"
# RNNparam="models/SAC_22Aug28060728/iter0020000"
RNNparam="models/SAC_22Sep06183450/iter0040000"
test=10



for test_dyn_range in 0.0 0.1 0.2 0.3
do
    echo "------------------------------------------------"
    echo "Dynamic randomization range" $test_dyn_range
    echo "------------------------------------------------"
    # PID
    python train_drone_sac.py --gpu -1 --rnn PID --rew_angvel_z 0.05 --seed 10000 --test $test --test_dyn_range $test_dyn_range 
    # None no rand
    python train_drone_sac.py --gpu -1 --rnn None --rew_angvel_z 0.05 --seed 10000 --test $test --path $None_norand --test_dyn_range $test_dyn_range --rnn_name FF-norand
    # None rand
    python train_drone_sac.py --gpu -1 --rnn None --rew_angvel_z 0.05 --seed 10000 --test $test --path $None_rand --test_dyn_range $test_dyn_range --rnn_name FF-rand
    # RNN
    # python train_drone_sac.py --gpu -1 --rnn GRU --rew_angvel_z 0.05 --seed 10000 --test $test --path $RNN --test_dyn_range $test_dyn_range 
    # RNNpolicy
    python train_drone_sac.py --gpu -1 --rnn GRUpolicy --rew_angvel_z 0.05 --seed 10000 --test $test --path $RNNpolicy --test_dyn_range $test_dyn_range 
    # RNNparam
    python train_drone_sac.py --gpu -1 --rnn GRUparam --rew_angvel_z 0.05 --seed 10000 --test $test --path $RNNparam --test_dyn_range $test_dyn_range  
done