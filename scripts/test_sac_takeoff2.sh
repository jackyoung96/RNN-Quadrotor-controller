None_norand="models/SAC_22Aug28025017/iter0020000"
# None_rand="models/SAC_22Aug26205040/iter0020000"
# None_rand="models/SAC_22Sep06183520/iter0040000"
None_rand="models/SAC_22Sep08220553/iter0040000"
RNN="models/SAC_22Aug24212908/iter0020000"
RNNpolicy="models/SAC_22Aug24213629/iter0020000"
# RNNparam="models/SAC_22Aug28060728/iter0020000"
RNNparam="models/SAC_22Sep06183450/iter0037500"
test=1

# set_dyn="--mass 0.2 --cm 0.1 0.1 --I 0 0 0 --T 0 --KM 0 0 0 0.3 --KF 0 0 0 -0.3"
set_dyn="--mass 0 --cm 0 0 --I 0 0 0 --T 0 --KM 0 0 0 0.3 --KF 0 0 0 -0.3"

# PID
# python test_drone_sac.py --gpu -1 --rnn PID --seed 10000 --test $test $set_dyn --task takeoff
# None no rand
# python test_drone_sac.py --gpu -1 --rnn None --seed 10000 --test $test --path $None_norand $set_dyn --rnn_name FF-norand --task takeoff
# None rand
python test_drone_sac.py --gpu -1 --rnn None --seed 10000 --test $test --path $None_rand $set_dyn --rnn_name FF-rand --task takeoff
# RNNpolicy
# python test_drone_sac.py --gpu -1 --rnn GRUpolicy --seed 10000 --test $test --path $RNNpolicy $set_dyn --task takeoff
# RNNparam
python test_drone_sac.py --gpu -1 --rnn GRUparam --seed 10000 --test $test --path $RNNparam $set_dyn --task takeoff
