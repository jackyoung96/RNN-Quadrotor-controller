from sac.sac import *
from sac.common.buffers import *

def sac_agent(env,
                rnn,
                device,
                hparam,
                replay_buffer_size=1e6,
                ):
    if rnn in ["RNN", "LSTM", "GRU"]:
        replay_buffer = ReplayBufferRNN(replay_buffer_size, **hparam)
        td3_trainer = SACRNN_Trainer(replay_buffer,
                    env.observation_space, 
                    env.action_space,
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0,
                    device=device, 
                    **hparam)
    elif rnn in ["RNNparam", "LSTMparam", "GRUparam"]:
        replay_buffer = ReplayBufferRNN(replay_buffer_size, **hparam)
        td3_trainer = SACparam_Trainer(replay_buffer,
                    env.observation_space, 
                    env.action_space,
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0,
                    device=device, 
                    **hparam)
    elif rnn in ["RNNpolicy", "LSTMpolicy", "GRUpolicy"]:
        # Param + FF
        replay_buffer = ReplayBufferRNN(replay_buffer_size, **hparam)
        td3_trainer = SACRNNpolicy_Trainer(replay_buffer,
                    env.observation_space, 
                    env.action_space, 
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0,
                    device=device, 
                    **hparam)
    elif rnn == "None":
        replay_buffer = ReplayBuffer(replay_buffer_size, **hparam)
        td3_trainer = SAC_Trainer(replay_buffer,
                    env.observation_space, 
                    env.action_space, 
                    out_actf=F.tanh,
                    action_scale=1.0,
                    device=device, 
                    **hparam)
    else:
        raise "Something wrong"

    return td3_trainer