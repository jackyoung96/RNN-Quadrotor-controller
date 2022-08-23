from sac.sac import *
from sac.common.buffers import *

def td3_agent(env,
                rnn,
                device,
                hparam,
                replay_buffer_size=1e6,
                ):

    if rnn in ["RNN2", "LSTM2", "GRU2"]:
        # Param + FF
        replay_buffer = ReplayBufferRNN(replay_buffer_size, **hparam)
        td3_trainer = TD3RNN_Trainer2(replay_buffer,
                    env.env.observation_space, 
                    env.env.action_space, 
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env.env_name else 10.0,
                    device=device, 
                    **hparam)
    elif rnn in ["RNN3", "LSTM3", "GRU3"]:
        replay_buffer = ReplayBufferRNN(replay_buffer_size, **hparam)
        td3_trainer = TD3RNN_Trainer3(replay_buffer,
                    env.env.observation_space, 
                    env.env.action_space, 
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env.env_name else 10.0,
                    device=device, 
                    **hparam)
    elif rnn in ["FFHER"]:
        replay_buffer = HindsightReplayBuffer(replay_buffer_size,
                            env=env.env_name,
                            **hparam)
        # goal_dim = observation_space.shape[0]-4 if 'aviary' in env.env_name else observation_space.shape[0]
        td3_trainer = TD3_Trainer(replay_buffer,
                    env.env.observation_space, 
                    env.env.action_space, 
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env.env_name else 10.0,
                    device=device, 
                    **hparam)
    elif rnn in ["RNNHER", "LSTMHER", "GRUHER"]:
        replay_buffer = HindsightReplayBufferRNN(replay_buffer_size,
                            env=env.env_name,
                            **hparam)
        # goal_dim = observation_space.shape[0]-4 if 'aviary' in env.env_name else observation_space.shape[0]
        td3_trainer = TD3HERRNN_Trainer(replay_buffer,
                    env.env.observation_space, 
                    env.env.action_space, 
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env.env_name else 10.0,
                    device=device, 
                    **hparam)
    elif rnn in ["RNNsHER", "LSTMsHER", "GRUsHER"]:
        replay_buffer = SingleHindsightReplayBufferRNN(replay_buffer_size,
                            env=env.env_name,
                            **hparam)
        td3_trainer = TD3sHERRNN_Trainer(replay_buffer,
                    env.env.observation_space, 
                    env.env.action_space, 
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env.env_name else 10.0,
                    device=device, 
                    **hparam)
    elif rnn == "None":
        replay_buffer = ReplayBuffer(replay_buffer_size, **hparam)
        td3_trainer = TD3_Trainer(replay_buffer,
                    env.env.observation_space, 
                    env.env.action_space, 
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env.env_name else 10.0,
                    device=device, 
                    **hparam)
    else:
        raise "Something wrong"

    return td3_trainer

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
    elif rnn in ["RNN2", "LSTM2", "GRU2"]:
        # Param + FF
        replay_buffer = ReplayBufferRNN(replay_buffer_size, **hparam)
        td3_trainer = SACRNN_Trainer2(replay_buffer,
                    env.env.observation_space, 
                    env.env.action_space, 
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env.env_name else 10.0,
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