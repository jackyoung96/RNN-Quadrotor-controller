from td3.td3 import *
from td3.common.buffers import *

def td3_agent(env,
                rnn,
                device,
                hparam,
                replay_buffer_size=1e6,
                ):

    if rnn in ["RNN2", "LSTM2", "GRU2"]:
        # Param + FF
        replay_buffer = ReplayBufferFastAdaptRNN(replay_buffer_size, rnn, **hparam)
        td3_trainer = TD3RNN_Trainer2(replay_buffer,
                    env.env.action_space, 
                    env.env.action_space, 
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0 if 'aviary' in env.env.env_name else 10.0,
                    device=device, 
                    **hparam)
    if "HER" in rnn:
        # batch_size = batch_size*int(max_steps//her_sample_length / 2)
        replay_buffer = HindsightReplayBufferRNN(replay_buffer_size, rnn,
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
        # Use Behavior networks
        if "bhv" in rnn:
            if hparam['behavior_path']==None:
                raise FileNotFoundError("Need proper behavior_path")
            td3_trainer.load_behavior(hparam['behavior_path'])
            
    elif rnn == "None":
        replay_buffer = ReplayBufferFastAdaptRNN(replay_buffer_size, rnn, **hparam)
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