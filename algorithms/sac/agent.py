from .sac import *
from .common.buffers import *
from .PIDcontroller import SimplePIDcontroller, DSLPIDcontroller

def sac_agent(env,
            device,
            cfg):
    if cfg.algorithms.rnn in ["RNN", "LSTM", "GRU"]:
        replay_buffer = ReplayBufferRNN(cfg.algorithms.replay_buffer_size, cfg)
        agent = SACRNN_Trainer(replay_buffer,
                    env.observation_space, 
                    env.action_space,
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0,
                    device=device, 
                    **kwargs)
    elif cfg.algorithms.rnn in ["RNNfull", "LSTMfull", "GRUfull"]:
        replay_buffer = ReplayBufferRNN(cfg.algorithms.replay_buffer_size, **kwargs)
        agent = SACRNNfull_Trainer(replay_buffer,
                    env.observation_space, 
                    env.action_space,
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0,
                    device=device, 
                    **kwargs)
    elif cfg.algorithms.rnn in ["RNNparam", "LSTMparam", "GRUparam"]:
        replay_buffer = ReplayBufferRNN(cfg.algorithms.replay_buffer_size, **kwargs)
        agent = SACparam_Trainer(replay_buffer,
                    env.observation_space, 
                    env.action_space,
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0,
                    device=device, 
                    **kwargs)
    elif cfg.algorithms.rnn in ["RNNpolicy", "LSTMpolicy", "GRUpolicy"]:
        # Param + FF
        replay_buffer = ReplayBufferRNN(cfg.algorithms.replay_buffer_size, **kwargs)
        agent = SACRNNpolicy_Trainer(replay_buffer,
                    env.observation_space, 
                    env.action_space, 
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0,
                    device=device, 
                    **kwargs)
    elif cfg.algorithms.rnn in ["RNNpolicyfull", "LSTMpolicyfull", "GRUpolicyfull"]:
        # Param + FF
        replay_buffer = ReplayBufferRNN(cfg.algorithms.replay_buffer_size, **kwargs)
        agent = SACRNNpolicyfull_Trainer(replay_buffer,
                    env.observation_space, 
                    env.action_space, 
                    rnn_type=rnn,
                    out_actf=F.tanh,
                    action_scale=1.0,
                    device=device, 
                    **kwargs)
    elif cfg.algorithms.rnn == "None":
        replay_buffer = ReplayBuffer(cfg.algorithms.replay_buffer_size)
        agent = \
            SAC_Trainer(
                replay_buffer=replay_buffer,
                state_space=env.observation_space, 
                action_space=env.action_space, 
                actor_hidden_dim=cfg.algorithms.actor_hidden_dim,
                critic_hidden_dim=cfg.algorithms.critic_hidden_dim,
                actor_learning_rate=cfg.algorithms.actor_learning_rate,
                critic_learning_rate=cfg.algorithms.critic_learning_rate,
                activation=getattr(F,cfg.algorithms.activation),
                target_update_interval=cfg.algorithms.target_update_interval,
                action_scale=1.0,
                out_actf=F.tanh,
                device=device, 
            )
    elif cfg.algorithms.rnn == "PID":
        agent = DSLPIDcontroller(env)
    else:
        raise "Something wrong"

    return agent