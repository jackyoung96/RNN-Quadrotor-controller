import hydra
from omegaconf import OmegaConf


from algorithms.trainer import Trainer

@hydra.main(config_path="./configs", config_name="default")
def main(cfg):
    if cfg.envs.name == 'drone':
        from envs.dynRandDroneEnv import dynRandDroneEnv
        # Specified for Gym-pybullet-drone environment
        # https://github.com/utiasDSL/gym-pybullet-drones
        env = dynRandDroneEnv(cfg)
        eval_env = dynRandDroneEnv(cfg)
    else:
        from envs.dynRandGymEnv import dynRandGymEnv
        # General gym-based environments
        # https://www.gymlibrary.dev/
        env = dynRandGymEnv(
            env_name=cfg.envs.name,
            max_episode_len=cfg.max_steps,
            dyn_range=OmegaConf.to_container(cfg.envs.dyn_range),
            seed=cfg.seed,
            )
        eval_env = dynRandGymEnv(
            env_name=cfg.envs.name,
            max_episode_len=cfg.max_steps,
            dyn_range=OmegaConf.to_container(cfg.envs.dyn_range),
            seed=cfg.seed,
            )
    trainer = Trainer(cfg, env, eval_env)
    trainer.run()

if __name__ =="__main__":
    main()