import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##
gym.register(
    id="PPV233_Mask_WBC",
    entry_point="GRX_humanoid.tasks.locomotion.velocity.env.multi_stage_manager_based_rl_env:MultiStageManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PPV233_Mask_WBC,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPV233_Mask_WBC",
    },
)

gym.register(
    id="PPV233_Mask_WBC_Play",
    entry_point="GRX_humanoid.tasks.locomotion.velocity.env.multi_stage_manager_based_rl_env:MultiStageManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PPV233_Mask_WBC_Play,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPV233_Mask_WBC_Play",
    },
)

gym.register(
    id="PPV233HumanoidRoughEnvCfg_WBC_FULL",
    entry_point="GRX_humanoid.tasks.locomotion.velocity.env.multi_stage_manager_based_rl_env:MultiStageManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PPV233HumanoidRoughEnvCfg_WBC_FULL,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPV233HumanoidRoughPPORunnerCfg_WBC_FULL",
    },
)

gym.register(
    id="PPV233HumanoidRoughEnvCfg_WBC_FULL_Play",
    entry_point="GRX_humanoid.tasks.locomotion.velocity.env.multi_stage_manager_based_rl_env:MultiStageManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PPV233HumanoidRoughEnvCfg_WBC_FULL_Play,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPV233HumanoidRoughPPORunnerCfg_WBC_FULL",
    },
)


gym.register(
    id="PPV233HumanoidRoughEnvCfg_WBC_LOWER",
    entry_point="GRX_humanoid.tasks.locomotion.velocity.env.multi_stage_manager_based_rl_env:MultiStageManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PPV233HumanoidRoughEnvCfg_WBC_LOWER,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPV233HumanoidRoughPPORunnerCfg_WBC_LOWER",
    },
)

gym.register(
    id="PPV233HumanoidRoughEnvCfg_WBC_LOWER_Play",
    entry_point="GRX_humanoid.tasks.locomotion.velocity.env.multi_stage_manager_based_rl_env:MultiStageManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.PPV233HumanoidRoughEnvCfg_WBC_LOWER_Play,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PPV233HumanoidRoughPPORunnerCfg_WBC_LOWER",
    },
)