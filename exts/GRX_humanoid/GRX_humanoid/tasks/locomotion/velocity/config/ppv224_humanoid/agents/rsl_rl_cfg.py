import torch
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, 
    RslRlPpoActorCriticCfg, 
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlSymmetryCfg,
)
from GRX_humanoid.utils.wrappers.rsl_rl import (
    MYRslRlPpoAlgorithmCfg
)

@configclass
class PPV224HumanoidRoughPPORunnerCfg_WBC_LOWER(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10001
    save_interval = 500
    experiment_name = "ppv224_humanoid_rough_wbc_lower"
    empirical_normalization = False
    runner_class = "OnPolicyRunner"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 512, 128],
        critic_hidden_dims=[2048, 512, 128],
        activation="elu",
    )
    algorithm = MYRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        sym_loss = True,
        obs_terms = [
                {'name': 'base_ang_vel', 'permutation': [-0.0001, 1, -2]},
                {'name': 'projected_gravity', 'permutation': [0.0001, -1, 2]},
                {'name': 'is_walk_int', 'permutation': [0.0001]},
                {'name': 'velocity_commands', 'permutation': [0.0001, -1, -2]},
                {'name': 'height_attitude', 'permutation': [0.0001, 1, -2]},
                {'name': 'joint_pos', 'permutation': [6, -7, -8, 9, 10, -11, 0.0001, -1, -2, 3, 4, -5]},
                {'name': 'joint_vel', 'permutation': [6, -7, -8, 9, 10, -11, 0.0001, -1, -2, 3, 4, -5]},
                {'name': 'actions', 'permutation': [6, -7, -8, 9, 10, -11, 0.0001, -1, -2, 3, 4, -5]},
            ],
        act_permutation = [6, -7, -8, 9, 10, -11, 0.0001, -1, -2, 3, 4, -5],
        frame_stack = 40,
        sym_coef = 2.0,

    )


@configclass
class PPV224HumanoidRoughPPORunnerCfg_WBC_FULL(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30001
    save_interval = 500
    experiment_name = "ppv224_humanoid_rough_wbc_full"
    empirical_normalization = False
    runner_class = "OnPolicyRunner"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 512, 128],
        critic_hidden_dims=[2048, 512, 128],
        activation="elu",
    )
    algorithm = MYRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        sym_loss = True,
        obs_terms = [
                {'name': 'base_ang_vel', 'permutation': [-0.0001, 1, -2]},
                {'name': 'projected_gravity', 'permutation': [0.0001, -1, 2]},
                {'name': 'is_walk_int', 'permutation': [0.0001]},
                {'name': 'velocity_commands', 'permutation': [0.0001, -1, -2]},
                {'name': 'height_attitude', 'permutation': [0.0001, -1, -2, 3]},
                # {'name': 'joint_pos_cmd', 'permutation': [-0.0001, -1, 2, -3, 4, \
                #                                         12, -13, -14, 15, -16, 17, -18, 5, -6, -7, 8, -9, 10, -11]},
                {'name': 'joint_pos_cmd', 'permutation': [-0.0001, 1, \
                                                        9, -10, -11, 12, -13, 14, -15, 2, -3, -4, 5, -6, 7, -8]},                                        
                {'name': 'joint_pos', 'permutation': [6, -7, -8, 9, 10, -11, 0.0001, -1, -2, 3, 4, -5,\
                                                    -12, -13, 14, -15, 16, \
                                                    24, -25, -26, 27, -28, 29, -30, 17, -18, -19, 20, -21, 22, -23]},
                {'name': 'joint_vel', 'permutation': [6, -7, -8, 9, 10, -11, 0.0001, -1, -2, 3, 4, -5,\
                                                    -12, -13, 14, -15, 16, \
                                                    24, -25, -26, 27, -28, 29, -30, 17, -18, -19, 20, -21, 22, -23]},
                {'name': 'actions', 'permutation': [6, -7, -8, 9, 10, -11, 0.0001, -1, -2, 3, 4, -5,\
                                                    -12, -13, 14, -15, 16, \
                                                    24, -25, -26, 27, -28, 29, -30, 17, -18, -19, 20, -21, 22, -23]},
            ],
        act_permutation = [6, -7, -8, 9, 10, -11, 0.0001, -1, -2, 3, 4, -5,\
                        -12, -13, 14, -15, 16, \
                        24, -25, -26, 27, -28, 29, -30, 17, -18, -19, 20, -21, 22, -23],
        frame_stack = 40,
        sym_coef = 2.0,

    )
    resume = False
    #load_run = "2025-08-18_19-16-48"
    # load_run = ".*"
    #load_checkpoint = "model_.*.pt"
    # load_checkpoint = "model_16000.pt"

@configclass
class PPV224HumanoidRoughPPORunnerCfg_WBC_FULL_PLAY(PPV224HumanoidRoughPPORunnerCfg_WBC_FULL):
    #load_run = "2025-09-13_18-39-46"
    # load_run = ".*"
    # load_checkpoint = "model_49500.pt"
    #load_checkpoint = "model_.*.pt"
    pass