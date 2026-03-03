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
class GR2T2V2HumanoidRoughPPORunnerCfg_WBC_LOWER(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10001
    save_interval = 500
    experiment_name = "gr2t2v2_humanoid_rough_wbc_lower"
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
        sym_coef = 1.0,

    )
