"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
import numpy as np

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner, OnPolicyRunnerAMP, OnPolicyRunnerDreamWaq, OnPolicyRunnerDreamWaqAMP

# Import extensions to set up environment tasks
import GRX_humanoid.tasks

from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx, export_policy_as_jit
from export_class import EncoderPolicyExpoter

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"log_root_path: {log_root_path}")
    print(f"agent_cfg.load_run: {agent_cfg.load_run}")
    print(f"agent_cfg.load_checkpoint: {agent_cfg.load_checkpoint}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    runner_class = eval(agent_cfg.to_dict().pop("runner_class"))
    ppo_runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic


    ### export policy and scale params
    # extract checkpoint iterration
    ckpt = resume_path.rsplit("/")[-1].split(".")[0].rsplit("_", 1)[-1]
    # export policy to onnx
    export_model_dir = os.path.join(log_root_path, "exported", f'{agent_cfg.load_run}.{ckpt}')
    print(f"[INFO] Exporting policy to: {export_model_dir}")
    # export_policy_as_onnx(policy_nn, export_model_dir, filename="policy.onnx")
    # export_policy_as_jit(policy_nn, None, export_model_dir, filename=f"policy_{ckpt}.pt")
    policy_exporter = EncoderPolicyExpoter(policy_nn)
    policy_exporter.export(export_model_dir)
    # export action and observation scales
    ac_dim = len(env_cfg.actions.joint_pos.joint_names)
    action_scale = env_cfg.actions.joint_pos.scale
    if isinstance(action_scale, dict):
        action_scale = [value for value in action_scale.values()]
    else:
        action_scale = [action_scale]*ac_dim
    # print("action_scale:", action_scale)
    joint_obs_scale = env_cfg.observations.policy.joint_pos.scale
    joint_obs_scale = list(joint_obs_scale) if isinstance(joint_obs_scale, (list, tuple)) else [joint_obs_scale]*ac_dim
    # print("joint_obs_scale:", joint_obs_scale)
    obs_scale = np.array([env_cfg.observations.policy.base_ang_vel.scale]*3 + 
                         [1]*3 +
                         joint_obs_scale +
                         [env_cfg.observations.policy.joint_vel.scale]*ac_dim +
                         [1]*ac_dim + 
                         [1]*4 +
                         [env_cfg.observations.policy.velocity_commands.scale]*3
                         )
    # print("obs_scale:", obs_scale)
    action_scale = np.array(action_scale)
    np.savetxt(f"{export_model_dir}/obScale.csv", obs_scale, delimiter=",", fmt="%.2f")
    np.savetxt(f"{export_model_dir}/acScale.csv", action_scale, delimiter=",", fmt="%.2f")

    policy_reload = torch.jit.load(f"{export_model_dir}/policy.pt")
    policy_reload.to(env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()
    print(f"[INFO] obs: {obs.shape}")
    timestep = 0
    start_idx = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping unwrapped.
            # if timestep <=250:
            #     env.unwrapped.command_manager._terms["base_velocity"].vel_command_b[:,0] = 0.
            #     env.unwrapped.command_manager._terms["base_velocity"].vel_command_b[:,1] = 0.
            #     env.unwrapped.command_manager._terms["base_velocity"].vel_command_b[:,2] = 0
            # elif timestep <= 500:
            #     env.unwrapped.command_manager._terms["base_velocity"].vel_command_b[:,0] = 0.6
            #     env.unwrapped.command_manager._terms["base_velocity"].vel_command_b[:,1] = 0.
            #     env.unwrapped.command_manager._terms["base_velocity"].vel_command_b[:,2] = 0
            # actions = policy(obs)
            actions = policy_reload(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            # print(f"[INFO] obs:", timestep)
            # print(f"[INFO] base: {obs[0, start_idx:start_idx+6]}")
            # print(f"[INFO] jpos: {obs[0, start_idx+6:start_idx+6+23]}")
            # # print(f"[INFO] jvel: {obs[0, start_idx+6+23:start_idx+6+23*2]}")
            # print(f"[INFO] actions: {obs[0, start_idx+6+23*2:start_idx+6+23*3]}")
            # print(f"[INFO] gait: {obs[0, start_idx+6+23*3:start_idx+6+23*3+4]}")
            # print(f"[INFO] cmd: {obs[0, start_idx+6+23*3+4:start_idx+6+23*3+7]}")
        timestep += 1
        # Exit the play loop after recording one video
        if args_cli.video:
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
