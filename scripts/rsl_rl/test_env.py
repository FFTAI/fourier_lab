"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
import numpy as np

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Test an RL env.")
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

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import GRX_humanoid.tasks
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    log_dir = '/home/fourier/GRX_humanoid/grxisaaclab/scripts/'
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
    env.seed(agent_cfg.seed)

    # reset environment
    obs = env.get_observations()
    if RECORD_DATA:
        print("obs.shape", obs.shape)
        obs_np = np.zeros((1 * int(env.max_episode_length), obs.shape[1]), dtype=float)
        print(f"Recording observations for {env.max_episode_length} timesteps.")
    timestep = 0
    upper_joint_ids = [11, 16, 12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30]
    wb_joint_ids = [0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 11, 16, 12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30]
    # simulate environment
    while simulation_app.is_running():
    # while timestep < env.max_episode_length:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = torch.zeros(env.num_envs, env.action_space.shape[1], device=env.device)
            # actions = torch.randn(env.num_envs, env.action_space.shape[1], device=env.device)
            # env stepping
            obs, _, _, _ = env.step(actions)
            input()
            if RECORD_DATA:
                obs_np[timestep] = obs[0,:].detach().cpu().numpy()
                upper_joint_cmd = env.unwrapped.command_manager.get_command("joint_pos_cmd")[0,:].detach().cpu().numpy()
                timestep += 1
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()
    if RECORD_DATA:
        # save the recorded data
        save_path = os.path.join(log_dir, "obs_data.npy")
        if os.path.exists(save_path):
            os.remove(save_path)  # 删除旧文件
            print(f"File {save_path} already exists. Overwriting it.")
        np.save(save_path, obs_np)
        print(f"Recorded observations saved to {save_path}")


if __name__ == "__main__":
    RECORD_DATA = False
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
