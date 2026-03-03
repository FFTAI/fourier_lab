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

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import GRX_humanoid.tasks

from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx, export_policy_as_jit
import isaaclab.utils.math as math_utils

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

    if EXPORT_POLICY:
        # export policy to onnx
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        # export_policy_as_onnx(policy_nn, export_model_dir, filename="policy.onnx")
        export_policy_as_jit(policy_nn, None, export_model_dir, filename="policy.pt")

    # reset environment
    obs = env.get_observations()
    # camera_obs = obs["depth_camera"]
    # depth_camera_img = camera_obs["front_camera_1"]
    # depth_raycaster_camera_img = camera_obs["front_camera_2"]
    # print("depth_camera_img.shape:", depth_camera_img.shape)
    # print("depth_raycaster_camera_img.shape:", depth_raycaster_camera_img.shape)
    if RECORD_DATA:
        print("obs.shape", obs.shape)
        # obs_np = np.zeros((1 * int(env.max_episode_length), obs.shape[1]), dtype=float)
        cmd_track_np = np.zeros((1 * int(env.max_episode_length), (3+4+16)*2), dtype=float)
        joint_states_np = np.zeros((1 * int(env.max_episode_length), 31*4), dtype=float)
        actions_np = np.zeros((1 * int(env.max_episode_length), 31), dtype=float)
        print(f"Recording observations for {env.max_episode_length} timesteps.")
    timestep = 0
    upper_joint_ids = [11, 16, 12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30]
    wb_joint_ids = [0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 11, 16, 12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30]
    stack_joint_ids = [23, 25, 27, 29, 24, 26, 28, 30]
    stack_joint_ids_4jPtarg = [20, 21, 22, 23, 27, 28, 29, 30]
    stack_command_ids = [5,6,7,8,12,13,14,15]
    # simulate environment
    # while simulation_app.is_running():
    while timestep < env.max_episode_length:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # clip actions 
            # actions[:, 18] = 0
            # actions[:, 25] = 0
            # actions[:, 19] = -0.4
            # actions[:, 26] = 0.4

            # env stepping
            obs, _, _, _ = env.step(actions)
            # input()
            # camera_obs = obs["depth_camera"]
            # depth_camera_img = camera_obs["front_camera_1"]
            # depth_raycaster_camera_img = camera_obs["front_camera_2"]
            # camera_img_draw = depth_camera_img[1,:].squeeze(2).detach().cpu().numpy()
            # raycaster_img_draw = depth_raycaster_camera_img[1,:].squeeze(2).detach().cpu().numpy()
            # print("camera_img_draw:", camera_img_draw[0:5, 0:5])
            # print("raycaster_img_draw:", raycaster_img_draw[0:5, 0:5])
            # plot_camera_image(camera_img_draw, raycaster_img_draw)
            #print(f"torso cmd: {env.unwrapped.command_manager.get_command('height_attitude')[0, 1:]}")
            if RECORD_DATA:
                # obs_np[timestep] = obs[0,:].detach().cpu().numpy()
                asset = env.unwrapped.scene["robot"]
                vel_cmd = env.unwrapped.command_manager.get_command("base_velocity")[0,:].detach().cpu().numpy()
                height_torso_cmd = env.unwrapped.command_manager.get_command("height_attitude")[0,:].detach().cpu().numpy()
                upper_joint_cmd = env.unwrapped.command_manager.get_command("joint_pos_cmd")[0,:].detach().cpu().numpy()
                base_lin_vel = asset.data.root_com_lin_vel_b[0,:].detach().cpu().numpy()
                base_ang_vel = asset.data.root_ang_vel_w[0,:].detach().cpu().numpy()
                base_height = asset.data.root_pos_w[0, :].detach().cpu().numpy()
                # torso/waist pitch link body ids = 9
                heading = asset.data.heading_w[0].detach().cpu().numpy()
                height_torso_cmd[1] += heading
                torso_orient = asset.data.body_link_quat_w[0, 9, :]
                torso_mat = math_utils.matrix_from_quat(torso_orient)
                torso_yaw = torch.atan2(-torso_mat[0, 1], torso_mat[1, 1]).detach().cpu().numpy()
                torso_roll = torch.asin(torso_mat[2, 1]).detach().cpu().numpy()
                torso_pitch = torch.atan2(-torso_mat[2, 0], torso_mat[2, 2]).detach().cpu().numpy()
                upper_joint_pos = asset.data.joint_pos[0, upper_joint_ids].detach().cpu().numpy()
                # concatenate all data into a single array
                cmd_track_np[timestep] = np.concatenate((vel_cmd, base_lin_vel[:2], np.array([base_ang_vel[2]]), 
                                                         height_torso_cmd, np.array([base_height[2]]), 
                                                         np.array([torso_yaw]), np.array([torso_roll]), np.array([torso_pitch]), 
                                                         upper_joint_cmd, upper_joint_pos), axis=0)
                joint_pos = asset.data.joint_pos[0, wb_joint_ids].detach().cpu().numpy()
                joint_vel = asset.data.joint_vel[0, wb_joint_ids].detach().cpu().numpy()
                joint_tor = asset.data.applied_torque[0, wb_joint_ids].detach().cpu().numpy()
                joint_pos_targ = (0.5*actions[0,:] + asset.data.default_joint_pos[0, wb_joint_ids]).detach().cpu().numpy()
                joint_pos_targ[stack_joint_ids_4jPtarg] += upper_joint_cmd[stack_command_ids] - asset.data.default_joint_pos[0, stack_joint_ids].detach().cpu().numpy()
                joint_states_np[timestep] = np.concatenate((joint_pos_targ, joint_pos, joint_vel, joint_tor), axis=0)
                actions_np[timestep] = actions[0,:].detach().cpu().numpy()
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
        save_path = os.path.join(log_dir, "joint_states.npy")
        save_path2 = os.path.join(log_dir, "cmd_track_data.npy")
        save_path3 = os.path.join(log_dir, "actions.npy")
        if os.path.exists(save_path):
            os.remove(save_path)  # 删除旧文件
            print(f"File {save_path} already exists. Overwriting it.")
        np.save(save_path, joint_states_np)
        if os.path.exists(save_path2):
            os.remove(save_path2)  # 删除旧文件
            print(f"File {save_path2} already exists. Overwriting it.")
        np.save(save_path2, cmd_track_np)
        if os.path.exists(save_path3):
            os.remove(save_path3)  # 删除旧文件
            print(f"File {save_path3} already exists. Overwriting it.")
        np.save(save_path3, actions_np)
        print(f"Recorded observations saved to {log_dir}")

# import matplotlib.pyplot as plt
# plt.ion()   # 打开交互模式（非阻塞）

# fig, ax = plt.subplots(1,2, figsize=(20,10))
def plot_camera_image(img1, img2):
    if img1 is None or img2 is None:
        return
    ax[0].clear()
    # ax.imshow(img.cpu().numpy(), cmap="plasma")  # 使用伪彩色
    # ax.imshow(img.cpu().numpy(), cmap="gray")  # 
    # ax[0].imshow(img1)  # 
    ax[0].imshow(img1, cmap="Greys")
    ax[0].set_title("Depth Camera Front")
    ax[0].axis("off")
    ax[1].clear()
    # ax[1].imshow(img2)  #
    ax[1].imshow(img2, cmap="Greys")  #
    ax[1].set_title("Depth Raycaster Camera")
    ax[1].axis("off")
    fig.canvas.draw()
    fig.canvas.flush_events()
    # plt.draw()
    plt.tight_layout()
    # plt.pause(0.001)

if __name__ == "__main__":
    # run the main execution
    EXPORT_POLICY = True
    RECORD_DATA = False
    main()
    # close sim app
    simulation_app.close()
