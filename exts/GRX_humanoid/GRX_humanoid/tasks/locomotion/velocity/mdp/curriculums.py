"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from GRX_humanoid.tasks.locomotion.velocity.env.multi_stage_manager_based_rl_env import MultiStageManagerBasedRLEnv


def terrain_levels_vel(
    env: MultiStageManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), stage_threshold: int = -1
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = (distance > terrain.cfg.terrain_generator.size[0] / 2)&(env.stage >= stage_threshold)
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def modify_stage_cmd(
    env: MultiStageManagerBasedRLEnv, env_ids: Sequence[int]
):
    """Curriculum that modifies the command term based on the current stage of the environment.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the command term to modify.
        stage: The current stage of the environment.
    """
    # 所有系数暂时写定，之后应该用cfg传入
    # 要抄所有条件吗？
    # print(f"Step {env.common_step_counter}, stage: {env.stage}")
    if env.common_step_counter - env.prev_stage_update_step >= 1000 and env.common_step_counter > 0:
        # print(f"Stage {env.stage} update at step {env.common_step_counter}")
        env.prev_stage_update_step = env.common_step_counter
        # stage 1, vel command
        lin_vel_cfg = env.reward_manager.get_term_cfg("track_lin_vel_xy_exp")
        ang_vel_cfg = env.reward_manager.get_term_cfg("track_ang_vel_z_exp")
        avg_vel_rwd = torch.sum(lin_vel_cfg.func(env, **lin_vel_cfg.params) * lin_vel_cfg.weight)/ env.num_envs
        avg_ang_rwd = torch.sum(ang_vel_cfg.func(env, **ang_vel_cfg.params) * ang_vel_cfg.weight)/ env.num_envs
        cond1_flag = (avg_vel_rwd > 0.6 * lin_vel_cfg.weight) and (avg_ang_rwd > 0.6 * ang_vel_cfg.weight)
        print(f"Stage 1: {cond1_flag}, Avg Vel Rwd: {avg_vel_rwd}, Avg Ang Rwd: {avg_ang_rwd}")
        # stage 2, base height and pitch command
        # hip reward要不要？
        base_height_cfg = env.reward_manager.get_term_cfg("track_base_height_scanner_exp")
        torso_yaw_cfg = env.reward_manager.get_term_cfg("track_torso_yaw_exp")
        torso_roll_cfg = env.reward_manager.get_term_cfg("track_torso_roll_exp")
        torso_pitch_cfg = env.reward_manager.get_term_cfg("track_torso_pitch_exp")
        avg_hgt_rwd = torch.sum(base_height_cfg.func(env, **base_height_cfg.params) * base_height_cfg.weight)/ env.num_envs
        avg_torso_rot_rwd = 0.25*torch.sum(torso_yaw_cfg.func(env, **torso_yaw_cfg.params) * torso_yaw_cfg.weight)/ env.num_envs + \
                        torch.sum(torso_roll_cfg.func(env, **torso_roll_cfg.params) * torso_roll_cfg.weight)/ env.num_envs + \
                        2*torch.sum(torso_pitch_cfg.func(env, **torso_pitch_cfg.params) * torso_pitch_cfg.weight)/ env.num_envs
        cond2_flag = (avg_hgt_rwd > 0.6 * base_height_cfg.weight) and (avg_torso_rot_rwd > 0.9)
        print(f"Stage 2: {cond2_flag}, Avg Hgt Rwd: {avg_hgt_rwd}, Avg Torso Rwd: {avg_torso_rot_rwd}")
        # stage 3, joint position command
        # torso cmd呢？
        joint_pos_cfg = env.reward_manager.get_term_cfg("upper_joint_total_track")
        avg_jpos_rwd = torch.sum(joint_pos_cfg.func(env, **joint_pos_cfg.params) * joint_pos_cfg.weight)/ env.num_envs
        cond3_flag = (avg_jpos_rwd > 0.6 * joint_pos_cfg.weight)
        print(f"Stage 3: {cond3_flag}, Avg Joint Pos Rwd: {avg_jpos_rwd}")

        # print(f"Stage 1: {cond1_flag}, Stage 2: {cond2_flag}, Stage 3: {cond3_flag}")

        # 要不要加入stage回退机制，在进入stage但表现不好时启用？
        if env.stage == 1 and cond1_flag:
            # modify the command term to include base height and pitch
            base_cmd_cfg = env.command_manager.get_term_cfg("height_attitude")
            base_cmd_cfg.cfg.ranges.height = (0.8, 0.9)
            base_cmd_cfg.cfg.ranges.torso_yaw = (-0.5, 0.5)
            base_cmd_cfg.cfg.ranges.torso_roll = (-0.05, 0.05)
            base_cmd_cfg.cfg.ranges.torso_pitch = (-0.25, 0.5)
            base_cmd_cfg.task_progress = 0.3
            env.command_manager.set_term_cfg("height_attitude", base_cmd_cfg)
            torso_pos_rwd_cfg = env.reward_manager.get_term_cfg("torso_joint_pos")
            torso_pos_rwd_cfg.weight = 0.3
            env.reward_manager.set_term_cfg("torso_joint_pos", torso_pos_rwd_cfg)

            torso_rot_rwd_cfg = env.reward_manager.get_term_cfg("torso_orientation_exp")
            torso_rot_rwd_cfg.weight = 0.
            env.reward_manager.set_term_cfg("torso_orientation_exp", torso_rot_rwd_cfg)

            torso_yaw_rwd_cfg = env.reward_manager.get_term_cfg("track_torso_yaw_exp")
            torso_yaw_rwd_cfg.weight = 0.4
            env.reward_manager.set_term_cfg("track_torso_yaw_exp", torso_yaw_rwd_cfg)

            torso_roll_rwd_cfg = env.reward_manager.get_term_cfg("track_torso_roll_exp")
            torso_roll_rwd_cfg.weight = 0.4
            env.reward_manager.set_term_cfg("track_torso_roll_exp", torso_roll_rwd_cfg)

            torso_pitch_rwd_cfg = env.reward_manager.get_term_cfg("track_torso_pitch_exp")
            torso_pitch_rwd_cfg.weight = 0.4
            env.reward_manager.set_term_cfg("track_torso_pitch_exp", torso_pitch_rwd_cfg)
            # update the stage
            env.stage += 1
        elif env.stage == 2 and cond1_flag and cond2_flag:
            if env.command_manager.get_term("height_attitude").task_progress >= 0.89:
                env.command_manager.get_term("joint_pos_cmd").update_task_progress()
                action_rate_cfg = env.reward_manager.get_term_cfg("action_rate_l2")
                action_rate_cfg.weight = -0.1
                env.reward_manager.set_term_cfg("action_rate_l2", action_rate_cfg)
                upper_jnt_vel_cfg = env.reward_manager.get_term_cfg("upper_joint_stable")
                upper_jnt_vel_cfg.params["std"] = 2.0
                env.reward_manager.set_term_cfg("upper_joint_stable", upper_jnt_vel_cfg)
                # modify the command term to include joint position command
                env.stage += 1
            else:
                env.command_manager.get_term("height_attitude").update_task_progress()
        elif env.stage >= 3 and cond1_flag and cond2_flag and cond3_flag:
            env.command_manager.get_term("joint_pos_cmd").update_task_progress()
            # env.stage += 1
    return env.stage + (env.stage == 2)*env.command_manager.get_term("height_attitude").task_progress + (env.stage >= 3) * env.command_manager.get_term("joint_pos_cmd").task_progress

def modify_lower_stage_cmd(
    env: MultiStageManagerBasedRLEnv, env_ids: Sequence[int], height_range:tuple[float, float], pitch_range: tuple[float, float], yaw_range: tuple[float, float],
):
    """Curriculum that modifies the command term based on the current stage of the environment.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the command term to modify.
        stage: The current stage of the environment.
    """
    # 所有系数暂时写定，之后应该用cfg传入
    # 要抄所有条件吗？
    # print(f"Step {env.common_step_counter}, stage: {env.stage}")
    if env.common_step_counter - env.prev_stage_update_step >= 1000 and env.common_step_counter > 0:
        # print(f"Stage {env.stage} update at step {env.common_step_counter}")
        env.prev_stage_update_step = env.common_step_counter
        # stage 1, vel command
        lin_vel_cfg = env.reward_manager.get_term_cfg("track_lin_vel_xy_exp")
        ang_vel_cfg = env.reward_manager.get_term_cfg("track_ang_vel_z_exp")
        avg_vel_rwd = torch.sum(lin_vel_cfg.func(env, **lin_vel_cfg.params) * lin_vel_cfg.weight)/ env.num_envs
        avg_ang_rwd = torch.sum(ang_vel_cfg.func(env, **ang_vel_cfg.params) * ang_vel_cfg.weight)/ env.num_envs
        cond1_flag = (avg_vel_rwd > 0.5 * lin_vel_cfg.weight) and (avg_ang_rwd > 0.5 * ang_vel_cfg.weight)
        # stage 2, base height and pitch command
        # hip reward要不要？
        base_height_cfg = env.reward_manager.get_term_cfg("track_base_height_scanner_exp")
        base_pitch_cfg = env.reward_manager.get_term_cfg("flat_orientation_exp")
        avg_hgt_rwd = torch.sum(base_height_cfg.func(env, **base_height_cfg.params) * base_height_cfg.weight)/ env.num_envs
        avg_pitch_rwd = torch.sum(base_pitch_cfg.func(env, **base_pitch_cfg.params) * base_pitch_cfg.weight)/ env.num_envs
        cond2_flag = (avg_hgt_rwd > 0.5 * base_height_cfg.weight) and (avg_pitch_rwd > 0.4 * base_pitch_cfg.weight)

        # 要不要加入stage回退机制，在进入stage但表现不好时启用？
        if env.stage >= 1 and env.stage < 2 and cond1_flag and cond2_flag:
            # modify the command term to include base height and pitch
            command_term = env.command_manager.get_term("height_attitude")
            ranges = command_term.cfg.ranges
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.height = torch.clamp(torch.tensor(ranges.height, device=env.device) + delta_command,height_range[0],height_range[1],).tolist()
            ranges.pitch_angle = torch.clamp(torch.tensor(ranges.pitch_angle, device=env.device) + delta_command,pitch_range[0],pitch_range[1],).tolist()
            ranges.yaw_angle = torch.clamp(torch.tensor(ranges.yaw_angle, device=env.device) + delta_command,yaw_range[0],yaw_range[1],).tolist()
            env.stage += 0.1
        elif env.stage == 2 and cond1_flag and cond2_flag:
            # update the stage
            env.stage += 1
    return env.stage

def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight after some number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

def modify_push_force(
        env: ManagerBasedRLEnv, 
        env_ids: Sequence[int], 
        term_name: str, 
        max_velocity: Sequence[float], 
        interval: int, 
        starting_step: float = 0.0
        ):
    """Curriculum that modifies the maximum push (perturbation) velocity over some intervals. 

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        max_velocity: The maximum velocity of the push.
        interval: The number of steps after which the condition is checked again
        starting_step: The number of steps after which the curriculum is applied.
    """
    try:
        term_cfg = env.event_manager.get_term_cfg('push_robot')
    except:
        # print("No push_robot term found in the event manager")
        return 0.0
    curr_setting = term_cfg.params['velocity_range']['x'][1]
    if env.common_step_counter < starting_step:
        return curr_setting
    if env.common_step_counter % interval == 0:

        
        if torch.sum(env.termination_manager._term_dones["base_contact"]) < torch.sum(env.termination_manager._term_dones["time_out"]) * 2:
            # obtain term settings
            term_cfg = env.event_manager.get_term_cfg('push_robot')
            # update term settings
            curr_setting = term_cfg.params['velocity_range']['x'][1]
            curr_setting = np.clip(curr_setting * 1.5, 0.0, max_velocity[0])
            term_cfg.params['velocity_range']['x'] = (-curr_setting, curr_setting)
            curr_setting = term_cfg.params['velocity_range']['y'][1]
            curr_setting = np.clip(curr_setting * 1.5, 0.0, max_velocity[1])
            term_cfg.params['velocity_range']['y'] = (-curr_setting, curr_setting)
            env.event_manager.set_term_cfg('push_robot', term_cfg)
        

        if torch.sum(env.termination_manager._term_dones["base_contact"]) > torch.sum(env.termination_manager._term_dones["time_out"]) / 2:
            # obtain term settings
            term_cfg = env.event_manager.get_term_cfg('push_robot')
            # update term settings
            curr_setting = term_cfg.params['velocity_range']['x'][1]
            curr_setting = np.clip(curr_setting - 0.2, 0.0, max_velocity[0])
            term_cfg.params['velocity_range']['x'] = (-curr_setting, curr_setting)
            curr_setting = term_cfg.params['velocity_range']['y'][1]
            curr_setting = np.clip(curr_setting - 0.2, 0.0, max_velocity[1])
            term_cfg.params['velocity_range']['y'] = (-curr_setting, curr_setting)
            env.event_manager.set_term_cfg('push_robot', term_cfg)

    return curr_setting


def modify_command_velocity(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    term_name: str, 
    max_velocity: Sequence[float], 
    interval: int, 
    starting_step: float = 0.0
    ):
    """Curriculum that modifies the maximum command velocity over some intervals. 

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        max_velocity: The maximum velocity. 
        interval: The number of steps after which the condition is checked again
        starting_step: The number of steps after which the curriculum is applied.
    """

    command_cfg = env.command_manager.get_term('base_velocity').cfg
    curr_lin_vel_x = command_cfg.ranges.lin_vel_x

    if env.common_step_counter < starting_step:
        return curr_lin_vel_x[1]
    
    if env.common_step_counter % interval == 0:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        rew = env.reward_manager._episode_sums[term_name][env_ids]
        if torch.mean(rew) / env.max_episode_length > 0.8 * term_cfg.weight * env.step_dt:
            curr_lin_vel_x = (
                np.clip(curr_lin_vel_x[0] - 0.5, max_velocity[0], 0.), 
                np.clip(curr_lin_vel_x[1] + 0.5, 0., max_velocity[1])
            )
            command_cfg.ranges.lin_vel_x = curr_lin_vel_x

    return curr_lin_vel_x[1]

def modify_base_velocity_range(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, mod_range: dict, num_steps: int
):
    """
    Modifies the range of a command term (e.g., base_velocity) in the environment after a specific number of steps.

    Args:
        env: The environment instance.
        term_name: The name of the command term to modify (e.g., "base_velocity").
        end_range: The target range for the term (e.g., {"lin_vel_x": (-1.5, 1.5), "ang_vel_z": (-1.5, 1.5)}).
        activation_step: The step count after which the range modification is applied.
    """
    # Check if the curriculum step exceeds the activation step
    if env.common_step_counter >= num_steps:
        # Get the term object
        command_term = env.command_manager.get_term(term_name)

        # Update the ranges directly
        for key, target_range in mod_range.items():
            if hasattr(command_term.cfg.ranges, key):
                setattr(command_term.cfg.ranges, key, target_range)
