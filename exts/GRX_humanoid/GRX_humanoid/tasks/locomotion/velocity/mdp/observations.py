# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Callable

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera, ContactSensor

import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def get_signal_phase(env : ManagerBasedRLEnv, offset : float, cycle_time : float)-> torch.Tensor:
    """get sin signal phase -1~1 """
    # check if episode_length_buf is available
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    phase_l = (env.episode_length_buf *env.step_dt / cycle_time) % 1
    phase_r = (env.episode_length_buf *env.step_dt / cycle_time + offset) % 1
    return torch.stack([torch.sin(2.0*np.pi*phase_l),torch.sin(2.0*np.pi*phase_r)], dim = 1)

def base_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root xy-linear-velocity & z-angular-velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.cat([asset.data.root_lin_vel_b[..., :2], asset.data.root_ang_vel_b[..., 2:]], dim=-1).to(env.device)

def is_stand_bool(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The command vel is near zero."""
    return (torch.norm(env.command_manager.get_command(command_name)[:,0:3], dim=1) < 0.01)

def is_walk_bool(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The command vel is near zero."""
    return (torch.norm(env.command_manager.get_command(command_name)[:,0:3], dim=1) >= 0.01)

def is_stand_int(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The command vel is near zero."""
    return is_stand_bool(env, command_name).int().unsqueeze(-1)

def is_walk_int(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The command vel is near zero."""
    return is_walk_bool(env, command_name).int().unsqueeze(-1)

def joint_pos_cmd_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset relative to the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_cmd = env.command_manager.get_command("joint_pos_cmd")
    return joint_cmd - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

def command_enable(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The joint positions of the asset relative to the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    enable = env.command_manager.get_term(command_name).enable
    return enable

def joint_torque(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]

def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    power = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]*asset.data.joint_vel[:, asset_cfg.joint_ids])
    return power

def robot_material_properties(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """material properties of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    material_tensor = asset.root_physx_view.get_material_properties().to(device)
    # print("material_tensor",material_tensor.shape)
    return material_tensor.view(material_tensor.shape[0], -1)

def robot_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """contact force of the robot feet"""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    feet_contact_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    #print("foot force:",feet_contact_force_z)
    return feet_contact_force_z

  
def robot_feet_height(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """feet height of the robot"""
    sensor: RayCaster = env.scene[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height_w = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    feet_height = feet_height_w - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1).to(device) 
    #print("foot height:",feet_height)
    return feet_height.view(feet_height.shape[0], -1)

def get_gait_command_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the current gait phase as observation.

    The gait phase is represented by [sin(phase), cos(phase)] to ensure continuity.
    The phase is calculated based on the episode length and gait frequency.

    Returns:
        torch.Tensor: The gait phase observation. Shape: (num_envs, 2).
    """
    # check if episode_length_buf is available
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # Get the gait command from command manager
    command_term = env.command_manager.get_term("gait_command")
    # Calculate gait indices based on episode length
    gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * command_term.command[:, 0], 1.0)
    # Reshape gait_indices to (num_envs, 1)
    gait_indices = gait_indices.unsqueeze(-1)
    # Convert to sin/cos representation
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)


def get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current gait command parameters as observation.

    Returns:
        torch.Tensor: The gait command parameters [frequency, offset, duration].
                     Shape: (num_envs, 3).
    """
    return env.command_manager.get_command(command_name)

def get_height_attitude_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current H&A command parameters as observation.

    Returns:
        torch.Tensor: The H&A command parameters [height, pitch_angle].
                     Shape: (num_envs, 2).
    """
    return env.command_manager.get_command(command_name)

def get_behavior_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current Behavior command parameters as observation.

    Returns:
        torch.Tensor: The Behavior command parameters [height, pitch_angle, waist_angle, swing_height].
                     Shape: (num_envs, 4).
    """
    return env.command_manager.get_command(command_name)

def robot_joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.applied_torque.to(device)


def robot_joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint acc of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.joint_acc.to(device)


def robot_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_mass.to(device)


def robot_inertia(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """inertia of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inertia_tensor = asset.data.default_inertia.to(device)
    return inertia_tensor.view(inertia_tensor.shape[0], -1)


def robot_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint positions of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_pos.to(device)


def robot_joint_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint stiffness of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_stiffness.to(device)


def robot_joint_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint damping of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_damping.to(device)


def robot_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)


def robot_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """velocity of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_vel_w.to(device)

def robot_center_of_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """center of mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    link_mass = asset.root_physx_view.get_masses().to(device)
    body_com = asset.data.body_com_state_w[:,:,:3].to(device)
    body_com_pos = link_mass.unsqueeze(-1)*body_com
    robot_com_pos = torch.sum(body_com_pos, dim=1) / torch.sum(link_mass, dim=1, keepdim=True)
    return robot_com_pos

def robot_centroidal_momentum(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """centroidal momentum of the robot in world frame. first linear last angular. """
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    momentum_matrix = asset.root_physx_view.get_articulation_centroidal_momentum()[:,:,:-1].to(device)
    dof_vel = asset.data.joint_vel
    root_vel = asset.data.root_vel_w
    qd = torch.cat([root_vel,dof_vel],dim=-1)
    result = (momentum_matrix@qd.unsqueeze(2)).squeeze(2)
    return result

def robot_centroidal_momentum_except_root(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """centroidal momentum except root link of the robot in world frame. first linear last angular"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    momentum_matrix = asset.root_physx_view.get_articulation_centroidal_momentum()[:,:,:-1].to(device)
    dof_vel = asset.data.joint_vel
    root_vel = asset.data.root_vel_w
    qd_except_root = torch.cat([root_vel*0,dof_vel],dim=-1)
    result = (momentum_matrix@qd_except_root.unsqueeze(2)).squeeze(2)
    return result

def robot_centroidal_momentum_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """centroidal momentum of the robot in world frame. first linear last angular. """
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    momentum_matrix = asset.root_physx_view.get_articulation_centroidal_momentum()[:,:,:-1].to(device)
    dof_vel = asset.data.joint_vel
    root_vel = asset.data.root_vel_w
    # 除了asset_cfg.joint_ids之外的link速度置0
    dof_vel_masked = torch.zeros_like(dof_vel)
    dof_vel_masked[:, asset_cfg.joint_ids] = dof_vel[:, asset_cfg.joint_ids]
    dof_vel = dof_vel_masked
    qd = torch.cat([root_vel*0,dof_vel],dim=-1)
    result = (momentum_matrix@qd.unsqueeze(2)).squeeze(2)
    return result


def robot_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The contact forces of the body."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    body_contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    return body_contact_force.reshape(body_contact_force.shape[0], -1)

def is_feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check if the robot's feet are in contact with the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 30.0

def robot_base_link_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """mass of the base link of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_mass[:,0].unsqueeze(1).to(device)

def robot_feet_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """The contact forces of the body."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(asset.data.body_acc_w[:,asset_cfg.body_ids,:],dim=-1)

def get_camera_images(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,
                      update_hz: float = 10.0) -> torch.Tensor:
    """Get camera images from the specified camera sensor.

    Args:
        env (ManagerBasedRLEnv): The environment containing the camera sensor.
        sensor_cfg (SceneEntityCfg): Configuration of the camera sensor.

    Returns:
        torch.Tensor: The camera images. Shape: (num_envs, C, H, W).
    """
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _compute():
        depth_raw = camera.data.output["distance_to_camera"].to(device)
        return depth_raw
    
    key = f"front:{sensor_cfg.name}"
    return _rate_limited_depth(key, env, _compute, desired_hz=update_hz)

def _rate_limited_depth(key: str, env: ManagerBasedRLEnv, 
                        new_depth_provider: Callable[[], torch.Tensor], 
                        desired_hz: float = 10.0
) -> torch.Tensor:
    """Return depth updated at desired_hz by caching between calls.
    - key: unique string per stream (e.g., "front:depth_camera_front").
    - new_depth_provider: zero-arg callable that computes fresh normalized depth (N,H,Wc,1).
    - desired_hz: target update frequency.
    Uses env.step_dt to compute steps per update. If unavailable, updates every call.
    """
    # 初始化静态缓存
    if not hasattr(_rate_limited_depth, "_cache"):
        _rate_limited_depth._cache = {}
    cache = _rate_limited_depth._cache

    step_dt = getattr(env, "step_dt", None)
    # 无频率限制或无时间步长：每次调用都更新
    if step_dt is None or desired_hz <= 0:
        depth = new_depth_provider()
        cache[key] = {
            "last": depth,
            "tick": 0,
            "steps_per": 1,
            "shape": tuple(depth.shape),
            "step_dt": float(step_dt) if step_dt is not None else None,
            "hz": float(desired_hz),
        }
        return depth
    
    # 第一次或无状态：建立条目
    state = cache.get(key)
    if state is None:
        depth = new_depth_provider()
        steps_per = max(1, int(round(1.0 / (desired_hz * step_dt))))
        cache[key] = {
            "last": depth,
            "tick": 0,
            "steps_per": steps_per,
            "shape": tuple(depth.shape),
            "step_dt": float(step_dt),
            "hz": float(desired_hz),
        }
        return depth

    # recompute steps_per if timing changed
    steps_per = state.get("steps_per", 1)
    prev_dt = state.get("step_dt", step_dt)
    prev_hz = state.get("hz", desired_hz)
    if prev_dt is None or abs(prev_dt - step_dt) > 1e-9 or abs(prev_hz - desired_hz) > 1e-6:
        steps_per = max(1, int(round(1.0 / (desired_hz * step_dt))))
        state["steps_per"] = steps_per
        state["step_dt"] = float(step_dt)
        state["hz"] = float(desired_hz)

    # tick and decide update
    tick = int(state.get("tick", 0)) + 1
    state["tick"] = tick
    do_update = (tick % steps_per == 0)

    if do_update:
        depth = new_depth_provider()
        state["last"] = depth
        state["shape"] = tuple(depth.shape)
        return depth

    # shape mismatch -> recompute
    last = state.get("last")
    if last is None or tuple(last.shape) != state.get("shape"):
        depth = new_depth_provider()
        state["last"] = depth
        state["shape"] = tuple(depth.shape)
        return depth

    return last