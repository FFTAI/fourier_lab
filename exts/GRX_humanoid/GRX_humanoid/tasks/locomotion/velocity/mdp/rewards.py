from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING
from collections import deque

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_error_magnitude
from GRX_humanoid.utils.math import *
from .observations import *
from torch import distributions
from isaaclab.managers import RewardTermCfg as RewTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold_min: float,
                  threshold_max: float) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # negative reward for small steps
    air_time = (last_air_time - threshold_min) * first_contact
    # no reward for large steps
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    # no reward for zero command
    reward *= is_walk_bool(env, "base_velocity")
    return reward

def feet_slide_penalty(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def flat_link_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset = env.scene[asset_cfg.name]
    link_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    projected_gravity_link = math_utils.quat_apply_inverse(link_quat_w , asset.data.GRAVITY_VEC_W)

    return torch.sum(torch.square(projected_gravity_link[:, :2]), dim=1)


def foot_yaw_alignment_reward(
    env: 'ManagerBasedRLEnv',
    asset_cfg: 'SceneEntityCfg',
    max_angle: float,
    always_active: bool = True
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    _, _, base_yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)

    rewards = []
    for foot_pattern in asset_cfg.body_ids:
        
        foot_quat = asset.data.body_quat_w[:, foot_pattern, :]
        
        _, _, foot_yaw = math_utils.euler_xyz_from_quat(foot_quat)
        
        angle_diff = torch.abs(base_yaw - foot_yaw)
        
        angle_diff = torch.min(angle_diff, 2 * math.pi - angle_diff)
        angle_diff = angle_diff * (180 / math.pi)

        reward = torch.exp(-(angle_diff**2) / (2 * (max_angle / 3)**2))
        rewards.append(reward)
    if(always_active):
        return torch.mean(torch.stack(rewards, dim=1), dim=1)
    else:
        return torch.mean(torch.stack(rewards, dim=1), dim=1)*is_walk_bool(env, "base_velocity")


def track_base_height_exp(env: ManagerBasedRLEnv, target_height: float, std: float, sensor_cfg: SceneEntityCfg, \
asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 10.0
    mask = ~in_contact[:,0] & ~in_contact[:,1]
    in_contact[mask] = True
    foot_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    support_z = (in_contact * foot_pos_z).sum(dim=1) / (in_contact.sum(dim=1) + 1e-6)
    
    baselink_pos_z = asset.data.root_pos_w[:, 2]
    relative_height = baselink_pos_z - support_z
    reward = torch.exp(-(relative_height - target_height)**2/(std**2))
    return reward

def track_base_height_scanner_exp(env: ManagerBasedRLEnv, target_height: float, std: float, sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"), \
asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    sensor = env.scene[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    relative_height = asset.data.root_pos_w[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    reward = torch.exp(-(relative_height - target_height)**2/(std**2))
    return reward

def track_base_height_scanner_exp_ha(env: ManagerBasedRLEnv, std: float, sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"), \
asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    sensor = env.scene[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    relative_height = asset.data.root_pos_w[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    term = (relative_height - env.command_manager.get_command("height_attitude")[:,0])**2/(std**2)
    reward = torch.exp(-term) - 0.003*term
    return reward

def track_base_foot_yaw_exp_ha(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    left_quat_yaw = math_utils.yaw_quat(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :])
    right_quat_yaw = math_utils.yaw_quat(asset.data.body_quat_w[:, asset_cfg.body_ids[1], :])
    foot_quat_yaw = quaternion_mean_simple(left_quat_yaw, right_quat_yaw)
    base_quat_yaw = math_utils.yaw_quat(asset.data.root_quat_w)
    foot_quat_delta = math_utils.quat_box_minus(left_quat_yaw, right_quat_yaw)[:,2]
    base_quat_delta = math_utils.quat_box_minus(base_quat_yaw, foot_quat_yaw)[:,2]
    reward = 0.5*torch.exp(-torch.square(base_quat_delta-env.command_manager.get_command("height_attitude")[:,2])/std**2) \
        +0.5*torch.exp(-torch.square(foot_quat_delta)/std**2)
    reward[is_walk_bool(env, "base_velocity")] = 0
    return reward

def minimize_CoT_reward(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)-> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_abs = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    tau_abs = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    joint_power = torch.sum(vel_abs*tau_abs,dim=1)
    base_vel = torch.norm(asset.data.root_lin_vel_b[:,:2],dim=1)
    base_vel[base_vel < 0.1] = 0.1
    Mgv = base_vel * torch.sum(asset.root_physx_view.get_masses(), dim=1).to(env.device) * 9.8
    CoT = joint_power / Mgv - 0.3
    CoT[CoT < 0] = 0
    reward = torch.exp(-torch.square(CoT)/std**2)
    return reward

def get_phase(env: ManagerBasedRLEnv, offset : float, cycle_time : float)-> torch.Tensor:
    """get time phase 0~1 """
    phase = (env.episode_length_buf * env.step_dt / cycle_time + offset) % 1
    return phase

def get_gait_phase(env: ManagerBasedRLEnv, offset : float, cycle_time : float, std : float)-> torch.Tensor:
    """get gait phase 0~1 via cdf"""
    Mean = torch.tensor([0.0],device=env.device)  # 均值
    Std = torch.tensor([1.0],device=env.device)   # 标准差
    normal_dist = torch.distributions.Normal(Mean, Std)
    cdf1 = normal_dist.cdf(get_phase(env,offset,cycle_time)/std)
    cdf2 = normal_dist.cdf((get_phase(env,offset,cycle_time)-0.5)/std)
    cdf3 = normal_dist.cdf((get_phase(env,offset,cycle_time)-1)/std)
    cdf4 = normal_dist.cdf((get_phase(env,offset,cycle_time)-1.5)/std)
    return cdf1*(1-cdf2)+cdf3*(1-cdf4)

def gait_reward(env: ManagerBasedRLEnv, offset : float, cycle_time : float, vel_std : float, force_std : float, \
sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))-> torch.Tensor:
    #phase compute
    contact_phase = torch.stack([get_gait_phase(env,0,cycle_time,0.05),get_gait_phase(env,offset,cycle_time,0.05)],dim=1)
    swing_phase = 1 - contact_phase
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    force_term = torch.sum(swing_phase * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),dim=1)
    vel_term = torch.sum(contact_phase * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :3],dim=-1),dim=1)
    reward = 0.5*torch.exp(-torch.square(vel_term)/vel_std**2) + 0.5*torch.exp(-torch.square(force_term)/force_std**2)
    reward[is_stand_bool(env,"base_velocity")] = 1.0
    return reward


def flat_link_orientation_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset = env.scene[asset_cfg.name]
    link_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    grav = asset.data.GRAVITY_VEC_W.unsqueeze(1).repeat(1,len(asset_cfg.body_ids),1)
    projected_gravity_link = math_utils.quat_apply_inverse(link_quat_w , grav)
    term = -torch.sum(torch.square(projected_gravity_link[:,:,:2]),dim=-1)/std**2
    reward = torch.sum(torch.exp(term), dim=-1)
    return reward

def flat_link_orientation_yaw_frame_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset = env.scene[asset_cfg.name]
    link_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    root_quat_w = asset.data.root_quat_w
    root_yaw_quat = yaw_quat(root_quat_w)
    quat_err = torch.square(link_quat_w - root_yaw_quat)
    return torch.exp(-torch.sum(quat_err/std**2, dim=-1))

def flat_link_rpy_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset = env.scene[asset_cfg.name]
    link_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    root_quat_w = asset.data.root_quat_w
    euler_xyz = math_utils.euler_xyz_from_quat(link_quat_w.squeeze(1))
    root_euler_xyz = math_utils.euler_xyz_from_quat(root_quat_w)
    rpy_err = torch.zeros((env.num_envs, 3), device=env.device)
    rpy_err[:,0] = euler_xyz[0]
    rpy_err[:,1] = euler_xyz[1]
    # rpy_err[:,1] = euler_xyz[1] - root_euler_xyz[1]
    rpy_err[:,2] = euler_xyz[2] - root_euler_xyz[2]
    return torch.exp(-torch.sum(torch.square(rpy_err)/std**2, dim=-1))

def flat_link_roll_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset = env.scene[asset_cfg.name]
    link_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    grav = asset.data.GRAVITY_VEC_W.unsqueeze(1).repeat(1,len(asset_cfg.body_ids),1)
    projected_gravity_link = math_utils.quat_apply_inverse(link_quat_w , grav)
    term = -1.0*is_walk_bool(env, "base_velocity").unsqueeze(-1)*torch.square(projected_gravity_link[:,:,1])/std**2
    reward = torch.sum(torch.exp(term), dim=-1)
    return reward

def flat_orientation_l2_weight(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    grav_error = torch.square(asset.data.projected_gravity_b[:, :2])
    return grav_error[:, 0] + 4 * grav_error[:, 1]

def flat_orientation_l2_ha(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    R_des = math_utils._axis_angle_rotation('Y',env.command_manager.get_command("height_attitude")[:,1])
    quat_des = math_utils.quat_from_matrix(R_des)
    grav_des = math_utils.quat_apply_inverse(quat_des, asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2] - grav_des[:, :2]), dim=1)

def flat_orientation_exp(env: ManagerBasedRLEnv, std : float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    return torch.exp(-torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)/std**2)

def flat_orientation_exp_ha(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    R_des = math_utils._axis_angle_rotation('Y',env.command_manager.get_command("height_attitude")[:,1])
    quat_des = math_utils.quat_from_matrix(R_des)
    grav_des = math_utils.quat_apply_inverse(quat_des, asset.data.GRAVITY_VEC_W)
    grav_err = torch.square(asset.data.projected_gravity_b[:, :2] - grav_des[:, :2])
    quat_err = -torch.clamp(grav_err[:,0],-0.3, 0.3)-3.0*grav_err[:,1]
    return torch.exp(quat_err/std**2)

def track_feet_height_exp(env: ManagerBasedRLEnv, target_height: float, std: float, sensor_cfg: SceneEntityCfg, \
asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > torch.sum(asset.data.default_mass, dim=1).to(env.device).unsqueeze(-1) * 9.8 * 0.5
    mask = ~in_contact[:,0] & ~in_contact[:,1]
    in_contact[mask] = True
    foot_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    support_z = (in_contact * foot_pos_z).sum(dim=1) / (in_contact.sum(dim=1) + 1e-6)
    swing_z = (~in_contact * foot_pos_z).sum(dim=1) / (in_contact.sum(dim=1) + 1e-6)
    
    relative_height = swing_z - support_z
    # print(relative_height)
    reward = torch.exp(-(relative_height - target_height)**2/(std**2))
    reward[is_stand_bool(env,"base_velocity")] = 1.0
    return reward


def track_feet_height_exp_gait(env: ManagerBasedRLEnv, offset : float, cycle_time : float, target_height: float, std: float,\
sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))-> torch.Tensor:
    contact_phase = torch.stack([get_gait_phase(env,0,cycle_time,0.05),get_gait_phase(env,offset,cycle_time,0.05)],dim=1)
    swing_phase = 1 - contact_phase

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 100.0
    mask = ~in_contact[:,0] & ~in_contact[:,1]
    in_contact[mask] = True
    foot_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    support_z = (in_contact * foot_pos_z).sum(dim=1) / (in_contact.sum(dim=1) + 1e-6)
    swing_z = (~in_contact * foot_pos_z).sum(dim=1) / (in_contact.sum(dim=1) + 1e-6)
    
    target_height_gait = target_height * torch.max(~in_contact * swing_phase, dim=1)[0]
    # print(target_height_gait)
    relative_height = swing_z - support_z 
    reward = torch.exp(-(relative_height - target_height)**2/(std**2))
    reward[is_stand_bool(env,"base_velocity")] = 1.0
    return reward

def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalize joint positions that deviate from the default one.
    l2 l2 l2
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)

def link_lin_z_acc_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using exp."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-torch.mean(torch.square(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, 2],), dim=1) / std**2)

def link_lin_z_over_acc_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using exp."""
    asset: Articulation = env.scene[asset_cfg.name]
    acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, 2] - 2.0
    acc[acc<0]=0
    return torch.exp(-torch.mean(torch.square(acc), dim=1) / std**2)

def track_lin_vel_xy_yaw_frame_exp(env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_link_quat_w), asset.data.root_com_lin_vel_w[:, :3])
    lin_vel_error = torch.norm(env.command_manager.get_command(command_name)[:, :2]*is_walk_bool(env,"base_velocity").unsqueeze(-1) - vel_yaw[:, :2], dim=1)
    return torch.where(is_walk_bool(env,"base_velocity"),torch.exp(-torch.square(lin_vel_error) / std**2),torch.exp(-torch.abs(lin_vel_error) / std))

def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2]*is_walk_bool(env,"base_velocity") - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def track_link_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2]*is_walk_bool(env,"base_velocity") - asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], 2])
    return torch.exp(-ang_vel_error / std**2)

def link_yaw_alignment_reward(
    env: 'ManagerBasedRLEnv',
    asset_cfg: 'SceneEntityCfg',
    max_angle: float
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    _, _, base_yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)

    rewards = []
    for link_pattern in asset_cfg.body_ids:
        
        link_quat = asset.data.body_quat_w[:, link_pattern, :]
        
        _, _, link_yaw = math_utils.euler_xyz_from_quat(link_quat)
        
        angle_diff = torch.abs(base_yaw - link_yaw)
        
        angle_diff = torch.min(angle_diff, 2 * math.pi - angle_diff)
        angle_diff = angle_diff * (180 / math.pi)

        reward = torch.exp(-(angle_diff**2) / (2 * (max_angle / 3)**2))
        rewards.append(reward)

    return torch.mean(torch.stack(rewards, dim=1), dim=1)

def link_distance(env: ManagerBasedRLEnv,  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min: float = 0.2, max: float = 0.8) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    foot_distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    distance_min = torch.clamp(foot_distance - min, -0.5, 0.)
    distance_max = torch.clamp(foot_distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(distance_min) * 100) + torch.exp(-torch.abs(distance_max) * 100)) / 2

class GaitReward(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.vel_command_name = cfg.params["vel_command_name"]
        self.dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
        command_name,
        vel_command_name,
        sensor_cfg,
        asset_cfg,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name)

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward
        # foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1)
        foot_forces = torch.abs(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids, 2])
        # print("A:",foot_forces.shape)
        # print("B:",(torch.abs(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids, 2])).shape)
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        # foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids], dim=-1)
        foot_velocities = torch.abs(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, 2])
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # Combine rewards
        reward = force_reward + velocity_reward
        reward *= (torch.norm(env.command_manager.get_command(vel_command_name)[:, :2], dim=1)  + torch.abs(env.command_manager.get_command(vel_command_name)[:, 2]))> 0.1
        return reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all((offsets >= 0) & (offsets <= 1)), "Offsets must be between 0 and 1"
        assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

        gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [gait_indices.view(self.num_envs, 1), (gait_indices + offsets + 1).view(self.num_envs, 1)],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
            0.5 / (1 - durations[swing_idxs])
        )

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

        return desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-forces[:, i] ** 2 / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute velocity-based reward component."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * (1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale

class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def ang_vel_xy_world_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_w[:, :2] # only penalize roll and pitch
    # return torch.sum(torch.square(ang_vel), dim=1)
    return 0.5*torch.square(ang_vel[:,0]) + torch.square(ang_vel[:,1])

def body_ang_vel_xy_world_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 1
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :2] # only penalize roll and pitch
    return torch.sum(torch.square(ang_vel), dim=1)

def body_ang_vel_xy_world_l2_ha(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 1
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :2] # only penalize roll and pitch
    if torch.sum(env.stage) > 1:
        straight_flag = env.command_manager._terms['height_attitude'].is_straight_env
        ang_vel = ang_vel * straight_flag.unsqueeze(-1)
    return torch.sum(torch.square(ang_vel), dim=1)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward

def torque(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # torque_over = (torch.abs(asset.data.computed_torque) - 0.5*asset.data.joint_effort_limits).clamp(min=0.)
    torque_over = (torch.abs(asset.data.applied_torque) - 0.5*asset.data.joint_effort_limits).clamp(min=0.)
    torque_over_relative = (torque_over / (0.5 * asset.data.joint_effort_limits)).clamp(max=5.)
    return torch.mean(torch.square(torque_over_relative), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)

def contact_moment(env: ManagerBasedRLEnv, std: float, threshold: float, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contact_force = torch.clamp(torch.abs(contact_sensor.data.net_forces_w[:,sensor_cfg.body_ids, 2]) - 150.0, min=0)
    contact_vel = torch.clamp(torch.abs(torch.clamp(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2],max=0.0)) - 0.05, min=0)
    contact_moment = torch.clamp(torch.sum(contact_force*contact_vel,dim=-1) - 20.0, min=0, max=threshold)
    term = torch.square(contact_moment) / std**2
    return 1.0 - torch.exp(-term) + 0.003 * term

def contact_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contact_mask = torch.norm(contact_sensor.data.net_forces_w[:,sensor_cfg.body_ids, 0:3], dim=2) > 5.
    contact_vel = asset.data.body_lin_vel_w[:,asset_cfg.body_ids, 0:3] * contact_mask.unsqueeze(-1)
    return torch.sum(torch.square(contact_vel), dim=(1,2))

def joint_symmetry(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids[0]] - asset.data.default_joint_pos[:, asset_cfg.joint_ids[0]] + asset.data.joint_pos[:, asset_cfg.joint_ids[1]] - asset.data.default_joint_pos[:, asset_cfg.joint_ids[1]])
    return torch.exp(-joint_error / std**2)


def fly(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, -5:, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5

def torso_joint_pos_l2(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize torso joint position deviation from the default position."""
    asset: Articulation = env.scene[asset_cfg.name]
    torso_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    pos_err = torso_joint_pos - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    pos_term = torch.sum(torch.square(pos_err), dim=1)
    return torch.exp(-pos_term / std**2) 

def torso_joint_pos_l2_with_roll_balance(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize torso joint position deviation from the default position."""
    asset: Articulation = env.scene[asset_cfg.name]
    torso_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    pos_err = torso_joint_pos - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    pos_err[:,0] *= 0.5   # waist yaw 惩罚轻一些
    pos_err[:,1] = 0   # allow waist roll swing
    pos_term = torch.sum(torch.square(pos_err), dim=1)
    return pos_term
    # return torch.exp(-pos_term / std**2) 

def torso_joint_pos_l2_with_squat(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize torso joint position deviation from the default position."""
    asset: Articulation = env.scene[asset_cfg.name]
    torso_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    pos_err = torso_joint_pos - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    if torch.sum(env.stage) > 1:
        ha_cmd = env.command_manager.get_command("height_attitude").clone()  # [N, 3] ZXY, only roll and pitch angles
        # pos_err[:,1] = 0
        out_limit_pitch_cmd = ha_cmd[:,3] > 0.52
        pos_err = torso_joint_pos - ha_cmd[:, 1:4]
        pos_err[out_limit_pitch_cmd, 2] = torso_joint_pos[out_limit_pitch_cmd, 2] - 0.52
        pos_err[:,2] *= 0.25
    pos_term = torch.sum(torch.square(pos_err), dim=1)
    return torch.exp(-pos_term / std**2) 

def torso_joint_vel_l2(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize torso joint velocity deviation from the default position."""
    asset: Articulation = env.scene[asset_cfg.name]
    torso_joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    vel_term = torch.sum(torch.square(torso_joint_vel), dim=1)
    return torch.exp(-vel_term / std**2)

def torso_joint_vel_l2_with_roll_balance(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize torso joint velocity deviation from the default position."""
    asset: Articulation = env.scene[asset_cfg.name]
    torso_joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    torso_joint_vel[:,0] *= 0.25 # waist yaw 惩罚轻一些
    torso_joint_vel[:,1] = 0   # allow waist roll swing
    vel_term = torch.sum(torch.square(torso_joint_vel), dim=1)
    return vel_term
    # return torch.exp(-vel_term / std**2)

def base_orientation_exp(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-(torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1))/std**2)

def base_orientation_exp_roll_balance(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    rot_mat = math_utils.matrix_from_quat(quat)
    roll = torch.asin(rot_mat[:, 2, 1])
    pitch = torch.atan2(-rot_mat[:, 2, 0], rot_mat[:, 2, 2])
    flag = is_walk_bool(env,"base_velocity")
    std = std * torch.where(flag, 1.0, 0.5)
    roll_scale = 1.0 * torch.where(flag, 0.25, 1.0)
    return torch.exp(-(roll**2 * roll_scale + pitch**2)/std**2)

def base_orientation_exp_with_squat(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    rot_mat = math_utils.matrix_from_quat(quat)
    roll = torch.asin(rot_mat[:, 2, 1])
    pitch = torch.atan2(-rot_mat[:, 2, 0], rot_mat[:, 2, 2])
    flag = is_walk_bool(env,"base_velocity")
    std = std * torch.where(flag, 1.0, 0.5)
    torso_cmd = env.command_manager.get_command("height_attitude").clone()  # [N, 4] height and ZXY
    ## des_pitch 与 height cmd相关，鼓励下蹲时base pitch前倾, 最大0.5？
    height_mask = torso_cmd[:, 0] < 0.7
    des_pitch = torch.zeros_like(torso_cmd[:,0])
    des_pitch[height_mask] = -2*torso_cmd[height_mask,0] + 1.4  # height
    roll_mask = torch.abs(torso_cmd[:, 2]) < 0.1
    pitch_mask = torch.abs(torso_cmd[:, 3]) < 0.1
    return torch.exp(-(roll**2 * roll_mask + (pitch-des_pitch)**2 * pitch_mask)/std**2)
    # return torch.exp(-((roll*0.5)**2 * roll_mask + pitch**2 * pitch_mask * height_mask)/std**2)

# 期望torso欧拉角： ZXY， 旋转矩阵右乘计算：Z(y)X(a)Y(b)，其中yaw角是身体系，roll and pitch是地面系
# 期望欧拉角得旋转矩阵，旋转矩阵得期望torso四元数； 获取现有torso四元数
def track_torso_yaw(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Track the yaw angle of the torso."""
    asset: Articulation = env.scene[asset_cfg.name]
    torso_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    torso_mat = math_utils.matrix_from_quat(torso_quat)
    torso_yaw = torch.atan2(-torso_mat[:, 0, 1], torso_mat[:, 1, 1])
    yaw_cmd = env.command_manager.get_command("height_attitude")[:, 1].clone()  # [N, 1] ZXY, only yaw angle
    heading = asset.data.heading_w
    yaw_cmd += heading  # add the heading angle to the yaw command
    yaw_err = math_utils.wrap_to_pi(yaw_cmd - torso_yaw)
    return torch.exp(-(yaw_err/std)**2)

def track_torso_roll(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor: 
    """Track the roll angle of the torso."""
    asset: Articulation = env.scene[asset_cfg.name]
    torso_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    torso_mat = math_utils.matrix_from_quat(torso_quat)
    torso_roll = torch.asin(torso_mat[:, 2, 1])
    roll_cmd = env.command_manager.get_command("height_attitude")[:, 2].clone()  # [N, 1] ZXY, only roll angle
    roll_err = roll_cmd - torso_roll
    # waist roll pos 不应与euler roll cmd 反向
    waist_roll_pos = asset.data.joint_pos[:, asset_cfg.joint_ids[0]] - asset.data.default_joint_pos[:, asset_cfg.joint_ids[0]]
    waist_roll_pos_err = roll_cmd - waist_roll_pos
    same_sign = (roll_cmd * waist_roll_pos) > 0
    roll_err = torch.where(same_sign, roll_err, abs(roll_err) + abs(waist_roll_pos_err)*(abs(waist_roll_pos_err)>0.1))
    return torch.exp(-(roll_err/std)**2)

def track_torso_pitch(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Track the pitch angle of the torso."""
    asset: Articulation = env.scene[asset_cfg.name]
    torso_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    torso_mat = math_utils.matrix_from_quat(torso_quat)
    torso_pitch = torch.atan2(-torso_mat[:, 2, 0], torso_mat[:, 2, 2])
    pitch_cmd = env.command_manager.get_command("height_attitude")[:, 3].clone()  # [N, 1] ZXY, only pitch angle
    pitch_err = pitch_cmd - torso_pitch
    return torch.exp(-(pitch_err/std)**2)

def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg, delta: float = 0.0) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    contact_time[contact_time>(threshold+delta)] = 0
    air_time[air_time>(threshold-delta)] = 0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    reward = torch.mean(in_mode_time, dim=-1)
    # no reward for zero command
    reward *= is_walk_bool(env,"base_velocity")
    return reward

def feet_air_time_symmetry_biped(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = 0.5*torch.abs(air_time[:,0] - air_time[:,1]).clamp(max=0.2) + 0.5*torch.abs(contact_time[:,0] - contact_time[:,1]).clamp(max=0.2)
    reward *= is_walk_bool(env,"base_velocity")
    return reward

def feet_air_time_positive_biped_interp(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    vel_cmd = torch.norm(env.command_manager.get_command(command_name)[:,0:3], dim=1)
    bp = [0.2, 1.0]
    slopes = [0.25, 0, 0.1]
    intercepts = [0.4, 0.45, 0.35]
    threshold_vel = piecewise_linear(vel_cmd, bp, slopes, intercepts)
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    in_mode_time[in_mode_time> threshold_vel.unsqueeze(1)] = 0
    reward = torch.mean(in_mode_time, dim=-1)
    reward[reward>threshold_vel] = 0
    # no reward for zero command
    reward *= is_walk_bool(env,"base_velocity")
    return reward

def feet_slide(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids], asset.data.body_com_ang_vel_w[:, asset_cfg.body_ids])[:,:,2]
    lin_vel = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids], asset.data.body_lin_vel_w[:, asset_cfg.body_ids])[:,:,0:2]
    body_vel = torch.cat([ang_vel.unsqueeze(-1),lin_vel],dim=-1)
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1) - 0.02
    reward[reward<0] = 0
    return reward


def body_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def feet_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_reward: float = 400, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    threshold =  1.2*torch.sum(asset.root_physx_view.get_masses(), dim=1).to(env.device).unsqueeze(-1) * 9.8
    reward -= threshold
    reward[reward < 0] = 0
    reward = reward.sum(dim=-1)
    reward = reward.clamp(min=0, max=max_reward)
    return reward

def center_of_mass(
        env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 希望质心投影处于双脚连线中心 """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    com_pos = robot_center_of_mass(env, asset_cfg)
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    feet_pos_center = (feet_pos[:, 0] + feet_pos[:, 1]) / 2
    quat = asset.data.root_quat_w
    disp_vec = torch.tensor([0.02, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs,1)
    disp_vec_rotated = math_utils.quat_apply_yaw(quat, disp_vec)
    feet_pos_center += disp_vec_rotated
    com_xy_error = torch.norm((com_pos - feet_pos_center)[:,:2], dim=-1)*is_stand_bool(env,"base_velocity")
    return torch.exp(-torch.square(com_xy_error)/std**2)

def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

def joint_pos_total_track(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos_error = (asset.data.joint_pos[:, asset_cfg.joint_ids] - env.command_manager.get_command("joint_pos_cmd"))
    if torch.sum(env.stage) == 2:
        joint_cmd_enable = env.command_manager.get_term("joint_pos_cmd").enable
        # joint_cmd_enable (num_envs, 2)， 分别表示左臂和右臂是否有命令，pos_error (num_envs, num_upper_joint_cmd)，只计算有命令的关节误差
        # enable = [1,1], 双臂都有命令， enable = [1,0] 左臂有命令, [0,1], 右臂有命令， enable = [0,0], 无命令
        # upper_joint_cmd: head 2 joints, 7 left arm joints, 7 right arm joints
        enable = torch.ones_like(pos_error)
        enable[:,2:9] = joint_cmd_enable[:,0].unsqueeze(-1).repeat(1,7)
        enable[:,9:16] = joint_cmd_enable[:,1].unsqueeze(-1).repeat(1,7)
        pos_error = pos_error * enable
    pos_term = torch.sum(torch.square(pos_error), dim=1)/std**2
    return torch.exp(-pos_term)

def joint_pos_track(env: ManagerBasedRLEnv, std: float, weight: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos_error = (asset.data.joint_pos[:, asset_cfg.joint_ids] - env.command_manager.get_command("joint_pos_cmd"))
    # print("pos_error in joint_pos_track:", pos_error)
    pos_term = torch.square(pos_error)/std**2
    weight_torch = torch.tensor(weight,device=env.device).unsqueeze(0)
    weight_torch = weight_torch / sum(weight)
    # print("weight_torch:", weight_torch)
    return torch.sum(weight_torch*(torch.exp(-pos_term) - 0.003*pos_term),dim=1)

def joint_vel_stable(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_error = asset.data.joint_vel[:, asset_cfg.joint_ids]
    vel_term = torch.sum(torch.square(vel_error), dim=1)/std**2
    # return torch.exp(-vel_term/std**2)
    return torch.exp(-vel_term) - 0.003*vel_term

def arm_deviation_with_cmd_mask(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_cmd_enable = env.command_manager.get_term("joint_pos_cmd").enable
    enable = torch.ones_like(angle)
    enable[:,:7] = (1-joint_cmd_enable[:,0]).unsqueeze(-1).repeat(1,7)
    enable[:,7:] = (1-joint_cmd_enable[:,1]).unsqueeze(-1).repeat(1,7)
    angle = angle * enable
    return torch.sum(torch.abs(angle), dim=1)


def body_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2) > 5 + 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]), dim=1)


def feet_too_near_humanoid(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_quat = asset.data.root_quat_w
    feet_vel = math_utils.quat_apply_inverse(yaw_quat(root_quat).unsqueeze(1).repeat(1,2,1), asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :])
    # bad_vel: 脚速度与指令方向之差
    bad_vel = torch.abs(feet_vel[:,:,0]*env.command_manager.get_command("base_velocity")[:,1].unsqueeze(-1) - feet_vel[:,:,1]*env.command_manager.get_command("base_velocity")[:,0].unsqueeze(-1)) 
    command_vel_norm = torch.norm(env.command_manager.get_command("base_velocity")[:,0:2],dim=-1)
    command_vel_norm[command_vel_norm<0.1]=0.1
    bad_reward = torch.mean(bad_vel, dim=-1) / command_vel_norm - 0.2
    bad_reward[bad_reward<0] = 0
    feet_pos_error = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :] - asset.data.body_pos_w[:, asset_cfg.body_ids[1], :]
    root_quat = asset.data.root_quat_w
    feet_pos_yaw = math_utils.quat_apply_inverse(yaw_quat(root_quat), feet_pos_error)
    return (threshold - torch.norm(feet_pos_yaw[:,0:2],dim=1)).clamp(min=0) + bad_reward

def feet_too_near_humanoid_pure(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_error = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :] - asset.data.body_pos_w[:, asset_cfg.body_ids[1], :]
    root_quat = asset.data.root_quat_w
    feet_pos_yaw = math_utils.quat_apply_inverse(yaw_quat(root_quat), feet_pos_error)
    return 10.0*(threshold - torch.norm(feet_pos_yaw[:,0:2],dim=1)).clamp(min=0)

def stand_still(
    env: ManagerBasedRLEnv, lin_threshold: float = 0.1, ang_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    penalizing linear and angular motion when command velocities are near zero.
    """

    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    base_ang_vel = asset.data.root_ang_vel_w[:, -1]

    commands = env.command_manager.get_command("base_velocity")#todo

    lin_commands = commands[:, :2]
    ang_commands = commands[:, 2]

    reward_lin = torch.sum(
        torch.abs(base_lin_vel) * (torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold), dim=-1
    )

    reward_ang = torch.abs(base_ang_vel) * (torch.abs(ang_commands) < ang_threshold)

    total_reward = reward_lin + reward_ang
    return total_reward

def track_base_height_command_exp(env: ManagerBasedRLEnv, command_name: str, std: float, sensor_cfg: SceneEntityCfg, \
asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command(command_name)[:, 0]

    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 10.0
    mask = ~in_contact[:,0] & ~in_contact[:,1]
    in_contact[mask] = True
    foot_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    support_z = (in_contact * foot_pos_z).sum(dim=1) / (in_contact.sum(dim=1) + 1e-6)
    
    baselink_pos_z = asset.data.root_pos_w[:, 2]
    relative_height = baselink_pos_z - support_z
    reward = torch.exp(-(relative_height - target_height)**2/(std**2))
    return reward

def track_pitch_command_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track pitch command and maintain zero roll using exponential kernel.
    
    Penalizes deviations from:
    1. Target pitch angle (command)
    2. Zero roll angle
    
    Uses projected gravity vector components for efficient computation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    target_pitch = env.command_manager.get_command(command_name)[:, 1]
    
    desired_sin_pitch = torch.sin(target_pitch)
    projected_gravity = asset.data.projected_gravity_b
    
    pitch_error = torch.square(projected_gravity[:, 0] - desired_sin_pitch)  # X分量对应pitch
    roll_error = torch.square(projected_gravity[:, 1])                     # Y分量对应roll
    
    total_error = pitch_error + roll_error
    
    return torch.exp(-total_error / (std ** 2))

def track_pitch_command_exp_vel(
    env: ManagerBasedRLEnv,
    pitch_command_name: str,
    vel_command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track pitch command and maintain zero roll using exponential kernel.
    
    Penalizes deviations from:
    1. Target pitch angle (command)
    ps:if vel command > 0.1 then try to stable the attitude(0 pitch)
    2. Zero roll angle
    
    Uses projected gravity vector components for efficient computation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    target_pitch = env.command_manager.get_command(pitch_command_name)[:, 1]
    target_pitch *= torch.norm(env.command_manager.get_command(vel_command_name)[:, :2], dim=1) < 0.1
    desired_sin_pitch = torch.sin(target_pitch)
    projected_gravity = asset.data.projected_gravity_b
    
    pitch_error = torch.square(projected_gravity[:, 0] - desired_sin_pitch)  # X分量对应pitch
    roll_error = torch.square(projected_gravity[:, 1])                     # Y分量对应roll
    
    total_error = pitch_error + roll_error

    reward = torch.exp(-total_error / (std ** 2))

    #reward *= torch.norm(env.command_manager.get_command(vel_command_name)[:, :2], dim=1) < 0.1
    
    return reward
def track_base_height_command_exp_vel(env: ManagerBasedRLEnv, height_command_name: str,vel_command_name: str, std: float, sensor_cfg: SceneEntityCfg, \
asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command(height_command_name)[:, 0]
    mask = torch.norm(env.command_manager.get_command(vel_command_name)[:, :2], dim=1) > 0.1
    target_height = torch.where(mask, torch.full_like(target_height, 0.6), target_height)

    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 10.0
    mask = ~in_contact[:,0] & ~in_contact[:,1]
    in_contact[mask] = True
    foot_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    support_z = (in_contact * foot_pos_z).sum(dim=1) / (in_contact.sum(dim=1) + 1e-6)
    
    baselink_pos_z = asset.data.root_pos_w[:, 2]
    relative_height = baselink_pos_z - support_z
    reward = torch.exp(-(relative_height - target_height)**2/(std**2))
    return reward
    
def track_waist_command_l2_vel(
    env: ManagerBasedRLEnv,
    behavior_command_name: str,
    vel_command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track waist command and maintain zero roll using exponential kernel.
    
    Penalizes deviations from:
    1. Target waist angle (command)
    ps:if vel command > 0.1 then try to stable the attitude(0 waist)
    
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    target_waist = env.command_manager.get_command(behavior_command_name)[:, 2]
    target_waist *= (torch.norm(env.command_manager.get_command(vel_command_name)[:, :2], dim=1)  + torch.abs(env.command_manager.get_command(vel_command_name)[:, 2])) < 0.1

    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - target_waist
    
    reward = torch.sum(torch.square(angle), dim=1)
    
    return reward

def feet_contact_num(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg,command_name: str) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    contact_num = torch.sum(is_contact, dim=-1)
    is_swing = torch.min(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] < threshold
    swing_num = torch.sum(is_swing, dim=-1)
    reward = torch.where(
        is_stand_bool(env,"base_velocity"),
        contact_num == 2,    # 小速度时要求双脚接触
        torch.logical_or(contact_num == 1, swing_num == 1)    # 大速度时要求单脚接触
    )
    return reward

def lateral_distance(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    min_dist: float, 
    max_dist: float, 
    constant_reward: float = 0.1
) -> torch.Tensor:
    """Penalize inappropriate lateral distance between feet.
    
    This function encourages the robot to maintain an appropriate stance width
    by penalizing when feet are too close together (unstable) or too far apart (unnatural).
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot
        min_dist: Minimum allowed distance between feet
        max_dist: Maximum allowed distance between feet
        constant_reward: Reward value when within distance bounds
        
    Returns:
        Reward for appropriate lateral foot distance
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    asset_pos_world = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    asset_pos_body = quat_apply_inverse(yaw_quat(root_quat.unsqueeze(1)), asset_pos_world - root_pos.unsqueeze(1))
    asset_dist = torch.abs(asset_pos_body[:, 0, 1] - asset_pos_body[:, 1, 1]).unsqueeze(1)

    dist = torch.where(
        asset_dist < min_dist, 
        torch.abs(asset_dist - min_dist), 
        torch.where(
            asset_dist > max_dist, 
            torch.abs(asset_dist - max_dist), 
            torch.zeros_like(asset_dist) - constant_reward
        )
    )
    reward = torch.min(dist, dim=1).values
    return reward

def robot_momentum_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for maintaining low and angular momentum."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_momentum_Z = torch.square(robot_centroidal_momentum(env, asset_cfg)[:, 5])
    return ang_momentum_Z

def robot_limbs_momentum_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for maintaining low limb angular momentum."""
    ang_momentum_Z = torch.square(robot_centroidal_momentum_link(env, asset_cfg)[:, 5])
    # enable: [0,0] [1,0] [0,1] [1,1] for [left_arm, right_arm]
    # if enable, 以执行指令为主目标，不惩罚角动量
    joint_cmd_enable = env.command_manager.get_term("joint_pos_cmd").enable
    if asset_cfg.joint_names[0].startswith("right_hip"):  # 右腿左手角动量平衡
        ang_momentum_Z = ang_momentum_Z * (1 - joint_cmd_enable[:,0])
    elif asset_cfg.joint_names[0].startswith("left_hip"):  # 左腿右手角动量平衡
        ang_momentum_Z = ang_momentum_Z * (1 - joint_cmd_enable[:,1])
    return ang_momentum_Z

def hand_too_near_to_base(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold_x: float = 1.0, threshold_y: float = 0.2) -> torch.Tensor:
    """Reward for penalizing hands that are too close to the base."""
    asset: Articulation = env.scene[asset_cfg.name]
    hand_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    base_pos = asset.data.root_pos_w
    dist_w = hand_pos - base_pos.unsqueeze(1)
    base_quat = asset.data.root_quat_w
    dist_left_b = quat_apply_inverse(base_quat, dist_w[:,0,:])
    dist_right_b = quat_apply_inverse(base_quat, dist_w[:,1,:])
    joint_cmd_enable = env.command_manager.get_term("joint_pos_cmd").enable
    dist_base_l = 1-((dist_left_b[:,0]/threshold_x).square() + (dist_left_b[:,1]/threshold_y).square())
    dist_base_r = 1-((dist_right_b[:,0]/threshold_x).square() + (dist_right_b[:,1]/threshold_y).square())
    dist_b = dist_base_l.clamp(min=0)*(1-joint_cmd_enable[:,0]) + dist_base_r.clamp(min=0)*(1-joint_cmd_enable[:,1])
    return dist_b / 2.0

def waist_roll_track_exp(env: ManagerBasedRLEnv, std: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Track the roll angle of the waist joint."""
    asset: Articulation = env.scene[asset_cfg.name]
    waist_roll_pos = asset.data.joint_pos[:, asset_cfg.joint_ids[0]] - asset.data.default_joint_pos[:, asset_cfg.joint_ids[0]]
    base_roll, _, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    roll_err = base_roll + waist_roll_pos
    return torch.exp(-(roll_err/std)**2)

# 分段线性函数
def piecewise_linear(x, breakpoints, slopes, intercepts):
    """
    实现多段线性函数
    breakpoints: 分段点
    slopes: 每段的斜率
    intercepts: 每段的截距
    """
    result = torch.zeros_like(x)
    for i in range(len(breakpoints) + 1):
        if i == 0:
            mask = x <= breakpoints[i]
        elif i == len(breakpoints):
            mask = x > breakpoints[i-1]
        else:
            mask = (x > breakpoints[i-1]) & (x <= breakpoints[i])
        
        result[mask] = slopes[i] * x[mask] + intercepts[i]
    
    return result
class ContactVelPenalty(ManagerTermBase):
    """
    A reward term for penalizing contact vel.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.prev_vel = None
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg, sensor_cfg) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """

        current_vel = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, 0:3], dim=-1)
        if self.prev_vel is None:
            self.prev_vel = current_vel
            return torch.zeros(current_vel.shape[0], device=current_vel.device)
        
        contact_mask =torch.norm(self.contact_sensor.data.net_forces_w[:,self.sensor_cfg.body_ids, 0:3], dim=2) > 5.
        contact_vel = self.prev_vel * contact_mask
        self.prev_vel = current_vel
        return torch.sum(torch.square(contact_vel), dim=-1)

class FeetContactNumReward(ManagerTermBase):
    """
    A reward term for penalizing contact vel.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.threshold = cfg.params["threshold"]
        self.history_length = cfg.params["history_length"]
        self.command_name = cfg.params["command_name"]
        self.reward_history = deque(maxlen=self.history_length)

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]

    def __call__(self, env: ManagerBasedRLEnv, threshold, history_length, sensor_cfg, command_name) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        contact_forces = self.contact_sensor.data.net_forces_w
        is_contact = torch.norm(contact_forces[:, self.sensor_cfg.body_ids], dim=-1) > self.threshold
        contact_num = torch.sum(is_contact, dim=-1)

        reward = torch.where(is_stand_bool(env, self.command_name), contact_num == 2, contact_num == 1).int()

        self.reward_history.append(reward.clone())
        
        # 堆叠所有tensor并取最大值
        stacked = torch.stack(list(self.reward_history), dim=0)  # [n, 1024]
        return stacked.max(dim=0).values

def feet_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """The contact forces of the body."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    feet_acc = torch.norm(asset.data.body_acc_w[:,asset_cfg.body_ids,:],dim=-1)
    feet_acc = torch.clamp(feet_acc, 0, 100)
    return torch.sum(torch.square(feet_acc), dim=-1)