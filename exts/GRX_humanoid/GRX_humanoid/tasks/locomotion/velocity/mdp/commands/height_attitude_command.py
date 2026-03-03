"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import copy
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformHACommandCfg, UniformHRCommandCfg


class HACommand(CommandTerm):
    """Command generator that generates height and attitude."""

    cfg: UniformHACommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformHACommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # create buffers to store the command, 0: height, 1: pitch angle, 2: yaw angle
        self.height_attitude_command = torch.zeros(self.num_envs, 3, device=self.device)
        # create metrics dictionary for logging
        self.metrics = {}
        # clip vel command based on height command
        # |Vx| < (hgt_cmd - 0.45) *2 +0.45
        # |Vy| < (hgt_cmd - 0.45) *0.8 + 0.45
        self.vel_command_clip = torch.zeros(self.num_envs, 2, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "HACommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The H&A command. Shape is (num_envs, 2)."""
        return self.height_attitude_command

    def _update_metrics(self):
        """Update the metrics based on the current state.

        In this implementation, we don't track any specific metrics.
        """
        pass

    def _resample_command(self, env_ids):
        """Resample the H&A command for specified environments."""
        # sample gait parameters
        r = torch.empty(len(env_ids), device=self.device)
        # -- frequency
        self.height_attitude_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)
        self.height_attitude_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.pitch_angle)
        self.height_attitude_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.yaw_angle)
        self.vel_command_clip[env_ids, 0] = (self.height_attitude_command[env_ids, 0] - 0.45) * 2 + 0.2
        self.vel_command_clip[env_ids, 1] = (self.height_attitude_command[env_ids, 0] - 0.45) * 0.8 + 0.2
        

    def _update_command(self):
        """Update the command. No additional processing needed in this implementation."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        In this implementation, we don't provide any debug visualization.
        """
        pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        In this implementation, we don't provide any debug visualization.
        """
        pass


class HRCommand(CommandTerm):
    """Command generator that generates height and roll."""

    """Command generator that generates height and attitude."""

    cfg: UniformHRCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformHRCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # create buffers to store the command, 0: height, 1: torso yaw angle, 2: torso roll angle, 3: torso pitch angle
        self.height_attitude_command = torch.zeros(self.num_envs, 4, device=self.device)
        # create metrics dictionary for logging
        self.metrics = {}
        # clip vel command based on height command
        # |Vx| < (hgt_cmd - 0.45) *2 +0.45
        # |Vy| < (hgt_cmd - 0.45) *0.8 + 0.45
        self.vel_command_clip = torch.zeros(self.num_envs, 2, device=self.device)
        self.task_progress = 0.
        self.pitch_command_clip = torch.zeros(self.num_envs, 2, device=self.device)
        self.pitch_command_clip[:,1] = 1.2
        self.rel_straight = 0.25
        self.is_straight_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "HACommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The H&A command. Shape is (num_envs, 2)."""
        return self.height_attitude_command

    def update_task_progress(self):
        """Update the task progress.
        """
        self.task_progress = min(self.task_progress+0.1, 0.98)
        self.cfg.ranges.torso_yaw = (-self.task_progress*1.7, self.task_progress*1.7)
        self.cfg.ranges.torso_roll = (-self.task_progress*0.2, self.task_progress*0.2)
        self.cfg.ranges.torso_pitch = (-self.task_progress*0.2, self.task_progress*1.5)
        self.cfg.ranges.height = (0.9 - self.task_progress*0.45, 0.92)

    def _update_metrics(self):
        """Update the metrics based on the current state.

        In this implementation, we don't track any specific metrics.
        """
        pass

    def _resample_command(self, env_ids):
        """Resample the H&A command for specified environments."""
        # sample gait parameters
        r = torch.empty(len(env_ids), device=self.device)
        # -- frequency
        self.height_attitude_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)
        # vel_cmd = self._env.command_manager.get_command("base_velocity")[:, :2]
        self.height_attitude_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.torso_yaw)
        self.height_attitude_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.torso_roll)
        self.height_attitude_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.torso_pitch)
        self.vel_command_clip[env_ids, 0] = (self.height_attitude_command[env_ids, 0] - 0.45) * 2 + 0.2
        self.vel_command_clip[env_ids, 1] = (self.height_attitude_command[env_ids, 0] - 0.45) * 0.8 + 0.2
        self.pitch_command_clip[env_ids, 0] = -0.3 + (0.7 - self.height_attitude_command[env_ids, 0])*1.2 * (self.height_attitude_command[env_ids, 0] < 0.7).float()
        self.height_attitude_command[env_ids, 3] = torch.clamp(self.height_attitude_command[env_ids, 3], self.pitch_command_clip[env_ids, 0], self.pitch_command_clip[env_ids, 1])
        vel_clip_pitch = 0.1 + 1.0 - 0.5*(abs(self.height_attitude_command[env_ids, 3]))
        self.vel_command_clip[env_ids, 0] = torch.min(self.vel_command_clip[env_ids, 0], vel_clip_pitch)
        self.is_straight_env[env_ids] = r.uniform_(0.0, 1.0) <= self.rel_straight
        straight_env_ids = self.is_straight_env.nonzero(as_tuple=False).flatten()
        half_size = straight_env_ids.size(0) // 2 
        straight_env_ids_half = straight_env_ids[:half_size]  # 取前一半
        r2 = torch.empty(len(straight_env_ids_half), device=self.device)
        # r3 = torch.empty(len(straight_env_ids), device=self.device)
        self.height_attitude_command[straight_env_ids_half, 0] = r2.uniform_(0.8,0.92)  # 躯干基本直立的环境中，有一半在较高身体高度下行走，期望增加行走稳定性
        self.height_attitude_command[straight_env_ids, 1:] = 0.
        # self.height_attitude_command[straight_env_ids, 3] = r3.uniform_(0,0.3)
        self.vel_command_clip[straight_env_ids_half, 0] = (self.height_attitude_command[straight_env_ids_half, 0] - 0.45) * 2 + 0.2
        self.vel_command_clip[straight_env_ids_half, 1] = (self.height_attitude_command[straight_env_ids_half, 0] - 0.45) * 0.8 + 0.2

    def _update_command(self):
        """Update the command. No additional processing needed in this implementation."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        In this implementation, we don't provide any debug visualization.
        """
        pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        In this implementation, we don't provide any debug visualization.
        """
        pass