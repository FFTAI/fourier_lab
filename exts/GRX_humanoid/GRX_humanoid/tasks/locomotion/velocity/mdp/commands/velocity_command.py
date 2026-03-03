
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import CustomVelocityCommandCfg
    from .commands_cfg import FeedbackUniformVelocityCommandCfg

class CustomVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: CustomVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: CustomVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.is_zero_vel_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        self.is_zero_vel_x_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[0]
        self.is_zero_vel_y_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[1]
        self.is_zero_vel_yaw_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[2]

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        # Enforce zero velocity for individual elements
        # TODO: check if conversion is needed
        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[zero_vel_x_env_ids, 0] = 0.0
        self.vel_command_b[zero_vel_y_env_ids, 1] = 0.0
        self.vel_command_b[zero_vel_yaw_env_ids, 2] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).


class FeedbackUniformVelocityCommand(CommandTerm):
    cfg: FeedbackUniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: FeedbackUniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.scene = env.scene

        # obtain the size of the map
        terrain_gen_cfg = self.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        self.border_width = terrain_gen_cfg.border_width
        self.map_width = n_rows * grid_width + 2 * self.border_width
        self.map_height = n_cols * grid_length + 2 * self.border_width

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.need_heading_target = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.heading_target_vec = torch.ones(self.num_envs, 3, device=self.device)
        self.heading_target_vec[:, 1:] = 0.0
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.is_max_forward_velocity_env = torch.zeros_like(self.is_heading_env)
        self.is_max_backward_velocity_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = 0.
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = 0.
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        self.need_heading_target = (torch.abs(self.robot.data.root_pos_w[:, 0]) < self.map_width * 0.5 - self.border_width + 5.0) & ~self.is_standing_env

        # set max forward or backward velocity envs
        self.is_max_forward_velocity_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_max_velocity_envs / 2
        self.is_max_backward_velocity_env[env_ids] = (r.uniform_(0.0, 1.0) <= self.cfg.rel_max_velocity_envs / 2) \
              & ~self.is_max_forward_velocity_env[env_ids]
        max_forward_velocity_env_ids = (self.is_max_forward_velocity_env & ~self.is_standing_env).nonzero(as_tuple=False).flatten()
        max_backward_velocity_env_ids = (self.is_max_backward_velocity_env & ~self.is_standing_env).nonzero(as_tuple=False).flatten()
        self.vel_command_b[max_forward_velocity_env_ids, 0] = self.cfg.ranges.lin_vel_x[1]
        if self.cfg.ranges.lin_vel_x[0] < 0.0:
            self.vel_command_b[max_backward_velocity_env_ids, 0] = self.cfg.ranges.lin_vel_x[0]

        # set random velocity commands for at-bounds envs(don't need updating heading target)
        _env_ids = env_ids.clone().detach()[~self.need_heading_target[env_ids] & (self.robot.data.root_pos_w[env_ids, 0] > 0.0)]
        _ranges = {"lin_vel_x": (0.0, self.cfg.ranges.lin_vel_x[1]) if self.cfg.ranges.lin_vel_x[0] > 0.0 else (0.0, 0.0),
                   "lin_vel_y": self.cfg.ranges.lin_vel_y,
                   "ang_vel_z": self.cfg.ranges.ang_vel_z}
        self.vel_command_b[_env_ids, :] = self._resample_command_callback(_env_ids, _ranges)
        if self.cfg.ranges.lin_vel_x[0] < 0.0:  # backward velocity when at back bound
            _env_ids = env_ids.clone().detach()[~self.need_heading_target[env_ids] & (self.robot.data.root_pos_w[env_ids, 0] < 0.0)]
            _ranges = {"lin_vel_x": (self.cfg.ranges.lin_vel_x[0], 0.0),
                       "lin_vel_y": self.cfg.ranges.lin_vel_y,
                       "ang_vel_z": self.cfg.ranges.ang_vel_z}
            self.vel_command_b[_env_ids, :] = self._resample_command_callback(_env_ids, _ranges)

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from y-position feedback 
        forward_heading = math_utils.quat_apply(self.robot.data.root_quat_w, self.heading_target_vec)
        heading = torch.atan2(forward_heading[:, 1], forward_heading[:, 0])
        heading_direction = self.vel_command_b[:, 0].sign()
        y_position_error = self.scene.env_origins[:, 1] - self.robot.data.root_pos_w[:, 1]
        heading_error = math_utils.wrap_to_pi(torch.atan(y_position_error * heading_direction) - heading)

        # Only process feedback commands if self.need_heading_target == True
        # TODO: how to handle when back to the inner terrain?
        self.need_heading_target = (torch.abs(self.robot.data.root_pos_w[:, 0]) < self.map_width * 0.5 - self.border_width + 5.0) & ~self.is_standing_env
        clipped_values = torch.clip(
            self.cfg.heading_control_stiffness * heading_error[self.need_heading_target],
            min=self.cfg.ranges.ang_vel_z[0],
            max=self.cfg.ranges.ang_vel_z[1],
        )
        self.vel_command_b[self.need_heading_target, 2] = torch.where(
            torch.abs(clipped_values) < 0.1, 0.0, clipped_values
        )

        # check unsafe commands env and set commands to standing
        # target_envs = ((torch.abs(self.robot.data.root_pos_w[:, 0]) < self.map_width * 0.5 - self.border_width) & (torch.abs(self.vel_command_b[:, 1]) > 0.03)).nonzero(as_tuple=False).flatten()
        # self.is_standing_env[target_envs] = True

        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _resample_command_callback(self, env_ids: Sequence[int], ranges: dict) -> torch.Tensor:
        """optimal implementation of resampling velocity commands."""
        num_envs = len(env_ids)
        if num_envs == 0:
            return torch.zeros((0, 3), device=self.device)

        rand_vals = torch.rand(num_envs, 3, device=self.device)
        # weighted sampling over 4 command types with probs [0.1, 0.1, 0.6, 0.2]
        probs = torch.tensor([0.1, 0.1, 0.6, 0.2], device=self.device, dtype=torch.float)
        command_types = torch.multinomial(probs, num_samples=num_envs, replacement=True)
        vel_command = torch.zeros((num_envs, 3), device=self.device)
        
        # 为每种命令类型生成对应的速度命令
        mask_0 = command_types == 0  # 线速度x + 角速度z
        mask_1 = command_types == 1  # 只有角速度z
        mask_2 = command_types == 2  # 只有线速度y  
        mask_3 = command_types == 3  # 所有速度组合
        
        # 类型0: 线速度x + 角速度z
        if mask_0.any():
            vel_command[mask_0, 0] = rand_vals[mask_0, 0] * (ranges["lin_vel_x"][1] - ranges["lin_vel_x"][0]) + ranges["lin_vel_x"][0]
            vel_command[mask_0, 2] = rand_vals[mask_0, 2] * (ranges["ang_vel_z"][1] - ranges["ang_vel_z"][0]) + ranges["ang_vel_z"][0]
        
        # 类型1: 只有角速度z
        if mask_1.any():
            vel_command[mask_1, 2] = rand_vals[mask_1, 2] * (ranges["ang_vel_z"][1] - ranges["ang_vel_z"][0]) + ranges["ang_vel_z"][0]
        
        # 类型2: 只有线速度y
        if mask_2.any():
            vel_command[mask_2, 1] = rand_vals[mask_2, 1] * (ranges["lin_vel_y"][1] - ranges["lin_vel_y"][0]) + ranges["lin_vel_y"][0]
        
        # 类型3: 所有速度组合
        if mask_3.any():
            vel_command[mask_3, 0] = rand_vals[mask_3, 0] * (ranges["lin_vel_x"][1] - ranges["lin_vel_x"][0]) + ranges["lin_vel_x"][0]
            vel_command[mask_3, 1] = rand_vals[mask_3, 1] * (ranges["lin_vel_y"][1] - ranges["lin_vel_y"][0]) + ranges["lin_vel_y"][0]
            vel_command[mask_3, 2] = rand_vals[mask_3, 2] * (ranges["ang_vel_z"][1] - ranges["ang_vel_z"][0]) + ranges["ang_vel_z"][0]
        
        return vel_command

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.goal_ang_vel_visualizer = VisualizationMarkers(self.cfg.goal_ang_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            if self.cfg.vis_goal_ang_vel:
                self.goal_ang_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        ang_vel_des_arrow_scale, ang_vel_des_arrow_quat = self._resolve_ang_velocity_to_arrow(self.command[:, 2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        if self.cfg.vis_goal_ang_vel:
            self.goal_ang_vel_visualizer.visualize(base_pos_w, ang_vel_des_arrow_quat, ang_vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
    
    def _resolve_ang_velocity_to_arrow(self, ang_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the angular velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_ang_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(ang_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.abs(ang_velocity) * 30.0
        # arrow-direction
        heading_angle = torch.atan2(ang_velocity, torch.zeros_like(ang_velocity))
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

