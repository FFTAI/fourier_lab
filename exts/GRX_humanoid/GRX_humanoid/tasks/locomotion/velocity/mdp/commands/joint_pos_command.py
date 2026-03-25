"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from collections.abc import Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import UniformJointPosCommandCfg

"""
gr3_upper_joint_names = [ "head_yaw_joint", "head_pitch_joint",\
                        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", \
                        "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint", \
                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", \
                        "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"]
"""

class JointPosCommand(CommandTerm):
    """Command generator that generates height and attitude."""

    cfg: UniformJointPosCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformJointPosCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # create buffers to store the command
        cfg.asset_cfg.resolve(env.scene)
        asset: Articulation = env.scene[cfg.asset_cfg.name]
        self.default_joint_pos = asset.data.default_joint_pos[:, cfg.asset_cfg.joint_ids]
        self.lower = self.default_joint_pos - asset.data.default_joint_pos_limits[:, cfg.asset_cfg.joint_ids, 0]
        self.upper = asset.data.default_joint_pos_limits[:, cfg.asset_cfg.joint_ids, 1] - self.default_joint_pos
        self.joint_pos_cmd = self.default_joint_pos.clone()
        self.joint_num = len(cfg.asset_cfg.joint_ids) if isinstance(cfg.asset_cfg.joint_ids, list) else 1
        self.joint_pos_targ = self.default_joint_pos.clone()
        self.joint_pos_start = self.default_joint_pos.clone()
        self.joint_pos_interpolated = self.default_joint_pos.clone()
        self.task_progress = 0.
        self.exec_time = torch.zeros((env.scene.num_envs,1), device=env.device)
        self.max_exec_time = torch.ones((env.scene.num_envs,1), device=env.device)
        self.exec_time_range = torch.ones((env.scene.num_envs,2), device=env.device)
        self.dt = env.step_dt
        self.delay_buffer = torch.zeros((env.scene.num_envs, int(1/self.dt + 0.001)), device=env.device)
        # create metrics dictionary for logging
        self.metrics = {}
        # enable=0,无指令；enable=1，有指令，一开始都有指令，但锁在default pos处
        self.enable_flag = torch.ones((env.scene.num_envs,2), dtype=int, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "JointPosCommand:\n"
        msg += f"\tCommand dimension: {self.joint_num}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The joint pos command. Shape is (num_envs, num_upper_joints)."""
        return self.joint_pos_cmd
    
    @property
    def enable(self) -> torch.Tensor:
        """The joint pos command enable flags. Shape is (num_envs, 2)."""
        return self.enable_flag

    def update_task_progress(self):
        """Update the task progress.
        """
        self.task_progress = min(self.task_progress+0.05, 0.98)
        self.cfg.ranges_scaled = (-self.task_progress, self.task_progress)

    def _update_metrics(self):
        """Update the metrics based on the current state.

        In this implementation, we don't track any specific metrics.
        """
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
                # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)

        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0

        # set the command counter to zero
        self.command_counter[env_ids] = 0
        # reset joint pos command before resample
        self.joint_pos_targ[env_ids] = self.default_joint_pos[env_ids].clone()
        self.joint_pos_start[env_ids] = self.default_joint_pos[env_ids].clone()
        self.joint_pos_interpolated[env_ids] = self.default_joint_pos[env_ids].clone()
        # resample the command
        self._resample(env_ids)

        return extras

    def _resample_command(self, env_ids):
        """Resample the joint pos command for specified environments."""
        r =  torch.zeros(len(env_ids), self.joint_num, device=self.device)
        r_uniform = r.uniform_(*self.cfg.ranges_scaled)
        if self._env.stage < 3:
            self.joint_pos_cmd[env_ids, :] =  self.lower[env_ids, :] * r_uniform * (r_uniform < 0)  + \
                    self.upper[env_ids, :] * r_uniform * (r_uniform > 0) + self.default_joint_pos[env_ids, :]
        else:
            # previous joint pos command is the start position, keep smooth
            self.joint_pos_start[env_ids, :] = self.joint_pos_targ[env_ids, :]
            r =  torch.zeros(len(env_ids), self.joint_num, device=self.device)
            r_uniform = r.uniform_(*self.cfg.ranges_scaled)
            self.joint_pos_targ[env_ids, :] =  self.lower[env_ids, :] * r_uniform * (r_uniform < 0)  + \
                    self.upper[env_ids, :] * r_uniform * (r_uniform > 0) + self.default_joint_pos[env_ids, :]
            self.exec_time[env_ids] = 0.
            # 生成随机延迟分布 delay buffer=1, 当前帧joint pos cmd delay， 否则指令无延迟， 每帧延迟概率为0.3
            self.delay_buffer[env_ids, :] = torch.bernoulli(0.3*torch.ones((len(env_ids), int(1/self.dt + 0.001)), device=self.device))
                
    def _update_command(self):
        """Update the command. No additional processing needed in this implementation."""
        if self._env.stage >= 2.5:
            # interpolate the joint pos command，1s 时间内插值到终点
            delta_pos = self.joint_pos_targ - self.joint_pos_start

            # wrist cmd 插值时长0.5s，其余关节1s
            exec_env_ids = torch.where(self.exec_time < self.max_exec_time)[0]
            arr_env_ids = torch.where(self.exec_time >= self.max_exec_time)[0]
            if len(exec_env_ids) > 0:
                x = self.exec_time[exec_env_ids] / self.max_exec_time[exec_env_ids]
                self.joint_pos_interpolated[exec_env_ids, :] = self.joint_pos_start[exec_env_ids, :] + \
                    (10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5) * delta_pos[exec_env_ids, :]
                # delay the command
                if self.joint_num > 1:
                    exec_release = torch.where(self.delay_buffer[exec_env_ids][:,0] == 0)[0]
                    release_env_ids = exec_env_ids[exec_release]
                    self.joint_pos_cmd[release_env_ids, :] = self.joint_pos_interpolated[release_env_ids, :]

                self.delay_buffer[:, :-1] = self.delay_buffer[:, 1:]
                self.delay_buffer[:, -1] = 0

            # random delay
            if len(arr_env_ids) > 0:
                self.joint_pos_cmd[arr_env_ids, :] = self.joint_pos_targ[arr_env_ids, :]
            
            self.exec_time += self.dt
            self.exec_time = torch.min(self.exec_time, self.max_exec_time)


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
