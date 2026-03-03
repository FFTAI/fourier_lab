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
        if self._env.stage < 2:
            r =  torch.zeros(len(env_ids), self.joint_num, device=self.device)
            r_uniform = r.uniform_(*self.cfg.ranges_scaled)
            self.joint_pos_cmd[env_ids, :] =  self.lower[env_ids, :] * r_uniform * (r_uniform < 0)  + \
                    self.upper[env_ids, :] * r_uniform * (r_uniform > 0) + self.default_joint_pos[env_ids, :]
            self.enable_flag[env_ids, :] = 1
        else:
            # previous joint pos command is the start position, keep smooth
            self.joint_pos_start[env_ids, :] = self.joint_pos_targ[env_ids, :]
            # enable_flag 采样： 40% no cmd, 15% left hand, 15% right hand, 30% both hands
            r_cmd =  torch.zeros(len(env_ids), device=self.device)
            r_cmd_uniform = r_cmd.uniform_(0.0, 1.0)
            for i, env_id in enumerate(env_ids):
                if r_cmd_uniform[i] < 0.3:
                    self.enable_flag[env_id, :] = 0
                elif r_cmd_uniform[i] < 0.5:
                    self.enable_flag[env_id, 0] = 1
                    self.enable_flag[env_id, 1] = 0
                elif r_cmd_uniform[i] < 0.7:
                    self.enable_flag[env_id, 0] = 0
                    self.enable_flag[env_id, 1] = 1
                else:
                    self.enable_flag[env_id, :] = 1
            # 根据enable采样指令,分别切出env_ids中不启用，启用左手，启用右手，启用双手的env ids
            no_cmd_env_ids = [env_id for i, env_id in enumerate(env_ids) if self.enable_flag[env_id, 0] == 0 and self.enable_flag[env_id, 1] == 0]
            left_cmd_env_ids = [env_id for i, env_id in enumerate(env_ids) if self.enable_flag[env_id, 0] == 1 and self.enable_flag[env_id, 1] == 0]
            right_cmd_env_ids = [env_id for i, env_id in enumerate(env_ids) if self.enable_flag[env_id, 0] == 0 and self.enable_flag[env_id, 1] == 1]
            both_cmd_env_ids = [env_id for i, env_id in enumerate(env_ids) if self.enable_flag[env_id, 0] == 1 and self.enable_flag[env_id, 1] == 1]
            # 不启用指令，目标位置为默认位置
            self.joint_pos_targ[no_cmd_env_ids, :] = self.default_joint_pos[no_cmd_env_ids, :]
            # 启用左手指令，采样左手关节，右手为默认位置
            if len(left_cmd_env_ids) > 0:
                r_left =  torch.zeros(len(left_cmd_env_ids), self.joint_num, device=self.device)
                r_uniform = r_left.uniform_(*self.cfg.ranges_scaled)
                self.joint_pos_targ[left_cmd_env_ids, :] = self.default_joint_pos[left_cmd_env_ids, :]
                self.joint_pos_targ[left_cmd_env_ids, 2:9] =  self.lower[left_cmd_env_ids, 2:9] * r_uniform[:, 2:9] * (r_uniform[:, 2:9] < 0)  + \
                    self.upper[left_cmd_env_ids, 2:9] * r_uniform[:, 2:9] * (r_uniform[:, 2:9] > 0) + self.default_joint_pos[left_cmd_env_ids, 2:9]
            # 启用右手指令，采样右手关节，左手为默认位置
            if len(right_cmd_env_ids) > 0:
                r_right =  torch.zeros(len(right_cmd_env_ids), self.joint_num, device=self.device)
                r_uniform = r_right.uniform_(*self.cfg.ranges_scaled)
                self.joint_pos_targ[right_cmd_env_ids, :] = self.default_joint_pos[right_cmd_env_ids, :]
                self.joint_pos_targ[right_cmd_env_ids, 9:] =  self.lower[right_cmd_env_ids, 9:] * r_uniform[:, 9:] * (r_uniform[:, 9:] < 0)  + \
                    self.upper[right_cmd_env_ids, 9:] * r_uniform[:, 9:] * (r_uniform[:, 9:] > 0) + self.default_joint_pos[right_cmd_env_ids, 9:]
            # 启用双手指令，采样所有关节
            if len(both_cmd_env_ids) > 0:
                r_both =  torch.zeros(len(both_cmd_env_ids), self.joint_num, device=self.device)
                r_uniform = r_both.uniform_(*self.cfg.ranges_scaled)
                self.joint_pos_targ[both_cmd_env_ids, :] =  self.lower[both_cmd_env_ids, :] * r_uniform * (r_uniform < 0)  + \
                        self.upper[both_cmd_env_ids, :] * r_uniform * (r_uniform > 0) + self.default_joint_pos[both_cmd_env_ids, :]
            # reset execution time and sample new max execution time
            # print(f"[JointPosCommand] Enable flags for envs {env_ids}: ", self.enable_flag[env_ids, :])
            # print(f"[JointPosCommand] Resampled joint pos target for envs {env_ids}: ", self.joint_pos_targ[env_ids, :])
            # input()
            self.exec_time[env_ids] = 0.
            is_walk_bool = torch.norm(self._env.command_manager.get_command("base_velocity")[:,0:3], dim=1) >= 0.01
            self.exec_time_range[env_ids, 0] = torch.where(is_walk_bool[env_ids], 1.3, 0.9)
            self.exec_time_range[env_ids, 1] = torch.where(is_walk_bool[env_ids], 3.5, 2.5)
            self.max_exec_time[env_ids] = (torch.rand(len(env_ids), device=self.device) * \
                (self.exec_time_range[env_ids,1] - self.exec_time_range[env_ids,0]) + self.exec_time_range[env_ids,0]).unsqueeze(1)
            # 生成随机延迟分布 delay buffer=1, 当前帧joint pos cmd delay， 否则指令无延迟， 每帧延迟概率为0.3
            self.delay_buffer[env_ids, :] = torch.bernoulli(0.3*torch.ones((len(env_ids), int(1/self.dt + 0.001)), device=self.device))
        # testing: disable all commands
        # self.enable_flag[env_ids, :] = 0
        # testing: disable all left and enable right
        # self.enable_flag[env_ids, 0] = 0
        # self.enable_flag[env_ids, 1] = 1
        # testing: disable all right and enable left
        # self.enable_flag[env_ids, 0] = 1
        # self.enable_flag[env_ids, 1] = 0
        # testing: enable all commands
        # self.enable_flag[env_ids, :] = 1
                
    def _update_command(self):
        """Update the command. No additional processing needed in this implementation."""
        if self._env.stage == 2:
            # interpolate the joint pos command，1s 时间内插值到终点
            delta_pos = self.joint_pos_targ - self.joint_pos_start

            # wrist cmd 插值时长0.5s，其余关节1s
            exec_env_ids = torch.where(self.exec_time < self.max_exec_time)[0]
            arr_env_ids = torch.where(self.exec_time >= self.max_exec_time)[0]
            # exec_wrist_env_ids = torch.where(self.exec_time < 0.5)[0]
            # arr_wrist_env_ids = torch.where(self.exec_time >= 0.5)[0]
            # wrist_idx = torch.tensor([5,6,7,8,12,13,14,15], device=self.device)
            if len(exec_env_ids) > 0:
                x = self.exec_time[exec_env_ids] / self.max_exec_time[exec_env_ids]
                self.joint_pos_interpolated[exec_env_ids, :] = self.joint_pos_start[exec_env_ids, :] + \
                    (10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5) * delta_pos[exec_env_ids, :]
                # if len(exec_wrist_env_ids) > 0:
                #     self.joint_pos_interpolated[exec_wrist_env_ids.unsqueeze(1), wrist_idx] = self.joint_pos_start[exec_wrist_env_ids.unsqueeze(1), wrist_idx] + \
                #         (10 * (2 * self.exec_time[exec_wrist_env_ids]) ** 3 
                #         - 15 * (2 * self.exec_time[exec_wrist_env_ids]) ** 4 
                #         + 6 * (2 * self.exec_time[exec_wrist_env_ids]) ** 5) * delta_pos[exec_wrist_env_ids.unsqueeze(1), wrist_idx]
                # delay the command
                if self.joint_num > 1:
                    exec_release = torch.where(self.delay_buffer[exec_env_ids][:,0] == 0)[0]
                    release_env_ids = exec_env_ids[exec_release]
                    self.joint_pos_cmd[release_env_ids, :] = self.joint_pos_interpolated[release_env_ids, :]
                    # wrist delay
                    # exec_release = torch.where(self.delay_buffer[exec_wrist_env_ids][:,0] == 0)[0]
                    # release_env_ids = exec_wrist_env_ids[exec_release]
                    # self.joint_pos_cmd[release_env_ids.unsqueeze(1), wrist_idx] = self.joint_pos_interpolated[release_env_ids.unsqueeze(1), wrist_idx]

                self.delay_buffer[:, :-1] = self.delay_buffer[:, 1:]
                self.delay_buffer[:, -1] = 0

            # random delay
            if len(arr_env_ids) > 0:
                self.joint_pos_cmd[arr_env_ids, :] = self.joint_pos_targ[arr_env_ids, :]
            # if len(arr_wrist_env_ids) > 0:
            #     self.joint_pos_cmd[arr_wrist_env_ids.unsqueeze(1), wrist_idx] = self.joint_pos_targ[arr_wrist_env_ids.unsqueeze(1), wrist_idx]
            
            self.exec_time += self.dt
            self.exec_time = torch.min(self.exec_time, self.max_exec_time)
            # print(f"[JointPosCommand] Updated joint pos command: ", self.joint_pos_cmd)
        # pass


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
