from __future__ import annotations
from dataclasses import MISSING

import torch
from collections.abc import Sequence
from isaaclab.envs.mdp.actions import JointAction, JointPositionAction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from GRX_humanoid.tasks.locomotion.velocity.env.multi_stage_manager_based_rl_env import MultiStageManagerBasedRLEnv
    from .stacked_joint_actoins_cfg import StackedJointActionsCfg


class StackedJointActions(JointPositionAction):
    cfg: StackedJointActionsCfg

    def __init__(self, cfg: StackedJointActionsCfg, env: MultiStageManagerBasedRLEnv):
        # initialize the action term
        super().__init__(cfg, env)
        self.stack_joint_ids = [self.cfg.joint_names.index(name) for name in self.cfg.stack_joint_names]
        self.stack_command_ids = [self.cfg.command_joint_names.index(name) for name in self.cfg.stack_joint_names]
        print(  f"StackedJointActions: stack_joint_names={self.cfg.stack_joint_names}, "
                f"stack_joint_ids={self.stack_joint_ids}, "
                f"stack_command_ids={self.stack_command_ids}"  )
        self._num_stack_joints = len(self.stack_joint_ids)
    
    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        # 叠加参考轨迹
        joint_command = self._env.command_manager.get_command(self.cfg.command_name)
        self._processed_actions[:, self.stack_joint_ids] += joint_command[:, self.stack_command_ids] - self._offset[:, self.stack_joint_ids]
