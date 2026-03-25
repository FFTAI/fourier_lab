from __future__ import annotations
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar
from isaacsim.core.version import get_version
import time

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

from GRX_humanoid.tasks.locomotion.velocity.managers.command_modify_manager import CommandModifyManager

from pxr import Usd, Sdf
import omni.usd

class MultiStageManagerBasedRLEnv(ManagerBasedRLEnv):
    """A multi-stage manager-based RL environment. 多阶段任务强化学习环境
    新增类成员: self.stage, int, 从1开始, 表示当前任务阶段, 不同stage的指令不同
        stage = 1, 全向行走
        stage = 2, 增加高度和姿态(pitch)指令
        stage = 3, 增加上肢关节位置跟踪
    """
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""
    
    def __init__(self, cfg: ManagerBasedRLEnvCfg, **kwargs: Any):
        """Initialize the multi-stage environment."""
        super().__init__(cfg, **kwargs)
        self.stage = torch.tensor(1, dtype=torch.float32) # Initialize the stage to 1
        self.prev_stage_update_step = 0
    
    def load_managers(self):
        super().load_managers()
        self.command_manager = CommandModifyManager(self.cfg.commands, self)
        print("[INFO] Reload command Manager: ", self.command_manager)