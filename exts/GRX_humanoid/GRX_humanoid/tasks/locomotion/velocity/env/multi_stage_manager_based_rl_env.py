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
        self._apply_envelopes()
    
    # def load_managers(self):
    #     super().load_managers()
    #     self.command_manager = CommandModifyManager(self.cfg.commands, self)
    #     print("[INFO] Reload command Manager: ", self.command_manager)

    def _apply_envelopes(self):
        robot = self.scene["robot"]
        art_view = robot.root_physx_view
        joint_paths = art_view.dof_paths[0]
        self.action_control_joint_ids, _ = robot.find_joints(self.cfg.actions.joint_pos.joint_names, preserve_order=True)
        print(f"[Envelope] Applying drive envelopes to {len(self.action_control_joint_ids)} joints: ", self.action_control_joint_ids)
        # gr3 v233
        # max_act_vel_list = [731.667,737.969,737.969,731.667,953.975,953.975,
        #                     731.667,737.969,737.969,731.667,953.975,953.975,
        #                     737.969,750.0,750.0, 
        #                     527.694, 527.694, 
        #                     444.042,444.042,359.817,359.817,359.817,527.694,527.694,
        #                     444.042,444.042,359.817,359.817,359.817,527.694,527.694]
        # vel_resist_list = [0.062628,0.026513,0.026513,0.062628,0.0,0.0,
        #                     0.062628,0.026513,0.026513,0.062628,0.0,0.0,
        #                     0.026513,0.0,0.0,
        #                     0.0, 0.0,
        #                     0.077226,0.077226,0.036638,0.036638,0.036638,0.0,0.0,
        #                     0.077226,0.077226,0.036638,0.036638,0.036638,0.0,0.0]
        # speed_grad_List = [0.584573,2.025858,2.025858,0.584573,6.493521,6.493521,
        #                     0.584573,2.025858,2.025858,0.584573,6.493521,6.493521,
        #                     2.025858,2.111024,2.111024,
        #                     12.370959, 12.370959,
        #                     2.187657,2.187657,1.986253,1.986253,1.986253,12.370959, 12.370959,
        #                     2.187657,2.187657,1.986253,1.986253,1.986253,12.370959, 12.370959]
        # gr3 v224
        max_act_vel_list = [731.667,737.969,737.969,731.667,953.975,953.975,
                            731.667,737.969,737.969,731.667,953.975,953.975,
                            737.969,750.0,750.0, 
                            527.694, 527.694, 
                            444.042,444.042,359.817,359.817,359.817,527.694,527.694,
                            444.042,444.042,359.817,359.817,359.817,527.694,527.694]
        vel_resist_list = [ 0.286447,0.026513,0.026513,0.286447,0.0,0.0,
                            0.286447,0.026513,0.026513,0.286447,0.0,0.0,
                            0.026513,0.,0.,
                            0.0, 0.0,
                            0.077226,0.077226,0.036638,0.036638,0.036638,0.0,0.0,
                            0.077226,0.077226,0.036638,0.036638,0.036638,0.0,0.0]
        speed_grad_List = [ 0.204628,2.025858,2.025858,0.204628,6.493521,6.493521,
                            0.204628,2.025858,2.025858,0.204628,6.493521,6.493521,
                            2.025858,2.679679,2.679679,
                            12.370959, 12.370959,
                            2.187657,2.187657,1.986253,1.986253,1.986253,12.370959, 12.370959,
                            2.187657,2.187657,1.986253,1.986253,1.986253,12.370959, 12.370959]
        for j in self.action_control_joint_ids:
            joint_path = joint_paths[j]
            self.set_drive_envelope_raw(
                joint_prim_path=joint_path,
                axis="angular",
                max_act_vel=max_act_vel_list[j],
                vel_resist=vel_resist_list[j],
                speed_grad=speed_grad_List[j],
            )

    def set_drive_envelope_raw(
        self,
        joint_prim_path: str,
        axis: str = "angular",
        max_act_vel: float | None = None,
        vel_resist: float | None = None,
        speed_grad: float | None = None,
    ):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(joint_prim_path)

        ns = "physxDrivePerformanceEnvelope"

        def _attr(name: str):
            return f"{ns}:{axis}:{name}"

        if max_act_vel is not None:
            a = prim.CreateAttribute(_attr("maxActuatorVelocity"), Sdf.ValueTypeNames.Float)
            a.Set(max_act_vel)
        if vel_resist is not None:
            a = prim.CreateAttribute(_attr("velocityDependentResistance"), Sdf.ValueTypeNames.Float)
            a.Set(vel_resist)
        if speed_grad is not None:
            a = prim.CreateAttribute(_attr("speedEffortGradient"), Sdf.ValueTypeNames.Float)
            a.Set(speed_grad)

        # print(f"[Envelope] {joint_prim_path} axis={axis}")