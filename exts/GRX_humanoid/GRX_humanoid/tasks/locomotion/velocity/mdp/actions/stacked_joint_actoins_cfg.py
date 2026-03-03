from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import actions_cfg
from .stacked_joint_actions import StackedJointActions

@configclass
class StackedJointActionsCfg(actions_cfg.JointPositionActionCfg):
    """Configuration for StackedJointActions.

    This configuration class extends the JointActionCfg to support stacking reference joint positions.
    It includes parameters for the stacked joint names.
    JointPositionAction 中, self._joint_ids只在set_joint_position_target时使用, process_actions是对actions做整体处理, 并不需要self._joint_ids
    所以, 为了方便最好在process_actions中叠加参考轨迹/关节指令, stack_joint_ids应是action的切片, 和机器人joint names无关
    """

    class_type: type[ActionTerm] = StackedJointActions

    stack_joint_names: list[str] = MISSING
    """The names of the joints to stack."""
    command_joint_names: list[str] = MISSING
    """The names of the joints corresponding to the command to be stacked."""
    command_name: str = "joint_position"
    """The name of the command to be stacked."""