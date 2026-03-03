from __future__ import annotations

import inspect
import torch
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import omni.kit.app

from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
from isaaclab.managers import CommandManager
from isaaclab.managers.manager_term_cfg import CommandTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class CommandModifyManager(CommandManager):
    _env: ManagerBasedRLEnv
    
    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the command modify manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, CommandTermCfg]``).
            env: The environment instance.
        """
        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)

    """
    Operations - Term settings.
    """

    def compute(self, dt: float):
        """Updates the commands.

        This function calls each command term managed by the class.

        Args:
            dt: The time-step interval of the environment.

        """
        # iterate over all the command terms
        for term in self._terms.values():
            # compute term's value
            term.compute(dt)
        # clip vel cmd based on height cmd
        # vel和height 相互限制, torso orient 要根据vel cmd来限制
        # 干脆每次resample时随机侧重
        vel_cmd = self._terms["base_velocity"].vel_command_b
        vel_limit = self._terms["height_attitude"].vel_command_clip
        vel_cmd[:, 0] = torch.clamp(vel_cmd[:, 0], -vel_limit[:, 0], vel_limit[:, 0])
        vel_cmd[:, 1] = torch.clamp(vel_cmd[:, 1], -vel_limit[:, 1], vel_limit[:, 1])

    def set_term_cfg(self, term_name: str, cfg: CommandTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the reward term.
            cfg: The configuration for the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._terms:
            raise ValueError(f"Reward term '{term_name}' not found.")
        # set the configuration
        self._terms[term_name] = cfg

    def get_term_cfg(self, term_name: str) -> CommandTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the reward term.

        Returns:
            The configuration of the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._terms:
            raise ValueError(f"Reward term '{term_name}' not found.")
        # return the configuration
        return self._terms[term_name]