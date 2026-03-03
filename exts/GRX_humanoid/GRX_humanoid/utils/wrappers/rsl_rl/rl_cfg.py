# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


@configclass
class MYRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is MY PPO."""

    sym_loss: bool = MISSING

    obs_terms: list[dict] = MISSING

    act_permutation: list[float] = MISSING

    frame_stack: int = MISSING

    sym_coef: float = MISSING
