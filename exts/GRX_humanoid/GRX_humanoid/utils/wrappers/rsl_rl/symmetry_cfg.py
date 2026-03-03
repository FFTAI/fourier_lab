from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

@configclass
class MYRslRlSymmetryCfg(RslRlSymmetryCfg):
    """Configuration for the PPO algorithm."""
    obs_terms: list[dict] = MISSING

    act_permutation: list[float] = MISSING

    frame_stack: int = MISSING