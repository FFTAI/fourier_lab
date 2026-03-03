"""Functions to specify the symmetry in the observation and action space for GR3."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING
import numpy as np
from GRX_humanoid.utils.wrappers.rsl_rl import MYRslRlSymmetryCfg

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]

class SymmetryGR3:
    def __init__(self, cfg: MYRslRlSymmetryCfg):
        stacked_num = 0
        matrix_obs = []
        for term in cfg.obs_terms:
            single_obs_matrix = []
            single_obs_perm_mat = torch.zeros((len(term['permutation']), len(term['permutation']))).cuda()
            for i, perm in enumerate(term['permutation']):
                single_obs_perm_mat[int(abs(perm))][i] = np.sign(perm)
            for i in range(cfg.frame_stack):
                single_obs_matrix.append(single_obs_perm_mat)
                stacked_num += len(term['permutation'])
            matrix_obs.append(self.concatenate_given_diagonal_matrices(single_obs_matrix))
        self.obs_perm_mat = torch.zeros((stacked_num, stacked_num)).cuda()
        self.obs_perm_mat = self.concatenate_given_diagonal_matrices(matrix_obs)

        self.act_perm_mat = torch.zeros((len(cfg.act_permutation), len(cfg.act_permutation))).cuda()
        for i, perm in enumerate(cfg.act_permutation):
            self.act_perm_mat[int(abs(perm))][i] = np.sign(perm)
    
    def concatenate_given_diagonal_matrices(self, single_obs_matrix):
        total_size = sum([matrix.size(0) for matrix in single_obs_matrix])
        result = torch.zeros((total_size, total_size)).cuda()
        start_index = 0
        for matrix in single_obs_matrix:
            end_index = start_index + matrix.size(0)
            matrix_cpu = matrix.cpu()
            result[start_index:end_index, start_index:end_index] = matrix_cpu.cuda()
            start_index = end_index
        return result

@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        obs_aug = torch.matmul(obs["policy"][:], SymmetryGR3.obs_perm_mat)
    else:
        obs_aug = None
    
    # actions
    if actions is not None:
        actions_aug = torch.matmul(actions, SymmetryGR3.act_perm_mat)
    else:
        actions_aug = None

    return obs_aug, actions_aug