# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for various math operations."""

# needed to import for allowing type-hinting: torch.Tensor | np.ndarray
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Literal

import omni.log

"""
General
"""


@torch.jit.script
def quaternion_mean_simple(q1, q2):
    """
    简单算术平均 + 重新归一化
    适用于两个四元数夹角不大的情况
    """
    # 确保四元数在同一半球（避免q和-q的问题）
    dot_product = torch.sum(q1 * q2, dim=-1, keepdim=True)
    q2 = torch.where(dot_product < 0, -q2, q2)
    
    # 算术平均
    mean_quat = (q1 + q2) / 2.0
    
    # 重新归一化
    normlized_quat = F.normalize(mean_quat, p=2.0, dim=-1)
    
    return normlized_quat