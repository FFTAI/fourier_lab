# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")

def base_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    base_height_w = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
    sensor = env.scene[sensor_cfg.name]
    base_height =base_height_w - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)

    return base_height < minimum_height

def feet_distance_below_minimum(
    env: ManagerBasedRLEnv, minimum_distance: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's feet distance is below the minimum distance."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot1_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0], ].squeeze(1)
    foot2_w = asset.data.body_pos_w[:, asset_cfg.body_ids[1], ].squeeze(1)
    
    feet_distance = torch.norm(foot1_w - foot2_w,dim=-1)
    # print(feet_distance)

    return feet_distance < minimum_distance

def feet_distance_upper_maximum(
    env: ManagerBasedRLEnv, maximum_distance: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's feet distance is upper the maximum distance."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot1_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0], ].squeeze(1)
    foot2_w = asset.data.body_pos_w[:, asset_cfg.body_ids[1], ].squeeze(1)
    
    feet_distance = torch.norm(foot1_w - foot2_w,dim=-1)
    # print(feet_distance)

    return feet_distance > maximum_distance