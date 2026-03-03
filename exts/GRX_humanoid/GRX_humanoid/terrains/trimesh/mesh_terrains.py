# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane
from GRX_humanoid.terrains.utils import get_cfg_value, create_box_with_optional_rough_surface, my_make_border

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def pyramid_stairs_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MyMeshPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_length` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    _platform_length = get_cfg_value(cfg.platform_length)
    _step_width = get_cfg_value(cfg.step_width)

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - _platform_length) // (2 * _step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - _platform_length) // (2 * _step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    if cfg.border_width > 0.0 and not cfg.holes:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (_platform_length, _platform_length)
        else:
            box_size = (terrain_size[0] - 2 * k * _step_width, terrain_size[1] - 2 * k * _step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * _step_width
        # -- dimensions
        box_height = (k + 2) * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], _step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        if cfg.holes:
            box_dims = (_step_width, box_size[1], box_height)
        else:
            box_dims = (_step_width, box_size[1] - 2 * _step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]

    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * _step_width,
        terrain_size[1] - 2 * num_steps * _step_width,
        (num_steps + 2) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_steps * step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])

    return meshes_list, origin


def inverted_pyramid_stairs_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshInvertedPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a inverted pyramid stair pattern.

    The terrain is an inverted pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_length` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    _platform_length = get_cfg_value(cfg.platform_length)
    _step_width = get_cfg_value(cfg.step_width)

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - _platform_length) // (2 * _step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - _platform_length) // (2 * _step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))
    # total height of the terrain
    total_height = (num_steps + 1) * step_height

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    if cfg.border_width > 0.0 and not cfg.holes:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders
    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (_platform_length, _platform_length)
        else:
            box_size = (terrain_size[0] - 2 * k * _step_width, terrain_size[1] - 2 * k * _step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * _step_width
        # -- dimensions
        box_height = total_height - (k + 1) * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], _step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        if cfg.holes:
            box_dims = (_step_width, box_size[1], box_height)
        else:
            box_dims = (_step_width, box_size[1] - 2 * _step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]
    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * _step_width,
        terrain_size[1] - 2 * num_steps * _step_width,
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])

    return meshes_list, origin


def ridge_stairs_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MyMeshRidgeStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_length` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    _border_width_x = get_cfg_value(cfg.border_width_x)
    _border_width_y = get_cfg_value(cfg.border_width_y)
    _platform_length = get_cfg_value(cfg.platform_length)
    _step_width = get_cfg_value(cfg.step_width)

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * _border_width_x - _platform_length) // (2 * _step_width) + 1
    # Only one direction is needed for ridge stairs
    num_steps = int(num_steps_x)

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    if (_border_width_x > 0.0 or _border_width_y > 0.0) and not cfg.holes:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * _border_width_x, cfg.size[1] - 2 * _border_width_y)
        make_borders = my_make_border(cfg.if_rough, cfg.size, border_inner_size, step_height, border_center, cfg)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * _border_width_x, cfg.size[1] - 2 * _border_width_y)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (_platform_length, _platform_length)
        else:
            box_size = (terrain_size[0] - 2 * k * _step_width, terrain_size[1])
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * _step_width
        # -- dimensions
        box_height = (k + 2) * step_height
        # generate the boxes
        # Only one box per layer
        box_dims = (box_size[0], box_size[1], box_height)
        box_pos = (terrain_center[0], terrain_center[1], box_z)
        box = create_box_with_optional_rough_surface(cfg.if_rough, box_dims, box_pos, cfg)

        # add the boxes to the list of meshes
        meshes_list += [box]

    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * _step_width,
        terrain_size[1],
        (num_steps + 2) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_steps * step_height / 2)
    box_middle = create_box_with_optional_rough_surface(cfg.if_rough, box_dims, box_pos, cfg)
    meshes_list.append(box_middle)
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])

    return meshes_list, origin


def inverted_ridge_stairs_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshInvertedPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a inverted pyramid stair pattern.

    The terrain is an inverted pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_length` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    _border_width_x = get_cfg_value(cfg.border_width_x)
    _border_width_y = max(0.0, get_cfg_value(cfg.border_width_y))
    _platform_length = get_cfg_value(cfg.platform_length)
    _step_width = get_cfg_value(cfg.step_width)

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * _border_width_x - _platform_length) // (2 * _step_width) + 1
    # Only one direction is needed for ridge stairs
    num_steps = int(num_steps_x)
    # total height of the terrain
    total_height = (num_steps + 1) * step_height

    # initialize list of meshes
    meshes_list = list()
    
    # generate the border if needed
    if (_border_width_x > 0.0 or _border_width_y > 0.0) and not cfg.holes:
        # create the front and back borders
        front_border_center = [cfg.size[0] - 0.5 * _border_width_x, 0.5 * cfg.size[1], -step_height / 2]
        front_border_dims = (_border_width_x, cfg.size[1], step_height)
        front_border = create_box_with_optional_rough_surface(cfg.if_rough, front_border_dims, front_border_center, cfg)
        back_border_center = [0.5 * _border_width_x, 0.5 * cfg.size[1], -step_height / 2]
        back_border_dims = (_border_width_x, cfg.size[1], step_height)
        back_border = create_box_with_optional_rough_surface(cfg.if_rough, back_border_dims, back_border_center, cfg)
        # create the left and right borders
        left_border_center = [0.5 * cfg.size[0], cfg.size[1] - 0.5 * _border_width_y, -total_height / 2]
        left_border_dims = (cfg.size[0] - 2 * _border_width_x, _border_width_y, total_height)
        left_border = create_box_with_optional_rough_surface(False, left_border_dims, left_border_center, cfg)
        right_border_center = [0.5 * cfg.size[0], 0.5 * _border_width_y, -total_height / 2]
        right_border_dims = (cfg.size[0] - 2 * _border_width_x  , _border_width_y, total_height)
        right_border = create_box_with_optional_rough_surface(False, right_border_dims, right_border_center, cfg)
        # add the border meshes to the list of meshes
        meshes_list += [front_border, back_border, left_border, right_border]
    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * _border_width_x, cfg.size[1] - 2 * _border_width_y)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (_platform_length, _platform_length)
        else:
            box_size = (terrain_size[0] - 2 * k * _step_width, terrain_size[1])
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * _step_width
        # -- dimensions
        box_height = total_height - (k + 1) * step_height
        # generate the boxes
        # Two boxes per layer
        box_dims = (_step_width, box_size[1], box_height)
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_top = create_box_with_optional_rough_surface(cfg.if_rough, box_dims, box_pos, cfg)
        # -- bottom
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_bottom = create_box_with_optional_rough_surface(cfg.if_rough, box_dims, box_pos, cfg)
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom]
    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * _step_width,
        terrain_size[1],
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
    box_middle = create_box_with_optional_rough_surface(cfg.if_rough, box_dims, box_pos, cfg)
    meshes_list.append(box_middle)
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])

    return meshes_list, origin

def pit_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MyMeshPitTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with two pits.

    The terrain is a flat terrain with two pit in the front and back of the center platform.

    .. image:: ../../_static/terrains/trimesh/pit_terrain.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    pit_depth = cfg.pit_depth_range[0] + difficulty * (cfg.pit_depth_range[1] - cfg.pit_depth_range[0])
    _platform_length = get_cfg_value(cfg.platform_length)
    _pit_width_1 = get_cfg_value(cfg.pit_width)
    _pit_width_2 = get_cfg_value(cfg.pit_width)
    _pit_length_1 = get_cfg_value(cfg.pit_length)
    _pit_length_2 = get_cfg_value(cfg.pit_length)
    random_offset_x_1 = np.random.uniform(0.0, 0.5 * cfg.size[0] - _platform_length - _pit_length_1)
    random_offset_x_2 = np.random.uniform(0.0, 0.5 * cfg.size[0] - _platform_length - _pit_length_2)

    # initialize list of meshes
    meshes_list = list()

    # generate the flat ground
    ground_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.05]
    ground_size = (cfg.size[0], cfg.size[1], 0.1)
    ground = create_box_with_optional_rough_surface(cfg.if_rough, ground_size, ground_center, cfg)
    meshes_list.append(ground)

    # generate the front pit

    pit_center = [0.5 * cfg.size[0] + 0.5 * _platform_length + 0.5 * _pit_length_1 + random_offset_x_1, 0.5 * cfg.size[1], 0.5 * pit_depth]
    pit_size = (_pit_length_1, _pit_width_1, pit_depth)
    pit = create_box_with_optional_rough_surface(cfg.if_rough, pit_size, pit_center, cfg)
    meshes_list.append(pit)

    # generate the back pit
    pit_center = [0.5 * cfg.size[0] - 0.5 * _platform_length - 0.5 * _pit_length_2 - random_offset_x_2, 0.5 * cfg.size[1], 0.5 * pit_depth]
    pit_size = (_pit_length_2, _pit_width_2, pit_depth)
    pit = create_box_with_optional_rough_surface(cfg.if_rough, pit_size, pit_center, cfg)
    meshes_list.append(pit)

    # origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return meshes_list, origin

def gap_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MyMeshGapTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with two gaps.

    The terrain is a flat terrain with two gaps in the front and back of the center platform.

    .. image:: ../../_static/terrains/trimesh/gap_terrain.jpg
        :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    if difficulty < 0.1:
        difficulty = 0.
    gap_length = cfg.gap_length_range[0] + difficulty * (cfg.gap_length_range[1] - cfg.gap_length_range[0])
    _platform_length = get_cfg_value(cfg.platform_length)
    _gap_width_1 = get_cfg_value(cfg.gap_width)
    _gap_width_2 = get_cfg_value(cfg.gap_width)
    _gap_depth_1 = get_cfg_value(cfg.gap_depth)
    _gap_depth_2 = get_cfg_value(cfg.gap_depth)
    random_offset_x_1 = np.random.uniform(0.0, 0.5 * cfg.size[0] - _platform_length - gap_length)
    random_offset_x_2 = np.random.uniform(0.0, 0.5 * cfg.size[0] - _platform_length - gap_length)

    # initialize list of meshes
    meshes_list = list()

    # generate the front gap's border
    front_gap_center = [0.5 * cfg.size[0] + 0.5 * _platform_length + 0.5 * gap_length + random_offset_x_1, 0.5 * cfg.size[1], -_gap_depth_1 - 0.05]
    
    front_border_center = [cfg.size[0] - 0.5 * (cfg.size[0] - (front_gap_center[0] + 0.5 * gap_length)), 0.5 * cfg.size[1], -0.5 * _gap_depth_1]
    front_border_dims = (cfg.size[0] - (front_gap_center[0] + 0.5 * gap_length), _gap_width_1, _gap_depth_1)
    front_border_1 = create_box_with_optional_rough_surface(cfg.if_rough, front_border_dims, front_border_center, cfg)

    back_border_center = [0.5 * cfg.size[0] + 0.5 * (front_gap_center[0] - (0.5 * cfg.size[0] + 0.5 * gap_length)), 0.5 * cfg.size[1], -0.5 * _gap_depth_1]
    back_border_dims = (front_gap_center[0] - (0.5 * cfg.size[0] + 0.5 * gap_length), _gap_width_1, _gap_depth_1)
    back_border_1 = create_box_with_optional_rough_surface(cfg.if_rough, back_border_dims, back_border_center, cfg)

    right_border_center = [0.75 * cfg.size[0], cfg.size[1] - 0.5 * (0.5 * cfg.size[1] - 0.5 * _gap_width_1), -0.5 * _gap_depth_1]
    right_border_dims = (0.5 * cfg.size[0], 0.5 * cfg.size[1] - 0.5 * _gap_width_1, _gap_depth_1)
    right_border_1 = create_box_with_optional_rough_surface(False, right_border_dims, right_border_center, cfg)

    left_border_center = [0.75 * cfg.size[0], 0.5 * (0.5 * cfg.size[1] - 0.5 * _gap_width_1), -0.5 * _gap_depth_1]
    left_border_dims = (0.5 * cfg.size[0], 0.5 * cfg.size[1] - 0.5 * _gap_width_1, _gap_depth_1)
    left_border_1 = create_box_with_optional_rough_surface(False, left_border_dims, left_border_center, cfg)

    down_border_dims = (gap_length, _gap_width_1, 0.1)
    down_border_1 = create_box_with_optional_rough_surface(False, down_border_dims, front_gap_center, cfg)

    meshes_list += [front_border_1, back_border_1, left_border_1, right_border_1, down_border_1]

    # generate the back gap's border
    back_gap_center = [0.5 * cfg.size[0] - 0.5 * _platform_length - 0.5 * gap_length - random_offset_x_2, 0.5 * cfg.size[1], -_gap_depth_2 - 0.05]

    front_border_center = [0.5 * cfg.size[0] - 0.5 * (0.5 * cfg.size[0] - back_gap_center[0] - 0.5 * gap_length), 0.5 * cfg.size[1], -0.5 * _gap_depth_2]
    front_border_dims = (0.5 * cfg.size[0] - back_gap_center[0] - 0.5 * gap_length, _gap_width_2, _gap_depth_2)
    front_border_2 = create_box_with_optional_rough_surface(cfg.if_rough, front_border_dims, front_border_center, cfg)

    back_border_center = [0.5 * (back_gap_center[0] - 0.5 * gap_length), 0.5 * cfg.size[1], -0.5 * _gap_depth_2]
    back_border_dims = (back_gap_center[0] - 0.5 * gap_length, _gap_width_2, _gap_depth_2)
    back_border_2 = create_box_with_optional_rough_surface(cfg.if_rough, back_border_dims, back_border_center, cfg)

    right_border_center = [0.25 * cfg.size[0], cfg.size[1] - 0.5 * (0.5 * cfg.size[1] - 0.5 * _gap_width_2), -0.5 * _gap_depth_2]
    right_border_dims = (0.5 * cfg.size[0], 0.5 * cfg.size[1] - 0.5 * _gap_width_2, _gap_depth_2)
    right_border_2 = create_box_with_optional_rough_surface(False, right_border_dims, right_border_center, cfg)

    left_border_center = [0.25 * cfg.size[0], 0.5 * (0.5 * cfg.size[1] - 0.5 * _gap_width_2), -0.5 * _gap_depth_2]
    left_border_dims = (0.5 * cfg.size[0], 0.5 * cfg.size[1] - 0.5 * _gap_width_2, _gap_depth_2)
    left_border_2 = create_box_with_optional_rough_surface(False, left_border_dims, left_border_center, cfg)

    down_border_dims = (gap_length, _gap_width_2, 0.1)
    down_border_2 = create_box_with_optional_rough_surface(False, down_border_dims, back_gap_center, cfg)

    meshes_list += [front_border_2, back_border_2, left_border_2, right_border_2, down_border_2]

    # origin of the terrain
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return meshes_list, origin



