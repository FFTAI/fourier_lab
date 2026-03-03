# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

import GRX_humanoid.terrains.trimesh.mesh_terrains as mesh_terrains
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

@configclass
class RoughSurfaceCfg():
    """Configuration for generating a rough surface on top of a box."""

    horizontal_scale: float = 0.1
    """The horizontal scale (in m) for the high-resolution height field grid."""
    
    downsampled_scale: float = 0.1
    """The horizontal scale (in m) for the downsampled height field grid."""
    
    noise_range: tuple[float, float] = (0.0, 0.03)
    """The range of heights (in m) for the rough surface."""
    
    noise_step: float = 0.01
    """The step size (in m) for discrete heights in the rough surface."""

"""
Different trimesh terrain configurations.
"""



@configclass
class MyMeshPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""

    function = mesh_terrains.pyramid_stairs_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """
    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""
    step_width: float | tuple[float, float] = MISSING
    """The width of the steps (in m)."""
    platform_width: float | tuple[float, float] = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.
    """


@configclass
class MyMeshInvertedPyramidStairsTerrainCfg(MyMeshPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stair mesh terrain.

    Note:
        This is the same as :class:`MeshPyramidStairsTerrainCfg` except that the steps are inverted.
    """

    function = mesh_terrains.inverted_pyramid_stairs_terrain

@configclass
class MyMeshRidgeStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""

    function = mesh_terrains.ridge_stairs_terrain

    border_width_x: tuple[float, float] = MISSING
    border_width_y: tuple[float, float] = MISSING
    """The width of the border around the terrain in x and y direction (in m).

    The border is a flat terrain with the same height as the terrain.
    """
    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""
    step_width: float | tuple[float, float] = MISSING
    """The width of the steps (in m)."""
    platform_length: float | tuple[float, float] = 1.0
    """The length of the square platform at the center of the terrain. Defaults to 1.0."""
    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.
    """
    rough_surface_cfg: RoughSurfaceCfg  = RoughSurfaceCfg()
    if_rough: bool = False


@configclass
class MyMeshInvertedRidgeStairsTerrainCfg(MyMeshRidgeStairsTerrainCfg):
    """Configuration for an inverted pyramid stair mesh terrain.

    Note:
        This is the same as :class:`MeshPyramidStairsTerrainCfg` except that the steps are inverted.
    """

    function = mesh_terrains.inverted_ridge_stairs_terrain

@configclass
class MyMeshPitTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pit mesh terrain."""

    function = mesh_terrains.pit_terrain

    pit_depth_range: tuple[float, float] = MISSING
    """The minimum and maximum depth of the pit (in m)."""
    pit_width: float | tuple[float, float] = MISSING
    """The width of the pit (in m)."""
    pit_length: float | tuple[float, float] = MISSING
    """The length of the pit (in m)."""
    platform_length: float | tuple[float, float] = 2.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    rough_surface_cfg: RoughSurfaceCfg  = RoughSurfaceCfg()
    if_rough: bool = False



@configclass
class MyMeshGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a gap mesh terrain."""

    function = mesh_terrains.gap_terrain

    gap_depth: float | tuple[float, float] = MISSING
    """The depth of the gap (in m)."""
    gap_width: float | tuple[float, float] = MISSING
    """The width of the gap (in m)."""
    gap_length_range: tuple[float, float] = MISSING
    """The minimum and maximum length of the gap (in m)."""
    platform_length: float | tuple[float, float] = 2.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    rough_surface_cfg: RoughSurfaceCfg  = RoughSurfaceCfg()
    if_rough: bool = False