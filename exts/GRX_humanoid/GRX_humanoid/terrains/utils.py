# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | torch.Tensor | None
from __future__ import annotations

import numpy as np
import torch
import trimesh
import random
from typing import Union, Tuple
from scipy import interpolate

import warp as wp

from isaaclab.utils.warp import raycast_mesh

def get_cfg_value(cfg_value) -> float:
    if (
        isinstance(cfg_value, tuple)
        and len(cfg_value) == 2
        and all(isinstance(x, (float, int)) for x in cfg_value)
    ):
        min_val, max_val = min(cfg_value), max(cfg_value)
        return random.uniform(min_val, max_val)

    elif isinstance(cfg_value, (float, int)):
        return float(cfg_value)

    else:
        raise ValueError(
            "cfg.cfg_value must be one of tuple[float, float] or float, "
            f"but gets {type(cfg_value)}"
        )

def generate_rough_surface(box_dims, box_pos, cfg):
    """Generate a rough surface for the top of a box with downsampled height field."""
    # Extract box dimensions
    box_length, box_width, box_height = box_dims

    # Calculate the number of pixels for the full resolution
    width_pixels = int(box_length / cfg.horizontal_scale)
    length_pixels = int(box_width / cfg.horizontal_scale)

    # Calculate the number of pixels for the downsampled resolution
    width_downsampled = int(box_length / cfg.downsampled_scale)
    length_downsampled = int(box_width / cfg.downsampled_scale)

    # Generate the range of discrete heights
    height_min, height_max = cfg.noise_range
    height_step = cfg.noise_step  # Step size for discrete heights
    height_range = np.arange(height_min, height_max + height_step, height_step)

    # Generate the downsampled height field with discrete values
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))

    # Create interpolation function for the downsampled height field
    x_downsampled = np.linspace(-box_length / 2, box_length / 2, width_downsampled) + box_pos[0]
    y_downsampled = np.linspace(-box_width / 2, box_width / 2, length_downsampled) + box_pos[1]
    interp_func = interpolate.RectBivariateSpline(x_downsampled, y_downsampled, height_field_downsampled)

    # Interpolate to the full resolution
    x_full = np.linspace(-box_length / 2, box_length / 2, width_pixels) + box_pos[0]
    y_full = np.linspace(-box_width / 2, box_width / 2, length_pixels) + box_pos[1]
    x, y = np.meshgrid(x_full, y_full)
    z = interp_func(x_full, y_full).T  # Transpose to match meshgrid output

    # Round the interpolated heights to the nearest discrete value
    z = np.rint(z / height_step) * height_step

    # Adjust the height field to the box's top position
    z += box_pos[2] + box_height / 2

    return x, y, z

def height_field_to_mesh(x, y, z):
    """Convert a height field to a triangular mesh."""
    # Flatten the grid
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Create faces for the grid
    rows, cols = x.shape
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Define two triangles for each grid cell
            v0 = i * cols + j
            v1 = v0 + 1
            v2 = v0 + cols
            v3 = v2 + 1
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    faces = np.array(faces)

    # Create a trimesh object
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def create_rough_box(box_dims, box_pos, cfg):
    """Create a box with a rough surface on top, ensuring proper connection and average z-dimension consistency."""
    # Extract box dimensions
    box_length, box_width, box_height = box_dims

    # Adjust the box height to compensate for the negative noise range
    adjusted_box_height = box_height + cfg.noise_range[0]

    # Generate the rough surface
    x, y, z = generate_rough_surface(box_dims, box_pos, cfg)
    rough_surface = height_field_to_mesh(x, y, z)

    # Create the adjusted box (with compensated height)
    adjusted_box_dims = (box_length, box_width, adjusted_box_height)
    adjusted_box_pos = (box_pos[0], box_pos[1], box_pos[2] + cfg.noise_range[0] / 2)
    box = trimesh.creation.box(adjusted_box_dims, trimesh.transformations.translation_matrix(adjusted_box_pos))

    # Generate side faces to connect the rough surface with the box
    side_faces = []

    # Bottom vertices of the box's top face
    box_top_z = box_pos[2] + adjusted_box_height / 2

    # Process the four edges of the rough surface
    rows, cols = x.shape

    # Front edge (y = min)
    for j in range(cols - 1):
        v0 = [x[0, j], y[0, j], box_top_z]  # Bottom-left corner
        v1 = [x[0, j + 1], y[0, j + 1], box_top_z]  # Bottom-right corner
        v2 = [x[0, j], y[0, j], z[0, j]]  # Top-left corner
        v3 = [x[0, j + 1], y[0, j + 1], z[0, j + 1]]  # Top-right corner
        side_faces.append([v0, v2, v1])
        side_faces.append([v1, v2, v3])

    # Back edge (y = max)
    for j in range(cols - 1):
        v0 = [x[-1, j], y[-1, j], box_top_z]
        v1 = [x[-1, j + 1], y[-1, j + 1], box_top_z]
        v2 = [x[-1, j], y[-1, j], z[-1, j]]
        v3 = [x[-1, j + 1], y[-1, j + 1], z[-1, j + 1]]
        side_faces.append([v0, v2, v1])
        side_faces.append([v1, v2, v3])

    # Left edge (x = min)
    for i in range(rows - 1):
        v0 = [x[i, 0], y[i, 0], box_top_z]
        v1 = [x[i + 1, 0], y[i + 1, 0], box_top_z]
        v2 = [x[i, 0], y[i, 0], z[i, 0]]
        v3 = [x[i + 1, 0], y[i + 1, 0], z[i + 1, 0]]
        side_faces.append([v0, v2, v1])
        side_faces.append([v1, v2, v3])

    # Right edge (x = max)
    for i in range(rows - 1):
        v0 = [x[i, -1], y[i, -1], box_top_z]
        v1 = [x[i + 1, -1], y[i + 1, -1], box_top_z]
        v2 = [x[i, -1], y[i, -1], z[i, -1]]
        v3 = [x[i + 1, -1], y[i + 1, -1], z[i + 1, -1]]
        side_faces.append([v0, v2, v1])
        side_faces.append([v1, v2, v3])

    # Convert side faces to a mesh
    side_vertices = np.array([v for face in side_faces for v in face])
    side_faces = np.array([[i, i + 1, i + 2] for i in range(0, len(side_vertices), 3)])
    side_mesh = trimesh.Trimesh(vertices=side_vertices, faces=side_faces)

    # Combine the box, rough surface, and side faces
    combined = trimesh.util.concatenate([box, rough_surface, side_mesh])

    return combined

def create_box_with_optional_rough_surface(if_rough, box_dims, box_pos, cfg):
    """Create a box with or without a rough surface based on cfg.if_rough."""
    if if_rough:
        # Use create_rough_box and pass cfg.rough_surface_cfg
        return create_rough_box(box_dims, box_pos, cfg.rough_surface_cfg)
    else:
        # Use trimesh.creation.box
        return trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))

def my_make_border(if_rough, size: tuple[float, float], inner_size: tuple[float, float], height: float, position: tuple[float, float, float], cfg) -> list[trimesh.Trimesh]:
    """Generate meshes for a rectangular border with a hole in the middle.

    .. code:: text

        +---------------------+
        |#####################|
        |##+---------------+##|
        |##|               |##|
        |##|               |##| length
        |##|               |##| (y-axis)
        |##|               |##|
        |##+---------------+##|
        |#####################|
        +---------------------+
              width (x-axis)

    Args:
        if_rough: If True, the border will have a rough surface on top.
        size: The length (along x) and width (along y) of the terrain (in m).
        inner_size: The inner length (along x) and width (along y) of the hole (in m).
        height: The height of the border (in m).
        position: The center of the border (in m).
        cfg: Configuration for generating rough surface if if_rough is True.

    Returns:
        A list of trimesh.Trimesh objects that represent the border.
    """
    # compute thickness of the border
    thickness_x = (size[0] - inner_size[0]) / 2.0
    thickness_y = (size[1] - inner_size[1]) / 2.0
    # generate tri-meshes for the border
    meshes = []
    # top/bottom border
    box_dims = (size[0], thickness_y, height)
    # -- top
    box_pos_top = (position[0], position[1] + inner_size[1] / 2.0 + thickness_y / 2.0, position[2])
    box_mesh_top = create_box_with_optional_rough_surface(False, box_dims, box_pos_top, cfg)
    meshes.append(box_mesh_top)
    # -- bottom
    box_pos_bottom = (position[0], position[1] - inner_size[1] / 2.0 - thickness_y / 2.0, position[2])
    box_mesh_bottom = create_box_with_optional_rough_surface(False, box_dims, box_pos_bottom, cfg)
    meshes.append(box_mesh_bottom)
    # left/right border
    box_dims = (thickness_x, inner_size[1], height)
    # -- left
    box_pos_left = (position[0] - inner_size[0] / 2.0 - thickness_x / 2.0, position[1], position[2])
    box_mesh_left = create_box_with_optional_rough_surface(if_rough, box_dims, box_pos_left, cfg)
    meshes.append(box_mesh_left)
    # -- right
    box_pos_right = (position[0] + inner_size[0] / 2.0 + thickness_x / 2.0, position[1], position[2])
    box_mesh_right = create_box_with_optional_rough_surface(if_rough, box_dims, box_pos_right, cfg)
    meshes.append(box_mesh_right)
    # return the tri-meshes
    return meshes

