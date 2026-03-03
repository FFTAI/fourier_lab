"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from .trimesh import MyMeshPyramidStairsTerrainCfg, MyMeshInvertedPyramidStairsTerrainCfg, MyMeshRidgeStairsTerrainCfg, \
      MyMeshRidgeStairsTerrainCfg, MyMeshInvertedRidgeStairsTerrainCfg, MyMeshPitTerrainCfg, MyMeshGapTerrainCfg, \
      RoughSurfaceCfg

HOVER_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.4, size=(8.0, 8.0)),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.65, grid_height_range=(0.015, 0.03), platform_width=2.0, size=(8.0, 8.0)
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25, size=(8.0, 8.0)
        ),
    },
)

GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.4, size=(8.0, 8.0)),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.2), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.2), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.0, 0.1), num_waves=4, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.0, 0.06), noise_step=0.02, border_width=0.25
        ),
    },
)

ROUGH_TERRAINS_CFG_1 = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        ),
        "pyramid_stairs": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=0.4,
            platform_width=3.0,
            border_width=1.0,
        ),
        "pyramid_stairs_inv": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=0.4,
            platform_width=3.0,
            border_width=1.0,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.1, amplitude_range=(0.05,0.25), num_waves=4,border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)

SIMPLE_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2, size=(8.0, 8.0)),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.15), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.15), platform_width=2.0, border_width=0.25
        ),
    },
)

VISION_FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.15), platform_width=1.5, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.15), platform_width=1.5, border_width=0.25
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.3, amplitude_range=(0.0, 0.05), num_waves=4, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(0.0, 0.04), noise_step=0.02, border_width=0.25
        ),
    },
)

VISION_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(10.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pit_terrain": MyMeshPitTerrainCfg(
            proportion=0.0,
            pit_depth_range=(0.005, 0.3),
            pit_width=(1.0, 3.0),
            pit_length=(0.5, 1.5),
            platform_length=2.5,
            if_rough=True,
            rough_surface_cfg=RoughSurfaceCfg(horizontal_scale=0.05, downsampled_scale=0.05, noise_range=(0.0, 0.02), noise_step=0.005),
        ),
        "gap_terrain": MyMeshGapTerrainCfg(
            proportion=0.0,
            gap_length_range=(0.0, 0.7),
            gap_depth=(0.4, 0.8),
            gap_width=(1.5, 3.5),
            platform_length=2.5,
            if_rough=True,
            rough_surface_cfg=RoughSurfaceCfg(horizontal_scale=0.05, downsampled_scale=0.05, noise_range=(0.0, 0.02), noise_step=0.005),
        ),
        "pyramid_stairs": MyMeshRidgeStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.005, 0.25),
            step_width=(0.3, 0.45),
            platform_length=(1.5, 6.0),
            border_width_x=(0.5, 1.5),
            border_width_y=(0.0, 1.25),
            holes=False,
            if_rough=True,
            rough_surface_cfg=RoughSurfaceCfg(horizontal_scale=0.05, downsampled_scale=0.05, noise_range=(0.0, 0.02), noise_step=0.005),
        ),
        "pyramid_stairs_inv": MyMeshInvertedRidgeStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.005, 0.25),
            step_width=(0.3, 0.45),
            platform_length=(1.5, 6.0),
            border_width_x=(0.5, 1.5),
            border_width_y=(0.0, 1.25),
            holes=False,
            if_rough=True,
            rough_surface_cfg=RoughSurfaceCfg(horizontal_scale=0.05, downsampled_scale=0.05, noise_range=(0.0, 0.02), noise_step=0.005),
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.1, amplitude_range=(0.0, 0.075), num_waves=4, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.0, 0.04), noise_step=0.02, border_width=0.25
        ),
    },
)