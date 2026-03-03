from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, ContactSensorCfg, patterns, TiledCameraCfg, CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import GRX_humanoid.tasks.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from GRX_humanoid.terrains.terrain_generator_cfg import ROUGH_TERRAINS_CFG, GRAVEL_TERRAINS_CFG, HOVER_ROUGH_TERRAINS_CFG


##
# Scene definition
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=10,
                                                track_air_time=True, track_pose=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.CustomVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.7,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.CustomVelocityCommandCfg.Ranges(
            zero_prob=(0.1, 0.3, 0.3), lin_vel_x=(-1.3, 1.3), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-0.8, 0.8), heading=(-3.14, 3.14)
        ),
    )

    height_attitude = mdp.UniformHACommandCfg(
        resampling_time_range=(5.0, 10.0),  # Fixed resampling time of 5 seconds
        debug_vis=False,  # No debug visualization needed
        ranges=mdp.UniformHACommandCfg.Ranges(
            height = (0.88, 0.88),
            pitch_angle = (0., 0.),
            yaw_angle=(0., 0.)
        ),
    )

class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", clip=(-100.0, 100.0), joint_names=".*", scale=0.5, use_default_offset=True, preserve_order=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0)
        )
        is_walk_int = ObsTerm(func=mdp.is_walk_int, params={"command_name": "base_velocity"})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},clip=(-100.0, 100.0))
        height_attitude =  ObsTerm(func=mdp.generated_commands, params={"command_name": "height_attitude"},clip=(-100.0, 100.0))

        joint_pos = ObsTerm(func=mdp.joint_pos_rel,noise=Unoise(n_min=-0.05, n_max=0.05), scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*",preserve_order=True)})
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.05, n_max=0.05), scale=0.05, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*",preserve_order=True)})
        actions = ObsTerm(func=mdp.last_action,clip=(-100.0, 100.0))
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 40 #if history_length>0 then the obs will be AAABBBCCCDDD (eg. 3)

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_vel = ObsTerm(func=mdp.base_lin_vel,clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,clip=(-100.0, 100.0)
        )
        is_walk_int = ObsTerm(func=mdp.is_walk_int, params={"command_name": "base_velocity"})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},clip=(-100.0, 100.0))
        height_attitude =  ObsTerm(func=mdp.generated_commands, params={"command_name": "height_attitude"},clip=(-100.0, 100.0))

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*",preserve_order=True)})
        joint_vel = ObsTerm(func=mdp.joint_vel,  scale=0.05, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*",preserve_order=True)})
        actions = ObsTerm(func=mdp.last_action,clip=(-100.0, 100.0))
        contact_mask = ObsTerm(func=mdp.is_feet_contact, 
                               params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_roll.*")})
        base_mass = ObsTerm(func=mdp.robot_base_link_mass, scale = 0.02, clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot")})
        height_map = ObsTerm(func=mdp.height_scan, scale=2.0, clip=(-100.0, 100.0),params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.9})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5 #if history_length>0 then the obs will be AAABBBCCCDDD (eg. 3)

    @configclass
    class PlotCfg(ObsGroup):
        """Observations for plot group."""

        # observation terms (order preserved)
        joint_vel = ObsTerm(func=mdp.joint_vel,clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*",preserve_order=True)})
        joint_torque = ObsTerm(func=mdp.joint_torque,clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*",preserve_order=True)})
        joint_power = ObsTerm(func=mdp.joint_power,clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*",preserve_order=True)})
        feet_acc = ObsTerm(func=mdp.robot_feet_acc,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot_roll.*")})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1
    

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    plot: PlotCfg = PlotCfg()



@configclass
class EventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    scale_all_link_masses = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )
    
    scale_all_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={"com_range": {"x": (-0.04, 0.04),"y": (-0.03, 0.03),"z": (-0.03, 0.03),},
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),},
    )

    scale_all_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "armature_distribution_params": (0.9, 1.1),"friction_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )

    scale_all_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.1, 0.1), "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.5, 0.5),}},
    )

    # push_robot_by_force = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(13.0, 14.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "force_range": (-50, 50),
    #         "torque_range": (-20.0, 20.0),
    #     },
    # )

    # stop_robot_by_force = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(14.1, 15.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (0.0, 0.0),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    #task reward
    track_base_height_scanner_exp = RewTerm(func=mdp.track_base_height_scanner_exp_ha, weight=1.5,params={"std":0.1},)
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"command_name": "base_velocity", "std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.7})
    flat_orientation_exp = RewTerm(func=mdp.flat_orientation_exp_ha, weight=0.3, params={"std": 0.3})
    track_base_foot_yaw_exp = RewTerm(func=mdp.track_base_foot_yaw_exp_ha, weight=0.7, params={"std": 0.2, "asset_cfg": SceneEntityCfg("robot", body_names=[".*foot_roll.*"])})
    
    #contact reward
    fly = RewTerm(func=mdp.fly, weight=-0.2, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_roll.*"), "threshold": 30.0})
    feet_contact_num = RewTerm(func=mdp.FeetContactNumReward, weight=4.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_roll.*"), "threshold": 30.0, "history_length":10, "command_name": "base_velocity",})
    feet_air_time = RewTerm(func=mdp.feet_air_time_positive_biped,weight=2.0,params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_roll.*"),"command_name": "base_velocity","threshold": 0.5,"delta": 0.05},)
    feet_air_time_symmetry = RewTerm(func=mdp.feet_air_time_symmetry_biped,weight=-1.0,params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_roll.*"),"command_name": "base_velocity"},)
    
    #foot constraint reward
    feet_roll = RewTerm(func=mdp.flat_link_roll_exp, weight=0.1, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot_roll.*"), "std": 0.3})
    foot_alignment = RewTerm(func=mdp.foot_yaw_alignment_reward, weight=0.5,params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot_roll.*"),"max_angle": 30.0 ,"always_active": False})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.25, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_roll.*"), "asset_cfg": SceneEntityCfg("robot", body_names=".*foot_roll.*")})
    feet_force = RewTerm(func=mdp.feet_force, weight=-1e-2, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_roll.*"),  "max_reward": 1000})
    feet_too_near = RewTerm(func=mdp.feet_too_near_humanoid, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*foot_roll.*"]), "threshold": 0.15})
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-2.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_roll.*")})
    contact_moment = RewTerm(func=mdp.contact_moment, weight=-2.0, params={"std": 100.0, "threshold": 1000.0, "asset_cfg": SceneEntityCfg("robot", body_names=".*foot_roll.*"), "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_roll.*")})

    #regularization reward
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_world_l2 = RewTerm(func=mdp.ang_vel_xy_world_l2, weight=-0.2)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    dof_torque = RewTerm(func=mdp.torque, weight=-0.03)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    minimize_CoT_reward = RewTerm(func=mdp.minimize_CoT_reward,weight = 0.5,params={"std":1.0,"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}, )
    center_of_mass = RewTerm(func=mdp.center_of_mass, weight=0.7, params={"std": 0.1, "asset_cfg": SceneEntityCfg("robot", body_names=[".*foot_roll.*"])})
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-3.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*foot_roll.*).*"), "threshold": 1.0})
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,  
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"), 
            "limit_angle": 0.7,  
        },
    )
    base_height_below_minimum = DoneTerm(
        func=mdp.base_height_below_minimum,  
        params={"sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "minimum_height": 0.3}
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    task_stage = CurrTerm(
        func=mdp.modify_lower_stage_cmd,
        params={
            "height_range":(0.45, 0.92),
            "pitch_range":(-0.3, 0.5),
            "yaw_range":(-0.4, 0.4)
        },
    )


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
                print("test curriculum")
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
