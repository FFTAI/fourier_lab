from isaaclab.utils import configclass

from GRX_humanoid.tasks.locomotion.velocity.simple_wbc_env_cfg import \
    LocomotionVelocityRoughEnvCfg as WBCCFG
    
from GRX_humanoid.tasks.locomotion.velocity.simple_wbc_lower_env_cfg import \
    LocomotionVelocityRoughEnvCfg as LOWERCFG

from GRX_humanoid.tasks.locomotion.velocity.mask_wbc_env_cfg import \
    MaskLocomotionEnvCfg as MASK_WBC_CFG

from GRX_humanoid.terrains.terrain_generator_cfg import SIMPLE_TERRAINS_CFG, ROUGH_TERRAINS_CFG, ROUGH_TERRAINS_CFG_1, VISION_ROUGH_TERRAINS_CFG

##
# Pre-defined configs
##
from GRX_humanoid.assets.ppv233_humanoid import PPV233_HUMANOID_CFG, PPV233_HUMANOID_CFG_LOWER
import GRX_humanoid.tasks.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm


gr3_joint_names = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", \
                    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", \
                    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",  \
                    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",\
                    "waist_yaw_joint", "waist_roll_joint","waist_pitch_joint", \
                    "head_yaw_joint", "head_pitch_joint",\
                    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", \
                    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint", \
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", \
                    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"]
gr3_upper_joint_names = ["head_yaw_joint", "head_pitch_joint",\
                        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", \
                        "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint", \
                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", \
                        "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"]

gr3_lower_joint_names = ["waist_yaw_joint", "waist_roll_joint","waist_pitch_joint", \
                        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", \
                        "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", \
                        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",  \
                        "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]

feet_names = ".*foot_roll.*"

gr3_waist_joint_names = ["waist_yaw_joint", "waist_roll_joint","waist_pitch_joint"]
gr3_leg_joint_names = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", \
                            "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", \
                            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",  \
                            "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]

upper_weight = [0.1, 0.1,\
               0.5, 0.3, 0.3, 0.2, 0.1, 0.1, 0.1, \
               0.5, 0.3, 0.3, 0.2, 0.1, 0.1, 0.1]

gr3_stack_joint_names = [
                        "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint", \
                        "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"]

gr3_limbs_1_names = [ "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", \
                    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", \
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", \
                    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"]

gr3_limbs_2_names = [ "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",  \
                    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", \
                    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", \
                    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint"]
gr3_arm_joint_names = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", \
                        "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint", \
                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", \
                        "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"]

@configclass
class PPV233_Mask_WBC(MASK_WBC_CFG):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = PPV233_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = SIMPLE_TERRAINS_CFG
        # self.scene.num_envs = 20
        self.actions.joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=gr3_joint_names, scale=0.5, use_default_offset=True, preserve_order=True)

        self.commands.joint_pos_cmd = mdp.UniformJointPosCommandCfg(
            resampling_time_range=(10.0, 10.0),
            debug_vis=False,
            asset_cfg=SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True),
            ranges_scaled=(0.0, 0.0),
        )

        self.rewards.minimize_CoT_reward = RewTerm(func=mdp.minimize_CoT_reward,weight = 0.5,params={"std":0.5,"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_lower_joint_names,preserve_order=True)})
        self.rewards.upper_joint_stable = RewTerm(func=mdp.joint_vel_stable, weight=0.5, params={"std": 0.3, "asset_cfg": SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True)})
        self.rewards.upper_joint_total_track = RewTerm(func=mdp.joint_pos_total_track, weight=1.5, params={"std": 0.35, "asset_cfg": SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True)})
        self.rewards.upper_joint_track = RewTerm(func=mdp.joint_pos_track, weight=1.0, params={"std": 0.15, "weight": upper_weight, "asset_cfg": SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True)})
        self.rewards.torso_joint_pos = RewTerm(func=mdp.torso_joint_pos_l2_with_roll_balance, weight=-1.0, params={"std": 0.25, "asset_cfg": SceneEntityCfg("robot", joint_names=gr3_waist_joint_names,preserve_order=True)})
        # self.rewards.torso_joint_vel = RewTerm(func=mdp.torso_joint_vel_l2_with_roll_balance, weight=-0.2, params={"std": 0.25, "asset_cfg": SceneEntityCfg("robot", joint_names=gr3_waist_joint_names,preserve_order=True)})
        self.rewards.robot_limbs_centroidal_momentum_1 = RewTerm(func=mdp.robot_limbs_momentum_reward, weight=-0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_limbs_1_names,preserve_order=True)})
        self.rewards.robot_limbs_centroidal_momentum_2 = RewTerm(func=mdp.robot_limbs_momentum_reward, weight=-0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_limbs_2_names,preserve_order=True)})
        self.rewards.arm_deviation = RewTerm(func=mdp.arm_deviation_with_cmd_mask, weight=-0.15, params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_arm_joint_names,preserve_order=True)})
        self.rewards.hand_distance = RewTerm(func=mdp.hand_too_near_to_base, weight=-1.0, 
                                             params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_dummy_hand_link", "dummy_right_hand_link"],preserve_order=True), 
                                                     "threshold_x": 0.8, "threshold_y": 0.2})

        self.observations.policy.joint_pos = ObsTerm(func=mdp.joint_pos_rel,noise=Unoise(n_min=-0.02, n_max=0.02), scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.5, n_max=0.5), scale=0.05, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
        self.observations.policy.joint_pos_cmd = ObsTerm(func=mdp.joint_pos_cmd_rel, scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True)})
        self.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
        self.observations.critic.joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.05, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
        self.observations.critic.joint_pos_cmd = ObsTerm(func=mdp.joint_pos_cmd_rel, scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True)})  
        self.observations.critic.contact_mask = ObsTerm(func=mdp.is_feet_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet_names)})
        self.observations.critic.base_mass = ObsTerm(func=mdp.robot_base_link_mass, clip=(-100.0, 100.0), scale = 0.02, params={"asset_cfg": SceneEntityCfg("robot")})
        self.observations.critic.height_map = ObsTerm(func=mdp.height_scan, clip=(-100.0, 100.0), scale=2.0, params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.9})
        
        self.observations.plot.joint_vel = ObsTerm(func=mdp.joint_vel,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
        self.observations.plot.joint_torque = ObsTerm(func=mdp.joint_torque,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
        self.observations.plot.joint_power = ObsTerm(func=mdp.joint_power,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})

@configclass
class PPV233_Mask_WBC_Play(PPV233_Mask_WBC):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # self.commands.base_velocity.rel_standing_envs = 0.25
        self.commands.joint_pos_cmd = mdp.UniformJointPosCommandCfg(
            resampling_time_range=(3.0, 4.0),
            debug_vis=False,
            asset_cfg=SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True),
            ranges_scaled=(-0.8, 0.8),
        )
        # make a smaller scene for play
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum = None
        # self.scene.terrain.terrain_generator = SIMPLE_TERRAINS_CFG
        # self.scene.terrain.max_init_terrain_level = 10
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class PPV233HumanoidRoughEnvCfg_WBC_FULL(WBCCFG):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = PPV233_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = SIMPLE_TERRAINS_CFG
        self.actions.joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=gr3_joint_names, scale=0.5, use_default_offset=True, preserve_order=True)
        # self.actions.joint_pos = mdp.StackedJointActionsCfg(asset_name="robot", joint_names=gr3_joint_names, scale=0.5, use_default_offset=True, preserve_order=True,
        #                                                     stack_joint_names=gr3_stack_joint_names, 
        #                                                     command_joint_names=gr3_upper_joint_names, command_name="joint_pos_cmd")

        self.commands.joint_pos_cmd = mdp.UniformJointPosCommandCfg(
            resampling_time_range=(10.0, 10.0),
            debug_vis=False,
            asset_cfg=SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True),
            ranges_scaled=(0.0, 0.0),
        )
        self.rewards.minimize_CoT_reward = RewTerm(func=mdp.minimize_CoT_reward,weight = 1.0,params={"std":0.5,"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_lower_joint_names,preserve_order=True)})
        self.rewards.upper_joint_stable = RewTerm(func=mdp.joint_vel_stable, weight=0.5, params={"std": 0.3, "asset_cfg": SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True)})
        self.rewards.upper_joint_total_track = RewTerm(func=mdp.joint_pos_total_track, weight=1.5, params={"std": 0.35, "asset_cfg": SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True)})
        self.rewards.torso_joint_pos = RewTerm(func=mdp.torso_joint_pos_l2, weight=0.4, params={"std": 0.25, "asset_cfg": SceneEntityCfg("robot", joint_names=gr3_waist_joint_names,preserve_order=True)})

        self.observations.policy.joint_pos = ObsTerm(func=mdp.joint_pos_rel,noise=Unoise(n_min=-0.02, n_max=0.02), scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.0, n_max=1.0), scale=0.05, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
        self.observations.policy.joint_pos_cmd = ObsTerm(func=mdp.joint_pos_cmd_rel, scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True)})
        self.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
        self.observations.critic.joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.05, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
        self.observations.critic.joint_pos_cmd = ObsTerm(func=mdp.joint_pos_cmd_rel, scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True)})  
        self.observations.critic.contact_mask = ObsTerm(func=mdp.is_feet_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet_names)})
        self.observations.critic.base_mass = ObsTerm(func=mdp.robot_base_link_mass, clip=(-100.0, 100.0), scale = 0.02, params={"asset_cfg": SceneEntityCfg("robot")})
        self.observations.critic.height_map = ObsTerm(func=mdp.height_scan, clip=(-100.0, 100.0), scale=2.0, params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.9})
        
        self.observations.plot.joint_vel = ObsTerm(func=mdp.joint_vel,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
        self.observations.plot.joint_torque = ObsTerm(func=mdp.joint_torque,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
        self.observations.plot.joint_power = ObsTerm(func=mdp.joint_power,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_joint_names,preserve_order=True)})
@configclass
class PPV233HumanoidRoughEnvCfg_WBC_FULL_Play(PPV233HumanoidRoughEnvCfg_WBC_FULL):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # self.commands.base_velocity.rel_standing_envs = 1.0

        self.commands.height_attitude = mdp.UniformHRCommandCfg(
            resampling_time_range=(5.0, 10.0),  # Fixed resampling time of 5 seconds
            debug_vis=False,  # No debug visualization needed
            ranges=mdp.UniformHRCommandCfg.Ranges(
                height = (0.75, 0.95),
                # torso_yaw = (0., 0.),
                # torso_roll = (0., 0.),
                # torso_pitch = (0., 0.),
                torso_yaw = (-0, 0),
                torso_roll = (-0, 0),
                torso_pitch = (-0.0, 0.0),
            ),
        )
        self.commands.joint_pos_cmd = mdp.UniformJointPosCommandCfg(
            resampling_time_range=(3.0, 4.0),
            debug_vis=False,
            asset_cfg=SceneEntityCfg("robot", joint_names=gr3_upper_joint_names,preserve_order=True),
            ranges_scaled=(-1.0, 1.0),
        )
        # make a smaller scene for play
        self.scene.num_envs = 6
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum = None
        # self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_1
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
@configclass
class PPV233HumanoidRoughEnvCfg_WBC_LOWER(LOWERCFG):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = PPV233_HUMANOID_CFG_LOWER.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = SIMPLE_TERRAINS_CFG
        self.actions.joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=gr3_leg_joint_names, scale=0.5, use_default_offset=True, preserve_order=True)
        self.observations.policy.joint_pos = ObsTerm(func=mdp.joint_pos_rel,noise=Unoise(n_min=-0.02, n_max=0.02), scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_leg_joint_names,preserve_order=True)})
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.0, n_max=1.0), scale=0.05, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_leg_joint_names,preserve_order=True)})
        self.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_leg_joint_names,preserve_order=True)})
        self.observations.critic.joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.05, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_leg_joint_names,preserve_order=True)})
        self.observations.critic.contact_mask = ObsTerm(func=mdp.is_feet_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet_names)})   
        self.observations.critic.base_mass = ObsTerm(func=mdp.robot_base_link_mass, clip=(-100.0, 100.0), scale = 0.02, params={"asset_cfg": SceneEntityCfg("robot")})
        self.observations.critic.height_map = ObsTerm(func=mdp.height_scan, clip=(-100.0, 100.0), scale=2.0, params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.9})
        
        self.observations.plot.joint_vel = ObsTerm(func=mdp.joint_vel,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_leg_joint_names,preserve_order=True)})
        self.observations.plot.joint_torque = ObsTerm(func=mdp.joint_torque,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_leg_joint_names,preserve_order=True)})
        self.observations.plot.joint_power = ObsTerm(func=mdp.joint_power,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", joint_names=gr3_leg_joint_names,preserve_order=True)})
@configclass
class PPV233HumanoidRoughEnvCfg_WBC_LOWER_Play(PPV233HumanoidRoughEnvCfg_WBC_LOWER):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 10.0),
            rel_standing_envs=0.2,
            rel_heading_envs=1.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0, 1.0), lin_vel_y=(-0., 0.), ang_vel_z=(-0., 0.)
            ),
        )
        
        self.commands.height_attitude = mdp.UniformHACommandCfg(
            resampling_time_range=(5.0, 10.0),  # Fixed resampling time of 5 seconds
            debug_vis=False,  # No debug visualization needed
            ranges=mdp.UniformHACommandCfg.Ranges(
                height = (0.88, 0.88),
                pitch_angle = (-0., 0.),
                yaw_angle=(-0.5, 0.5)
            ),
        )
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_generator = SIMPLE_TERRAINS_CFG
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None