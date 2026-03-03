from isaaclab.utils import configclass
    
from GRX_humanoid.tasks.locomotion.velocity.simple_wbc_lower_env_cfg import \
    LocomotionVelocityRoughEnvCfg as LOWERCFG

from GRX_humanoid.terrains.terrain_generator_cfg import SIMPLE_TERRAINS_CFG,ROUGH_TERRAINS_CFG_1

##
# Pre-defined configs
##
from GRX_humanoid.assets.gr2t2v2_humanoid import GR2T2V2_HUMANOID_CFG, GR2T2V2_HUMANOID_CFG_LOWER
import GRX_humanoid.tasks.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
import GRX_humanoid.tasks.locomotion.velocity.mdp as mdp


gr2_joint_names = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", \
                        "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", \
                        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",  \
                        "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",\
                        "waist_yaw_joint", \
                        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", \
                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint"]

GR2T2V2_lower_joint_names = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", \
                        "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", \
                        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",  \
                        "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]

feet_names = ".*foot_roll.*"
@configclass
class GR2T2V2HumanoidRoughEnvCfg_WBC_LOWER(LOWERCFG):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = GR2T2V2_HUMANOID_CFG_LOWER.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = SIMPLE_TERRAINS_CFG
        self.actions.joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=GR2T2V2_lower_joint_names, scale=0.5, use_default_offset=True, preserve_order=True)
        self.observations.policy.joint_pos = ObsTerm(func=mdp.joint_pos_rel,noise=Unoise(n_min=-0.05, n_max=0.05), scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=GR2T2V2_lower_joint_names,preserve_order=True)})
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.6, n_max=0.6), scale=0.05, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=GR2T2V2_lower_joint_names,preserve_order=True)})
        self.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=0.5, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=GR2T2V2_lower_joint_names,preserve_order=True)})
        self.observations.critic.joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.05, clip=(-100.0, 100.0), params={"asset_cfg": SceneEntityCfg("robot", joint_names=GR2T2V2_lower_joint_names,preserve_order=True)})
        self.observations.critic.contact_mask = ObsTerm(func=mdp.is_feet_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet_names)})   
        self.observations.critic.base_mass = ObsTerm(func=mdp.robot_base_link_mass, clip=(-100.0, 100.0), scale = 0.02, params={"asset_cfg": SceneEntityCfg("robot")})
        self.observations.critic.height_map = ObsTerm(func=mdp.height_scan, clip=(-100.0, 100.0), scale=2.0, params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.9})
        
        self.observations.plot.joint_vel = ObsTerm(func=mdp.joint_vel,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", joint_names=GR2T2V2_lower_joint_names,preserve_order=True)})
        self.observations.plot.joint_torque = ObsTerm(func=mdp.joint_torque,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", joint_names=GR2T2V2_lower_joint_names,preserve_order=True)})
        self.observations.plot.joint_power = ObsTerm(func=mdp.joint_power,clip=(-100.0, 100.0),params={"asset_cfg": SceneEntityCfg("robot", joint_names=GR2T2V2_lower_joint_names,preserve_order=True)})

        self.rewards.feet_air_time = RewTerm(func=mdp.feet_air_time_positive_biped,weight=3.0,params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_roll.*"),"command_name": "base_velocity","threshold": 0.5,},)
        self.curriculum.task_stage = CurrTerm(
            func=mdp.modify_lower_stage_cmd,
            params={
                "height_range":(0.45, 0.97),
                "pitch_range":(-0.3, 0.5),
                "yaw_range":(-0.5, 0.5)
            },
        )
@configclass
class GR2T2V2HumanoidRoughEnvCfg_WBC_LOWER_Play(GR2T2V2HumanoidRoughEnvCfg_WBC_LOWER):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.commands.height_attitude = mdp.UniformHACommandCfg(
            resampling_time_range=(5.0, 10.0),  # Fixed resampling time of 5 seconds
            debug_vis=False,  # No debug visualization needed
            ranges=mdp.UniformHACommandCfg.Ranges(
                height = (0.45, 0.97),
                pitch_angle = (-0.3, 0.5),
                yaw_angle = (-0.5, 0.5)
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

