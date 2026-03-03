import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.assets.articulation import ArticulationCfg

from GRX_humanoid.assets import ISAAC_ASSET_DIR

GR2T2V2_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/gr2t2v2_humanoid.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            ".*_hip_roll_joint": 0.0, 
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_pitch_joint": -0.2618,
            ".*knee_pitch_joint": 0.5236,
            ".*_ankle_pitch_joint": -0.2618,
            ".*_ankle_roll_joint": 0.0,
            "waist.*": 0.0,
            ".*_shoulder_pitch.*": 0.0,
            ".*_shoulder_roll.*": 0.0,
            ".*_shoulder_yaw.*": 0.0,
            ".*_elbow_.*": 0.0,
            # ".*_wrist_.*": 0.0,
            # "head_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_pitch_joint",
                "waist.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 54.33,
                ".*_hip_roll_joint": 95.472,
                ".*_hip_pitch_joint": 366.05,
                ".*_knee_pitch_joint": 366.05,
                "waist.*": 74.45,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 14.745,
                ".*_hip_roll_joint": 12.362,
                ".*_hip_pitch_joint": 6.4997,
                ".*_knee_pitch_joint": 6.4997,
                "waist.*": 7.7568,
            },
            stiffness={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 300.0,
                ".*_knee_pitch_joint": 300.0,
                "waist.*": 300,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 6.0,
                ".*_knee_pitch_joint": 6.0,
                "waist.*": 6,
            },
            armature={
                ".*_hip_yaw_joint": 0.165,
                ".*_hip_roll_joint": 0.121,
                ".*_hip_pitch_joint": 0.592,
                ".*_knee_pitch_joint": 0.121,
                "waist.*": 0.606,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 54.33, 
                ".*_ankle_roll_joint": 29.835,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 14.745, 
                ".*_ankle_roll_joint": 16.755,
            },
            stiffness={
                ".*_ankle_pitch_joint": 40, 
                ".*_ankle_roll_joint": 40,
            },
            damping={
                ".*_ankle_pitch_joint": 2, 
                ".*_ankle_roll_joint": 2,
            },
            armature={
                ".*_ankle_pitch_joint": 0.165, 
                ".*_ankle_roll_joint": 0.031,
            }
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch.*",
                ".*_shoulder_roll.*",
            ],
            effort_limit_sim=74.45,
            velocity_limit_sim = 7.7568,
            stiffness=200.0,
            damping=5.0,
            armature=0.606,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw.*",
                ".*_elbow_.*",
            ],
            effort_limit_sim=42.75,
            velocity_limit_sim = 6.238,
            stiffness=50.0,
            damping=2.0,
            armature=0.222,
        ),
    },
)

GR2T2V2_HUMANOID_CFG_LOWER = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/gr2t2v2_humanoid_lower.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            ".*_hip_roll_joint": 0.0, 
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_pitch_joint": -0.2618,
            ".*knee_pitch_joint": 0.5236,
            ".*_ankle_pitch_joint": -0.2618,
            ".*_ankle_roll_joint": 0.0,
            # "waist.*": 0.0,
            # ".*_shoulder_pitch.*": 0.0,
            # ".*_shoulder_roll.*": 0.0,
            # ".*_shoulder_yaw.*": 0.0,
            # ".*_elbow_.*": 0.0,
            # ".*_wrist_.*": 0.0,
            # "head_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_pitch_joint",
                #"waist.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 54.33,
                ".*_hip_roll_joint": 95.472,
                ".*_hip_pitch_joint": 366.05,
                ".*_knee_pitch_joint": 366.05,
                #"waist.*": 74.45,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 14.745,
                ".*_hip_roll_joint": 12.362,
                ".*_hip_pitch_joint": 6.4997,
                ".*_knee_pitch_joint": 6.4997,
                #"waist.*": 7.7568,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_pitch_joint": 200.0,
                #"waist.*": 300,
            },
            damping={
                ".*_hip_yaw_joint": 10.0,
                ".*_hip_roll_joint": 10.0,
                ".*_hip_pitch_joint": 20.0,
                ".*_knee_pitch_joint": 20.0,
                #"waist.*": 6,
            },
            armature={
                ".*_hip_yaw_joint": 0.165,
                ".*_hip_roll_joint": 0.121,
                ".*_hip_pitch_joint": 0.592,
                ".*_knee_pitch_joint": 0.121,
                #"waist.*": 0.606,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 40, 
                ".*_ankle_roll_joint": 20,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 14.745, 
                ".*_ankle_roll_joint": 16.755,
            },
            stiffness={
                ".*_ankle_pitch_joint": 60, 
                ".*_ankle_roll_joint": 30,
            },
            damping={
                ".*_ankle_pitch_joint": 5, 
                ".*_ankle_roll_joint": 5,
            },
            armature={
                ".*_ankle_pitch_joint": 0.165, 
                ".*_ankle_roll_joint": 0.031,
            }
        ),
    },
)