import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from GRX_humanoid.assets import ISAAC_ASSET_DIR


FOURIERN1_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/fouriern1_humanoid.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_hip_roll_joint": 0.0, 
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_pitch_joint": -0.2618,
            ".*_knee_pitch_joint": 0.5236,
            ".*_ankle_pitch_joint": -0.2618,
            ".*_ankle_roll_joint": 0.0,
            "waist.*": 0.0,
            ".*_shoulder_pitch.*": 0.0,
            ".*_shoulder_roll.*": 0.0,
            ".*_shoulder_yaw.*": 0.0,
            ".*_elbow_.*": 0.0,
            ".*_wrist_.*": 0.0,
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
                ".*_hip_yaw_joint": 54.0,
                ".*_hip_roll_joint": 54.0,
                ".*_hip_pitch_joint": 95,
                ".*_knee_pitch_joint": 95,
                "waist.*": 54,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 14.738,
                ".*_hip_roll_joint": 14.738,
                ".*_hip_pitch_joint": 12.356,
                ".*_knee_pitch_joint": 12.356,
                "waist.*": 14.738,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_pitch_joint": 200.0,
                "waist.*": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_pitch_joint": 5.0,
                "waist.*": 5.0,
            },
            armature={
                ".*_hip_yaw_joint": 0.165,
                ".*_hip_roll_joint": 0.165,
                ".*_hip_pitch_joint": 0.121,
                ".*_knee_pitch_joint": 0.121,
                "waist.*": 0.165,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 30.0, 
                ".*_ankle_roll_joint": 30.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 16.747, 
                ".*_ankle_roll_joint": 16.747,
            },
            stiffness=20.0,
            damping=2.0,
            armature=0.031
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch.*",
                ".*_shoulder_roll.*",
            ],
            effort_limit_sim=54,
            velocity_limit_sim = 14.738,
            stiffness=100.0,
            damping=5.0,
            armature={
                ".*_shoulder_pitch.*": 0.165,
                ".*_shoulder_roll.*": 0.031,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw.*",
                ".*_elbow_.*",
            ],
            effort_limit_sim = 30,
            velocity_limit_sim = 16.747,
            stiffness=50.0,
            damping=2.0,
            armature={
                ".*_shoulder_yaw.*": 0.031,
                ".*_elbow_.*": 0.031,
            },
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_.*",
            ],
            effort_limit_sim = 30,
            velocity_limit_sim = 16.747,
            stiffness=50,
            damping=2.0,
            armature=0.031,
        ),
    },
)

FOURIERN1_HUMANOID_DELAY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/fouriern1_humanoid.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_hip_roll_joint": 0.0, 
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_pitch_joint": -0.2618,
            ".*_knee_pitch_joint": 0.5236,
            ".*_ankle_pitch_joint": -0.2618,
            ".*_ankle_roll_joint": 0.0,
            "waist.*": 0.0,
            ".*_shoulder_pitch.*": 0.0,
            ".*_shoulder_roll.*": 0.0,
            ".*_shoulder_yaw.*": 0.0,
            ".*_elbow_.*": 0.0,
            ".*_wrist_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_pitch_joint",
                "waist.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 54.0,
                ".*_hip_roll_joint": 54.0,
                ".*_hip_pitch_joint": 95,
                ".*_knee_pitch_joint": 95,
                "waist.*": 54,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 14.738,
                ".*_hip_roll_joint": 14.738,
                ".*_hip_pitch_joint": 12.356,
                ".*_knee_pitch_joint": 12.356,
                "waist.*": 14.738,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_pitch_joint": 200.0,
                "waist.*": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_pitch_joint": 5.0,
                "waist.*": 5.0,
            },
            armature={
                ".*_hip_yaw_joint": 0.165,
                ".*_hip_roll_joint": 0.165,
                ".*_hip_pitch_joint": 0.121,
                ".*_knee_pitch_joint": 0.121,
                "waist.*": 0.165,
            },
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 30.0, 
                ".*_ankle_roll_joint": 30.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 16.747, 
                ".*_ankle_roll_joint": 16.747,
            },
            stiffness=20.0,
            damping=2.0,
            armature=0.031,
            min_delay=0,
            max_delay=2,
        ),
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch.*",
                ".*_shoulder_roll.*",
            ],
            effort_limit_sim=54,
            velocity_limit_sim = 14.738,
            stiffness=100.0,
            damping=5.0,
            armature={
                ".*_shoulder_pitch.*": 0.165,
                ".*_shoulder_roll.*": 0.031,
            },
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw.*",
                ".*_elbow_.*",
            ],
            effort_limit_sim = 30,
            velocity_limit_sim = 16.747,
            stiffness=50.0,
            damping=2.0,
            armature={
                ".*_shoulder_yaw.*": 0.031,
                ".*_elbow_.*": 0.031,
            },
            min_delay=0,
            max_delay=2,
        ),
        "wrist": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_wrist_.*",
            ],
            effort_limit_sim = 30,
            velocity_limit_sim = 16.747,
            stiffness=50,
            damping=2.0,
            armature=0.031,
            min_delay=0,
            max_delay=2,
        ),
    },
)