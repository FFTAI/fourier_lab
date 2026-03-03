import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.assets.articulation import ArticulationCfg

from GRX_humanoid.assets import ISAAC_ASSET_DIR

PPV222_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/ppv222_noArmCollision.usd",
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
        pos=(0.0, 0.0, 0.95),
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
            ".*_wrist_.*": 0.0,
            "head_.*": 0.0,
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
                ".*_hip_yaw_joint": 140,
                ".*_hip_roll_joint": 140,
                ".*_hip_pitch_joint": 221,
                ".*_knee_pitch_joint": 221,
                "waist_yaw.*": 140,
                "waist_roll.*": 108.6,
                "waist_pitch.*": 108.6,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 13,
                ".*_hip_roll_joint": 13,
                ".*_hip_pitch_joint": 16,
                ".*_knee_pitch_joint": 16,
                "waist_yaw.*": 13,
                "waist_roll.*": 14,
                "waist_pitch.*": 14,
            },
            stiffness={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 300.0,
                ".*_knee_pitch_joint": 300.0,
                "waist_yaw_joint": 200.0,
                "waist_roll_joint": 300.0,
                "waist_pitch_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 6.0,
                ".*_knee_pitch_joint": 6.0,
                "waist_yaw_joint": 20.0,
                "waist_roll_joint": 30.0,
                "waist_pitch_joint": 20.0,
            },
            armature={
                ".*_hip_pitch_joint": 0.336,
                ".*_hip_roll_joint": 0.18,
                ".*_hip_yaw_joint": 0.18,
                ".*_knee_pitch_joint": 0.336,
                "waist_yaw.*": 0.18,
                "waist_roll.*": 0.495,
                "waist_pitch.*": 0.330,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=59.4, 
            velocity_limit_sim=16.76, 
            stiffness={
                ".*_ankle_pitch_joint": 40,
                ".*_ankle_roll_joint": 5,
            },
            damping={
                ".*_ankle_pitch_joint": 2,
                ".*_ankle_roll_joint": 1,
            },
            armature={
                ".*_ankle_pitch_joint": 0.078, 
                ".*_ankle_roll_joint":0.012
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw.*", "head_pitch.*"],
            effort_limit_sim={
                "head_yaw.*": 17.4,
                "head_pitch.*": 17.4,
            },
            velocity_limit_sim={
                "head_yaw.*": 9.21,
                "head_pitch.*": 9.21,
            },
            stiffness={
                "head_yaw.*": 50.0,
                "head_pitch.*": 50.0,
            },
            damping={
                "head_yaw.*": 5.0,
                "head_pitch.*": 5.0,
            },
            armature={
                "head_yaw.*": 0.111,
                "head_pitch.*": 0.111,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch.*",
                ".*_shoulder_roll.*",
                ".*_shoulder_yaw.*",
                ".*_elbow_.*",
                ".*wrist_yaw.*",
                ".*wrist_pitch.*",
                ".*wrist_roll.*",
            ],
            effort_limit_sim ={
                ".*_shoulder_pitch.*": 74.4,
                ".*_shoulder_roll.*": 74.4,
                ".*_shoulder_yaw.*": 42.9,
                ".*_elbow_.*": 42.9,
                ".*wrist_yaw.*": 42.9,
                ".*wrist_pitch.*": 17.4,
                ".*wrist_roll.*": 17.4,
            },
            velocity_limit_sim = {
                ".*_shoulder_pitch.*": 7.75,
                ".*_shoulder_roll.*": 7.75,
                ".*_shoulder_yaw.*": 6.28,
                ".*_elbow_.*": 6.28,
                ".*wrist_yaw.*": 6.28,
                ".*wrist_pitch.*": 9.21,
                ".*wrist_roll.*": 9.21,
            },
            stiffness = {
                ".*_shoulder_pitch.*": 200.0,
                ".*_shoulder_roll.*": 200.0,
                ".*_shoulder_yaw.*": 200.0,
                ".*_elbow_.*": 100.0,
                ".*wrist_yaw.*": 50.0,
                ".*wrist_pitch.*": 50.0,
                ".*wrist_roll.*": 50.0,
            },
            damping = {
                ".*_shoulder_pitch.*": 20.0,
                ".*_shoulder_roll.*": 20.0,
                ".*_shoulder_yaw.*": 20.0,
                ".*_elbow_.*": 10.0,
                ".*wrist_yaw.*": 5.0,
                ".*wrist_pitch.*": 5.0,
                ".*wrist_roll.*": 5.0,
            },
            armature = {
                ".*_shoulder_pitch.*": 0.606,
                ".*_shoulder_roll.*": 0.606,
                ".*_shoulder_yaw.*": 0.222,
                ".*_elbow_.*": 0.222,
                ".*wrist_yaw.*": 0.222,
                ".*wrist_pitch.*": 0.111,
                ".*wrist_roll.*": 0.111,
            },
        ),
    },
)

PPV222_HUMANOID_CFG_LOWER = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/ppv222_humanoid_lower.usd",
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
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_roll_joint": 0.0, 
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_pitch_joint": -0.2618,
            ".*knee_pitch_joint": 0.5236,
            ".*_ankle_pitch_joint": -0.2618,
            ".*_ankle_roll_joint": 0.0,
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
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 140,
                ".*_hip_roll_joint": 140,
                ".*_hip_pitch_joint": 221,
                ".*_knee_pitch_joint": 221,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 13,
                ".*_hip_roll_joint": 13,
                ".*_hip_pitch_joint": 16,
                ".*_knee_pitch_joint": 16,
            },
            stiffness={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_pitch_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 12.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 20.0,
                ".*_knee_pitch_joint": 20.0,
            },
            armature={
                ".*_hip_yaw_joint": 0.18,
                ".*_hip_roll_joint": 0.18,
                ".*_hip_pitch_joint": 0.336,
                ".*_knee_pitch_joint": 0.336,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim = {
                ".*_ankle_pitch_joint": 40,
                ".*_ankle_roll_joint": 20,
            },
            velocity_limit_sim=16.755, 
            stiffness={
                ".*_ankle_pitch_joint": 115.4,
                ".*_ankle_roll_joint": 15,
            },
            damping={
                ".*_ankle_pitch_joint": 11.54,
                ".*_ankle_roll_joint": 1.5,
            },
            armature={
                ".*_ankle_pitch_joint": 0.078, 
                ".*_ankle_roll_joint":0.012
            },
        ),
    },
)