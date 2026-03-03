import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from GRX_humanoid.assets import ISAAC_ASSET_DIR

PPV233_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/ppv233_noArmCollision.usd",
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
        pos=(0.0, 0.0, 0.92),
        joint_pos={
            ".*_hip_roll_joint": 0.0, 
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_pitch_joint": -0.2618,
            ".*knee_pitch_joint": 0.5236,
            ".*_ankle_pitch_joint": -0.2618,
            ".*_ankle_roll_joint": 0.0,
            ".*waist.*": 0.0,
            ".*head.*": 0.0,
            ".*_shoulder_pitch.*": 0.1,
            ".*left_shoulder_roll.*": 0.15,
            ".*right_shoulder_roll.*": -0.15,
            ".*left_shoulder_yaw.*": -0.25,
            ".*right_shoulder_yaw.*": 0.25,
            ".*_elbow_.*": -0.5,
            ".*_wrist_.*": 0.0,
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
                ".*_hip_pitch_joint": 263,
                ".*_knee_pitch_joint": 263,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 12.88,
                ".*_hip_roll_joint": 12.88,
                ".*_hip_pitch_joint": 12.77,
                ".*_knee_pitch_joint": 12.77,
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
                ".*_ankle_pitch_joint": 60,
                ".*_ankle_roll_joint": 30,
            },
            velocity_limit_sim=16.65, 
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
                ".*_ankle_roll_joint":0.0117
            },
        ),
        "upper_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*waist.*",
                ".*head.*",
                ".*shoulder.*",
                ".*elbow.*",
                ".*wrist.*"
            ],
            effort_limit_sim = {
                "waist_yaw_joint": 140,
                "waist_roll_joint": 143.4,
                "waist_pitch_joint": 143.4,
                "head_yaw_joint": 23.25,
                "head_pitch_joint": 23.25,
                ".*shoulder_pitch_joint": 80, 
                ".*shoulder_roll_joint": 80, 
                ".*shoulder_yaw_joint": 41, 
                ".*elbow_pitch_joint": 41, 
                ".*wrist_yaw_joint": 41,
                ".*wrist_pitch_joint": 23.25,
                ".*wrist_roll_joint": 23.25,
            },
            velocity_limit_sim = {
                "waist_yaw_joint": 12.88,
                "waist_roll_joint": 13.09,
                "waist_pitch_joint": 13.09,
                "head_yaw_joint": 9.21,
                "head_pitch_joint": 9.21,
                ".*shoulder_pitch_joint": 7.75, 
                ".*shoulder_roll_joint": 7.75, 
                ".*shoulder_yaw_joint": 6.28, 
                ".*elbow_pitch_joint": 6.28, 
                ".*wrist_yaw_joint": 6.28,
                ".*wrist_pitch_joint": 9.21,
                ".*wrist_roll_joint": 9.21,
            },
            stiffness = {
                "waist_yaw_joint": 200.0,
                "waist_roll_joint": 300.0,
                "waist_pitch_joint": 200.0,
                "head_yaw_joint": 50.0,
                "head_pitch_joint": 50.0,
                ".*shoulder_pitch_joint": 200.0, 
                ".*shoulder_roll_joint": 200.0, 
                ".*shoulder_yaw_joint": 200.0, 
                ".*elbow_pitch_joint": 100.0, 
                ".*wrist_yaw_joint": 100.0,
                ".*wrist_pitch_joint": 50.0,
                ".*wrist_roll_joint": 50.0,
            },
            damping = {
                "waist_yaw_joint": 20.0,
                "waist_roll_joint": 30.0,
                "waist_pitch_joint": 20.0,
                "head_yaw_joint": 5.0,
                "head_pitch_joint": 5.0,
                ".*shoulder_pitch_joint": 20.0, 
                ".*shoulder_roll_joint": 20.0, 
                ".*shoulder_yaw_joint": 20.0, 
                ".*elbow_pitch_joint": 10.0, 
                ".*wrist_yaw_joint": 10.0,
                ".*wrist_pitch_joint": 5.0,
                ".*wrist_roll_joint": 5.0,
            },
            armature = {
                "waist_yaw_joint": 0.18,
                "waist_roll_joint": 0.291,
                "waist_pitch_joint": 0.194,
                "head_yaw_joint": 0.111,
                "head_pitch_joint": 0.111,
                ".*shoulder_pitch_joint": 0.606, 
                ".*shoulder_roll_joint": 0.606, 
                ".*shoulder_yaw_joint": 0.222, 
                ".*elbow_pitch_joint": 0.222, 
                ".*wrist_yaw_joint": 0.222,
                ".*wrist_pitch_joint": 0.111,
                ".*wrist_roll_joint": 0.111,
            },
        ),
    },
)

PPV233_HUMANOID_CFG_LOWER = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/ppv233_humanoid_lower.usd",
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
                ".*_hip_yaw_joint": 140.4,
                ".*_hip_roll_joint": 140.4,
                ".*_hip_pitch_joint": 221.4,
                ".*_knee_pitch_joint": 221.4,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 12.985,
                ".*_hip_roll_joint": 12.985,
                ".*_hip_pitch_joint": 16.44,
                ".*_knee_pitch_joint": 16.44,
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
            velocity_limit_sim=16.65, 
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
                ".*_ankle_roll_joint":0.0117
            },
        ),
    },
)