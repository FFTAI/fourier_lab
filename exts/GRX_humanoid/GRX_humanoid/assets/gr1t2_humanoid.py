import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from GRX_humanoid.assets import ISAAC_ASSET_DIR

GR1T2_HUMANOID_HIP_ROLL_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*hip_roll_joint"],
    effort_limit_sim=48.0,
    velocity_limit_sim=12.15,
    stiffness={".*": 251.625},
    damping={".*": 14.72},
)

GR1T2_HUMANOID_HIP_YAW_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*hip_yaw_joint"],
    effort_limit_sim=66.0,
    velocity_limit_sim=16.76,
    stiffness={".*": 362.5214},
    damping={".*": 10.0833},
)

GR1T2_HUMANOID_HIP_PITCH_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*hip_pitch_joint"],
    effort_limit_sim=225.0,
    velocity_limit_sim=37.38,
    stiffness={".*": 200.0},
    damping={".*": 11.0},
)

GR1T2_HUMANOID_KNEE_PITCH_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*knee_pitch_joint"],
    effort_limit_sim=225.0,
    velocity_limit_sim=37.38,
    stiffness={".*": 200.0},
    damping={".*": 11},
)

GR1T2_HUMANOID_ANKLE_PITCH_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*ankle_pitch_joint"],
    effort_limit_sim=15.0,
    velocity_limit_sim=20.32,
    stiffness={".*": 10.9805},
    damping={".*": 0.5991},
)

GR1T2_HUMANOID_ANKLE_ROLL_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*ankle_roll_joint"],
    effort_limit_sim=30.0,
    velocity_limit_sim=20.32,
    stiffness={".*": 10.9805},
    damping={".*": 0.5991},
)


GR1T2_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/gr1t2_humanoid.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.93),
        joint_pos={
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_hip_pitch_joint': -0.2618,
            'left_knee_pitch_joint': 0.5236,
            'left_ankle_pitch_joint': -0.2618,
            'left_ankle_roll_joint': 0.0,

            # right leg
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_hip_pitch_joint': -0.2618,
            'right_knee_pitch_joint': 0.5236,
            'right_ankle_pitch_joint': -0.2618,
            'right_ankle_roll_joint': 0.0,
        },
    ),
    actuators={"hip_roll_joint": GR1T2_HUMANOID_HIP_ROLL_ACTUATOR_CFG, "hip_yaw_joint": GR1T2_HUMANOID_HIP_YAW_ACTUATOR_CFG,
               "hip_pitch_joint": GR1T2_HUMANOID_HIP_PITCH_ACTUATOR_CFG, "knee_pitch_joint": GR1T2_HUMANOID_KNEE_PITCH_ACTUATOR_CFG,
               "ankle_pitch_joint": GR1T2_HUMANOID_ANKLE_PITCH_ACTUATOR_CFG, "ankle_roll_joint": GR1T2_HUMANOID_ANKLE_ROLL_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
