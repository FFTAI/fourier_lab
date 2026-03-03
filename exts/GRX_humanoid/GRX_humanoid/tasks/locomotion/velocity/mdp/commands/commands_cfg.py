import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .gait_command import GaitCommand  # Import the GaitCommand class
from .height_attitude_command import HACommand, HRCommand
from .behavior_command import BehaviorCommand
from .joint_pos_command import JointPosCommand
from .velocity_command import CustomVelocityCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from .velocity_command import FeedbackUniformVelocityCommand


@configclass
class UniformGaitCommandCfg(CommandTermCfg):
    """Configuration for the gait command generator."""

    class_type: type = GaitCommand  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait parameters."""

        frequencies: tuple[float, float] | list[float] = MISSING
        """Range for gait frequencies [Hz]. Can be a tuple or a list of discrete values."""
        frequencies_probs: list[float] = MISSING
        """Probabilities for each frequency value if frequencies is a list."""
        offsets: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        durations: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""

    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""

@configclass
class UniformHACommandCfg(CommandTermCfg):
    """Configuration for the H&A command generator."""

    class_type: type = HACommand  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the H&A parameters."""

        height: tuple[float, float]= MISSING
        """expected height"""
        pitch_angle: tuple[float, float] = MISSING
        """expected pitch angle """
        yaw_angle: tuple[float, float] = MISSING
        """expected yaw angle when stand state"""

    ranges: Ranges = MISSING
    """Distribution ranges for the height & attitude parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the height and attitude (in seconds)."""

@configclass
class UniformHRCommandCfg(CommandTermCfg):
    """Configuration for the Height and torso rotation command generator."""

    class_type: type = HRCommand  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the H&A parameters."""

        height: tuple[float, float]= MISSING
        """expected height"""
        torso_yaw: tuple[float, float] = MISSING
        """expected torso yaw angle """
        torso_roll: tuple[float, float] = MISSING
        """expected torso roll angle """
        torso_pitch: tuple[float, float] = MISSING
        """expected torso pitch angle """

    ranges: Ranges = MISSING
    """Distribution ranges for the height & attitude parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the height and attitude (in seconds)."""

@configclass
class UniformJointPosCommandCfg(CommandTermCfg):
    """Configuration for the joint pos command generator."""

    class_type: type = JointPosCommand  # Specify the class type for dynamic instantiation
    
    asset_cfg: SceneEntityCfg = MISSING

    ranges_scaled: tuple[float, float]= MISSING
    """Distribution ranges for the joint pos parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the joint pos (in seconds)."""

@configclass
class UniformBehaviorCommandCfg(CommandTermCfg):
    """Configuration for the H&A command generator."""

    class_type: type = BehaviorCommand  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the H&A parameters."""

        height: tuple[float, float]= MISSING
        """expected height"""
        pitch_angle: list[float] = MISSING
        """expected pitch angle """
        waist_angle: list[float] = MISSING
        """expected waist angle """
        swing_height: list[float] = MISSING
        """expected swing foot height """

    ranges: Ranges = MISSING
    """Distribution ranges for the height & attitude parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the height and attitude (in seconds)."""

@configclass
class CustomVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = CustomVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """

        zero_prob: tuple[float, float, float] = (0, 0, 0)

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

@configclass
class FeedbackUniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = FeedbackUniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_max_velocity_envs: float = 0.0
    """The sampled probability of environments that should be moving at maximum velocity. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s).
        Used for set the maximum angular velocity command for feedback of the robot's y-position.
        """

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    goal_ang_vel_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/angular_velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    vis_goal_ang_vel: bool = False
    """Whether to visualize the goal angular velocity. Defaults to False."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    goal_ang_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.2, 0.2)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)