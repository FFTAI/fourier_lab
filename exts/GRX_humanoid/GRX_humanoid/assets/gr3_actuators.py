from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.actuators import \
    DelayedPDActuator, DelayedPDActuatorCfg, ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

### 用不了啊ImplicitActuator的计算在PhysX里完成，只能调用PhysX的接口

class GR3ActuatorCfg(ImplicitActuatorCfg):
    """GR3 Humanoid Actuator Model.
    Add motor performance envelope parameters for GR3 humanoid robot.
    """
    velocity_dependent_resistance: dict[str, float] | float | None = None
    speed_effort_gradient: dict[str, float] | float | None = None
    max_actuator_velocity: dict[str, float] | float | None = None

    