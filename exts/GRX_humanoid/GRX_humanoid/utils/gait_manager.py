import torch
import random
import collections


class GaitParam:
    def __init__(self, freq: float = 2, sw_rate: float = 0.5, ofs=None, sym: str = "NONE"):
        if ofs is None:
            ofs = [0, 0, 0, 0]
        self.frequency = freq
        self.offset = ofs
        self.symmetry = sym
        self.swing_ratio = sw_rate


quadruped_gait_library = {
    "pronk": GaitParam(freq=2, sw_rate=0.7, ofs=[0, 0, 0, 0], sym="NONE"),
    "pace": GaitParam(freq=2, sw_rate=0.5, ofs=[0, 0.5, 0, 0.5], sym="SAGITTAL"),
    "trot": GaitParam(freq=2, sw_rate=0.5, ofs=[0, 0.5, 0.5, 0], sym="CORONAL"),
    "walk": GaitParam(freq=1.2, sw_rate=0.25, ofs=[0, 0.25, 0.75, 0.5], sym="DOUBLE_SPIRAL"),
    "gallop": GaitParam(freq=2.5, sw_rate=0.75, ofs=[0, 0.2, 0.6, 0.8], sym="DOUBLE_SPIRAL"),
    "bound": GaitParam(freq=2, sw_rate=0.5, ofs=[0, 0, 0.5, 0.5], sym="SAGITTAL"),
    "tripod": GaitParam(freq=2, sw_rate=0.33, ofs=[0, 0.333, 0.667, 0], sym="DOUBLE_SPIRAL")
}

humanoid_gait_library = {
    "leap": GaitParam(freq=1.8, sw_rate=0.25, ofs=[0, 0], sym="NONE"),
    "walk": GaitParam(freq=1.5, sw_rate=0.5, ofs=[0, 0.5], sym="CORONAL"),
    "stance_walk": GaitParam(freq=1.2, sw_rate=0.45, ofs=[0, 0.5], sym="CORONAL"),
    "run": GaitParam(freq=2, sw_rate=0.65, ofs=[0, 0.5], sym="CORONAL"),
    "stand": GaitParam(freq=0., sw_rate=1e-8, ofs=[0.5, 0.5], sym="NONE")
}

def piecewise_2var_torch(x: torch.Tensor, r: torch.Tensor, condlist, funclist, default=None) -> torch.Tensor:
    """
    A PyTorch version of piecewise_2var: evaluates `funclist[i](x, r)` wherever `condlist[i]` is True.
    Supports a final default value if no conditions match.
    """
    out = torch.zeros_like(x)
    matched = torch.zeros_like(x, dtype=torch.bool)

    for cond, func in zip(condlist, funclist):
        mask = cond(x, r)
        out = torch.where(mask, func(x, r), out)
        matched |= mask

    if default is not None:
        out = torch.where(~matched, default if isinstance(default, torch.Tensor) else torch.full_like(x, default), out)

    return out


class GaitManager:
    """
    Gait signal generator. Also is in charge of computing the gait reward coefficients.
    """
    PhaseTypes = ['RAMP', 'BALANCED_SINE', 'ADAPTIVE_SINE', 'STEP']
    Symmetries = ['NONE', 'SAGITTAL', 'CORONAL', 'Z_AXIAL', 'CENTRAL', 'SPIRAL', 'DOUBLE_SPIRAL']
    # * Definition of gait symmetry
    # * NONE: Cannot do any mirroring or spinning
    # * SAGITTAL: Can do Left/Right mirroring
    # * CORONAL: Can Front/Back mirroring
    # * Z_AXIAL: Can do both SAGITTAL and CORONAL flip
    # * CENTRAL: Can spin 180deg
    # * SPIRAL: Can spin any x*90deg clock
    # * DOUBLE_SPIRAL:  Can do both SPIRAL and SAGITTAL
    num_legs: int = 2
    num_robots: int = 1
    symmetry: str = "NONE"
    # swingRatio: float = 0
    contactTolerance: float = 0
    signalType: str = ""
    def __init__(self, cfg, num_robots: int = 1, num_legs: int = 2, dt: float = 1e-2):
        self.cfg = cfg
        self.num_robots = num_robots
        self.num_legs = num_legs
        self.time_step = dt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.phaseVal = torch.zeros((num_robots, 1), dtype=torch.float32, device=self.device)
        self.transitionCountDown = torch.zeros((num_robots, 1), dtype=torch.int, device=self.device)
        self.isStance = torch.ones((num_robots, num_legs), dtype=torch.bool, device=self.device)
        self.onStartingUp = torch.zeros((num_robots, num_legs), dtype=torch.bool, device=self.device)
        self.onStopping = torch.zeros((num_robots, num_legs), dtype=torch.bool, device=self.device)
        self.footPhases = torch.zeros((num_robots, num_legs), dtype=torch.float32, device=self.device)
        self.offset = torch.zeros((num_robots, num_legs), dtype=torch.float32, device=self.device)
        self.swingRatio = torch.full((num_robots, num_legs), 0.5, dtype=torch.float32, device=self.device)
        self.frequency = torch.ones((num_robots, num_legs), dtype=torch.float32, device=self.device)

        self.frequency_range = None
        self.swing_ratio_range = None

        self.load_config()

        self.force_reward_weight = torch.zeros((num_robots, num_legs), dtype=torch.float32, device=self.device)
        self.speed_reward_weight = torch.zeros((num_robots, num_legs), dtype=torch.float32, device=self.device)

    def load_config(self):
        if self.num_legs == 2:
            gait = humanoid_gait_library[self.cfg.name]
        else:
            gait = quadruped_gait_library[self.cfg.name]

        self.symmetry = gait.symmetry
        self.offset[:] = torch.tensor(gait.offset, device=self.device).repeat(self.num_robots, 1)
        self.contactTolerance = self.cfg.contactTolerance
        self.signalType = self.cfg.state_type.upper() if self.cfg.state_type.upper() in ['RAMP', 'BALANCED_SINE', 'ADAPTIVE_SINE', 'STEP'] else 'BALANCED_SINE'

        if self.cfg.frequency == "default" or self.cfg.frequency is None:
            self.frequency[:] = gait.frequency
        elif isinstance(self.cfg.frequency, list):
            self.frequency_range = self.cfg.frequency
            self.generate_random_frequency()
        else:
            self.frequency[:] = self.cfg.frequency

        if self.cfg.swingRatio == "default" or self.cfg.swingRatio is None:
            self.swingRatio[:] = gait.swing_ratio
        elif isinstance(self.cfg.swingRatio, list):
            self.swing_ratio_range = self.cfg.swingRatio
            self.generate_random_swing_ratio()
        else:
            self.swingRatio[:] = self.cfg.swingRatio

    def reset(self, env_ids):
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.cpu().numpy()

        self.phaseVal[env_ids] = 0.
        self.transitionCountDown[env_ids] = 0
        self.isStance[env_ids] = True
        self.onStartingUp[env_ids] = False
        self.onStopping[env_ids] = False
        self.footPhases[env_ids] = 0.

        if self.frequency_range is not None:
            self.generate_random_frequency(env_ids)
        if self.swing_ratio_range is not None:
            self.generate_random_swing_ratio(env_ids)

        ids_tensor = torch.tensor(env_ids, device=self.device)

        if self.symmetry in ["SAGITTAL", "Z_AXIAL", "DOUBLE_SPIRAL"]:
            mask = torch.rand(len(env_ids), device=self.device) < 0.5
            flip_ids = ids_tensor[mask]
            if flip_ids.numel() > 0:
                orig = self.offset[flip_ids]
                new = torch.empty_like(orig)
                new[:, ::2] = orig[:, 1::2]
                new[:, 1::2] = orig[:, ::2]
                self.offset[flip_ids] = new

        if self.symmetry in ["CORONAL", "Z_AXIAL"]:
            mask = torch.rand(len(env_ids), device=self.device) < 0.5
            flip_ids = ids_tensor[mask]
            if flip_ids.numel() > 0:
                orig = self.offset[flip_ids]
                half = self.num_legs // 2
                new = torch.cat([orig[:, half:], orig[:, :half]], dim=1)
                self.offset[flip_ids] = new

        if self.symmetry == "CENTRAL" and self.num_legs == 2:
            mask = torch.rand(len(env_ids), device=self.device) < 0.5
            flip_ids = ids_tensor[mask]
            if flip_ids.numel() > 0:
                orig = self.offset[flip_ids]
                new = torch.empty_like(orig)
                new[:, 0] = orig[:, 1]
                new[:, 1] = orig[:, 0]
                self.offset[flip_ids] = new

        if self.symmetry == "SPIRAL" and self.num_legs == 4:
            mask = torch.rand(len(env_ids), device=self.device) < 0.5
            flip_ids = ids_tensor[mask]
            if flip_ids.numel() > 0:
                orig = self.offset[flip_ids]
                new = orig[:, [2, 3, 0, 1]]
                self.offset[flip_ids] = new

        self.footPhases[env_ids] = self.offset[env_ids]

    def run(self, cmd: torch.Tensor = None):
        if cmd is not None:
            vel_mag = torch.norm(cmd[:, :2], dim=1)
            lin_flag = vel_mag >= 0.1
            ang_flag = torch.abs(cmd[:, 2]) >= 0.1
            move_flag = (lin_flag | ang_flag).unsqueeze(1).repeat(1, self.num_legs)

            self.onStartingUp[~move_flag & self.onStartingUp] = False
            self.onStopping[move_flag & self.onStopping] = False
            self.onStartingUp[move_flag & self.isStance] = True
            self.onStopping[~(move_flag | self.isStance)] = True
        else:
            self.onStartingUp[self.isStance] = True
            self.onStopping[:] = False

        prev = (self.offset + self.phaseVal) % 1.0
        self.phaseVal = (self.phaseVal + self.time_step * self.frequency) % 1.0
        self.footPhases = (self.offset + self.phaseVal) % 1.0

        stance_center = 0.5 + self.swingRatio / 2
        on_center = (self.footPhases >= stance_center) & (prev < stance_center)

        self.isStance &= ~(self.onStartingUp & on_center)
        self.onStartingUp &= ~on_center

        self.isStance |= self.onStopping & on_center
        self.onStopping &= ~on_center

        self.footPhases[self.isStance] = stance_center[self.isStance]

    # def get_frc_penalty_coeff(self):
    #     x = self.footPhases
    #     r = self.swingRatio
    #     tol = self.contactTolerance

    #     conds = [
    #         x <= tol,
    #         x >= 1 - tol,
    #         torch.abs(x - r) <= tol,
    #         (x <= 1 - tol) & ((r + tol) <= x)
    #     ]

    #     funcs = [
    #         lambda x, r: 0.5 + x / (2 * tol),
    #         lambda x, r: 0.5 + (x - 1) / (2 * tol),
    #         lambda x, r: 0.5 - (x - r) / (2 * tol)
    #     ]

    #     out = torch.zeros_like(x)
    #     for cond, func in zip(conds[:3], funcs):
    #         out = torch.where(cond, func(x, r), out)

    #     out = torch.where(conds[3], torch.tensor(0.0, device=self.device), out)
    #     out = torch.where(~(conds[0] | conds[1] | conds[2] | conds[3]), torch.tensor(1.0, device=self.device), out)

    #     self.force_reward_weight = out
    #     return out
    def get_frc_penalty_coeff(self):
        x = self.footPhases
        r = self.swingRatio
        tol = self.contactTolerance

        condlist = [
            lambda x, r: x <= tol,
            lambda x, r: x >= 1 - tol,
            lambda x, r: torch.abs(x - r) <= tol,
            lambda x, r: (x <= 1 - tol) & ((r + tol) <= x)
        ]

        funclist = [
            lambda x, r: 0.5 + x / (2 * tol),
            lambda x, r: 0.5 + (x - 1) / (2 * tol),
            lambda x, r: 0.5 - (x - r) / (2 * tol),
            lambda x, r: torch.zeros_like(x)  # instead of 0.0
        ]


        self.force_reward_weight = piecewise_2var_torch(x, r, condlist, funclist, default=1.0)
        return self.force_reward_weight


    def get_vel_penalty_coeff(self):
        self.speed_reward_weight = 1.0 - self.force_reward_weight
        return self.speed_reward_weight

    def generate_random_frequency(self, env_ids=None):
        N = len(env_ids) if env_ids is not None else self.num_robots
        freq = torch.rand((N, 1), device=self.device) * (self.frequency_range[1] - self.frequency_range[0]) + self.frequency_range[0]
        freq_full = freq.repeat(1, self.num_legs)
        if env_ids is None:
            self.frequency = freq_full
        else:
            self.frequency[env_ids] = freq_full

    def generate_random_swing_ratio(self, env_ids=None):
        N = len(env_ids) if env_ids is not None else self.num_robots
        ratio = torch.rand((N, 1), device=self.device) * (self.swing_ratio_range[1] - self.swing_ratio_range[0]) + self.swing_ratio_range[0]
        ratio_full = ratio.repeat(1, self.num_legs)
        if env_ids is None:
            self.swingRatio = ratio_full
        else:
            self.swingRatio[env_ids] = ratio_full

    def get_phase_states(self):
        if self.signalType == 'RAMP':
            return self.footPhases

        elif self.signalType.endswith("SINE"):
            if self.signalType.startswith("ADAPTIVE"):
                condlist = [lambda x, r: x >= r]
                funclist = [
                    lambda x, r: 1 + (x - 1) / (2 * (1 - r)),
                    lambda x, r: x / (2 * r)
                ]
                internal_phase = piecewise_2var_torch(self.footPhases, self.swingRatio, condlist, funclist, default=0.0) * 2 * torch.pi
            else:
                internal_phase = self.footPhases * 2 * torch.pi

            sin = torch.sin(internal_phase)
            cos = torch.cos(internal_phase)
            return torch.stack((sin, cos), dim=-1).reshape(self.num_robots, -1)

        elif self.signalType == 'STEP':
            state = torch.where(self.footPhases < self.swingRatio, torch.tensor(1.0, device=self.device), torch.tensor(-1.0, device=self.device))
            return state

        else:
            print("[GaitManager.get_phase_states] Unknown gait signal type.")
            return None
        
    