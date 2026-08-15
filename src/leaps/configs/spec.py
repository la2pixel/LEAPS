"""The RunSpec dataclass.
"""

from dataclasses import dataclass
from enum import Enum

# body
BODIES = {"h0918": 18, "h1622": 22, "h2190": 90}


class Category(str, Enum):
    MPO = "mpo"
    DEP_MPO = "dep-mpo"
    EMG_LAP = "emg_lap"
    NO_LAP = "no_lap"
    UNTRAINED_LAP = "untrained_lap"
    # I have not included other side experiments like (self_synergy, coactivation_reward, corrnoise, ...)


LAP_CATEGORIES = frozenset({Category.EMG_LAP, Category.NO_LAP, Category.UNTRAINED_LAP})


class ExperimentGroup(str, Enum):
    # 2026-08-15: baselines_DEPRL reorg -- everything before this date lives
    # under early_tests/ (both here and in the media results tree), new runs
    # default to final_experiments/.
    EARLY_TESTS = "early_tests"
    FINAL_EXPERIMENTS = "final_experiments"

#for thesis, lets for now report full and onlyvel.
class RewardVariant(str, Enum):
    FULL = "full"
    ONLYVELREW = "onlyvelrew"
    GAUSSIANVEL = "gaussianvel"

#Decoder variants changed after feedback from 14.08 meeting
class DecoderSource(str, Enum):
    POOLED = "pooled"
    AB06 = "AB06"
    AB20 = "AB20"


@dataclass(frozen=True)
class RunSpec:
    category: Category
    body: str
    experiment_group: ExperimentGroup = ExperimentGroup.FINAL_EXPERIMENTS
    reward_variant: RewardVariant = RewardVariant.FULL
    seed: int = 0

    clip: bool = False  # -> env_args.clip_actions (True clips to [0,0.5], False to [0,1.0]), we need to test the 3rd option of no clip at all.
    net_size: int = 256  # -> mpo_args.hidden_size iff != 256
    #try with 512x512 

    # LAP-only knobs, ignored for category in (MPO, DEP_MPO).
    dim_latent: int = 6  # "k"
    mapped_residual_weight: float = 0.5  # "w" -- NOT residual_weight, which stays 1.0 (what????)
    residual_weight: float = 1.0
    mirror_left: bool = True
    mirror_mode: str = "static"  # "static" (existing) or "phase" (heel-strike-triggered, 2026-08-15)
    decoder_source: DecoderSource = DecoderSource.POOLED

    # DEP_MPO-only.
    dep_kappa: int = 1000  # there's no kappa i can take for the sconewalk task, the closest is to use k=1986 from the run task but need to confirm

    resume: bool = True

    synergy_root: str = "/home/nadinebadie/lalitha/LEAPS/results/synergy"
    baselines_root: str = "/home/nadinebadie/lalitha/LEAPS/baselines_DEPRL"

    def __post_init__(self) -> None:
        if self.body not in BODIES:
            raise ValueError(f"unknown body {self.body!r}, expected one of {sorted(BODIES)}")
        if self.dim_latent <= 0:
            raise ValueError(f"dim_latent must be positive, got {self.dim_latent}")
        if self.net_size <= 0:
            raise ValueError(f"net_size must be positive, got {self.net_size}")
        if self.mirror_mode not in ("static", "phase"):
            raise ValueError(f"mirror_mode must be 'static' or 'phase', got {self.mirror_mode!r}")

    @property
    def num_acts(self) -> int:
        return BODIES[self.body]

    @property
    def is_lap(self) -> bool:
        return self.category in LAP_CATEGORIES

    @property
    def is_dep(self) -> bool:
        return self.category == Category.DEP_MPO
