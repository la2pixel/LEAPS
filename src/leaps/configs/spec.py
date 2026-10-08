"""The RunSpec dataclass."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from leaps.paths import PRIORS_DIR, RESULTS_DIR, RUNS_DIR

# body
BODIES = {"h0918": 18, "h1622": 22, "h2190": 90}


class Category(str, Enum):
    MPO = "mpo"
    DEP_MPO = "dep-mpo"
    EMG_LAP = "emg_lap"
    NO_LAP = "no_lap"
    UNTRAINED_LAP = "untrained_lap"
    DEP_PRIOR = "dep_prior"  # a_hat from DEP's correlation matrix


LAP_CATEGORIES = frozenset(
    {Category.EMG_LAP, Category.NO_LAP, Category.UNTRAINED_LAP, Category.DEP_PRIOR}
)


class ExperimentGroup(str, Enum):
    # runs before 2026-08-15 live under early_tests/
    EARLY_TESTS = "early_tests"
    FINAL_EXPERIMENTS = "final_experiments"

class RewardVariant(str, Enum):
    FULL = "full"
    ONLYVELREW = "onlyvelrew"
    GAUSSIANVEL = "gaussianvel"

class DecoderSource(str, Enum):
    POOLED = "pooled"
    AB06 = "AB06"
    AB20 = "AB20"
    AB06_CORRECTED = "AB06_corrected"  # depth=1, threshold=0.6, scale=0.9


@dataclass(frozen=True)
class RunSpec:
    category: Category
    body: str
    experiment_group: ExperimentGroup = ExperimentGroup.FINAL_EXPERIMENTS
    reward_variant: RewardVariant = RewardVariant.FULL
    seed: int = 0

    clip: bool = False  # -> env_args.clip_actions (True clips to [0, 0.5], False to [0, 1])
    net_size: int = 256  # -> mpo_args.hidden_size

    # LAP-only knobs, ignored for category in (MPO, DEP_MPO).
    dim_latent: int = 6  # "k"
    mapped_residual_weight: float = 0.5  # "w"; residual_weight stays 1.0 for unmapped muscles
    residual_weight: float = 1.0
    # per-actuator override of mapped_residual_weight
    loosened_actuators: tuple[str, ...] = ()
    loosened_residual_weight: Optional[float] = None
    mirror_left: bool = True
    mirror_mode: str = "static"
    decoder_source: DecoderSource = DecoderSource.POOLED

    # DEP-MPO only; value from the DEP-RL authors' shipped configs
    dep_kappa: int = 1000

    resume: bool = True
    steps: str = "2e7"  # training budget, environment steps

    # non-released decoders; final_experiments looks under f"{synergy_root}_priors/"
    synergy_root: str = str(RESULTS_DIR / "synergy")
    # released decoders (AB06_corrected, one per k) live here instead
    priors_root: str = str(PRIORS_DIR)
    baselines_root: str = str(RUNS_DIR)

    def __post_init__(self) -> None:
        if self.body not in BODIES:
            raise ValueError(f"unknown body {self.body!r}, expected one of {sorted(BODIES)}")
        if self.dim_latent <= 0:
            raise ValueError(f"dim_latent must be positive, got {self.dim_latent}")
        if self.net_size <= 0:
            raise ValueError(f"net_size must be positive, got {self.net_size}")
        if self.mirror_mode != "static":
            raise ValueError(f"mirror_mode must be 'static' (phase mirroring removed 2026-08-18), got {self.mirror_mode!r}")
        if bool(self.loosened_actuators) != (self.loosened_residual_weight is not None):
            raise ValueError(
                "loosened_actuators and loosened_residual_weight must be set together "
                f"(got loosened_actuators={self.loosened_actuators!r}, "
                f"loosened_residual_weight={self.loosened_residual_weight!r})"
            )

    @property
    def num_acts(self) -> int:
        return BODIES[self.body]

    @property
    def is_lap(self) -> bool:
        return self.category in LAP_CATEGORIES

    @property
    def is_dep(self) -> bool:
        return self.category == Category.DEP_MPO

    @property
    def is_default_recipe(self) -> bool:
        """net256 + noclip. Anything else goes to final_experiments/other/."""
        return self.net_size == 256 and not self.clip and not self.loosened_actuators
