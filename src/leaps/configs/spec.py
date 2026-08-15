"""RunSpec: the single source of truth for a training run's config.yaml.

Replaces hand-copying config.yaml files (217 of them by 2026-08-15, with
real schema drift as a result -- 61/217 missing wandb.tags, some older
emg_lap/no_lap "full" configs missing env_args.self_contact_coeff/run
entirely). A RunSpec is the small set of knobs that actually varies; every
other field in config.yaml is derived from it in builder.py/naming.py.

category encodes null_prior/untrained_decoder/plain rather than separate
booleans -- those three are mutually exclusive in every run this project
has ever done, so an independent-booleans design would let callers build
combinations that can't actually occur (e.g. null_prior=True and
untrained_decoder=True on the same run).
"""

from dataclasses import dataclass
from enum import Enum

# body -> num_acts (muscle count), read off each model's actual SCONE file,
# not assumed -- confirmed from real config.yaml agent expressions.
BODIES = {"h0918": 18, "h1622": 22, "h2190": 90}


class Category(str, Enum):
    MPO = "mpo"
    DEP_MPO = "dep-mpo"
    EMG_LAP = "emg_lap"
    NO_LAP = "no_lap"
    UNTRAINED_LAP = "untrained_lap"
    # "other" deliberately excluded: bespoke one-off mechanisms
    # (self_synergy, coactivation_reward, corrnoise, ...), each with its own
    # tonic.environment expression, not part of this systematic grid.


LAP_CATEGORIES = frozenset({Category.EMG_LAP, Category.NO_LAP, Category.UNTRAINED_LAP})


class RewardVariant(str, Enum):
    FULL = "full"
    ONLYVELREW = "onlyvelrew"
    GAUSSIANVEL = "gaussianvel"


class DecoderSource(str, Enum):
    POOLED = "pooled"
    AB06 = "AB06"
    AB20 = "AB20"


@dataclass(frozen=True)
class RunSpec:
    category: Category
    body: str
    reward_variant: RewardVariant = RewardVariant.FULL
    seed: int = 0

    clip: bool = False  # -> env_args.clip_actions (True clips to [0,0.5], False to [0,1.0])
    net_size: int = 256  # -> mpo_args.hidden_size iff != 256

    # LAP-only knobs, ignored for category in (MPO, DEP_MPO).
    dim_latent: int = 6  # "k"
    mapped_residual_weight: float = 0.5  # "w" -- NOT residual_weight, which stays 1.0
    residual_weight: float = 1.0
    mirror_left: bool = True
    decoder_source: DecoderSource = DecoderSource.POOLED

    # DEP_MPO-only.
    dep_kappa: int = 1000  # documented constant (see reference_leaps_related_papers memory)

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

    @property
    def num_acts(self) -> int:
        return BODIES[self.body]

    @property
    def is_lap(self) -> bool:
        return self.category in LAP_CATEGORIES

    @property
    def is_dep(self) -> bool:
        return self.category == Category.DEP_MPO
