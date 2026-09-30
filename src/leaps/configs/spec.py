"""The RunSpec dataclass.
"""

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
    DEP_PRIOR = "dep_prior"  # a_hat sourced from DEP's self-organized C matrix instead of the EMG decoder -- content-source control, not a competing controller


LAP_CATEGORIES = frozenset(
    {Category.EMG_LAP, Category.NO_LAP, Category.UNTRAINED_LAP, Category.DEP_PRIOR}
)


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
    AB06_CORRECTED = "AB06_corrected"  # 2026-08-16: depth=1,threshold=0.6,scale=0.9 (locked recipe) + _to_z_domain fix


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
    # per-actuator override on top of mapped_residual_weight -- e.g. h1622's
    # glut_med_r/glut_med_l, which get a real EMG channel (unlike their
    # add_mag antagonist) but are otherwise forced through the same tight
    # blend as every other mapped muscle. () reproduces every existing
    # config's behavior exactly.
    loosened_actuators: tuple[str, ...] = ()
    loosened_residual_weight: Optional[float] = None
    mirror_left: bool = True
    mirror_mode: str = "static"  # phase-mirroring mechanism removed 2026-08-18 -- static is the only option now
    decoder_source: DecoderSource = DecoderSource.POOLED

    # DEP_MPO-only. 1000 -- verified 2026-08-18 directly against the DEP-RL
    # authors' own shipped baseline configs for these exact bodies (both the
    # live martius-lab/depRL GitHub repo and their Google-Drive-hosted
    # pretrained checkpoints' config.yaml), not inferred from a table. The
    # earlier 1896 default (DEP-RL paper Table 4c "human-run") was reasoned
    # from a different task/embodiment's hyperparameter listing, never
    # confirmed to match these bodies -- superseded once the actual shipped
    # configs were checked.
    dep_kappa: int = 1000

    resume: bool = True

    # 2026-08-19: within final_experiments, builder._decoder_paths() resolves
    # decoders under f"{synergy_root}_priors/{single_subject,pooled_subjects,
    # small_pooled}/" instead of this flat root -- see that dir's README.
    # This value itself is still used as-is for early_tests (unchanged, flat
    # layout, kept for golden-file test fidelity) and as the string prefix
    # the "_priors" suffix is appended to.
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
        """net256 + noclip -- the only recipe currently populating
        final_experiments/{category}/... (verified 2026-08-19: all 55
        configs there). Anything else (net512, clip=True, future
        ablations) routes to final_experiments/other/ instead, via
        writer.config_path()/naming.derive_tonic_name(), so the main
        tree's names never need net/clip tokens at all -- see
        naming.derive_run_name()."""
        return self.net_size == 256 and not self.clip and not self.loosened_actuators
