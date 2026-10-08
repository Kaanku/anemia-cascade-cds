"""
CDS Pipeline - Central Configuration
====================================
Every notebook starts by importing this file:
    import sys
    sys.path.insert(0, '/content/drive/MyDrive/0000_Kaan_CDS/src')
    from cds_config import *

Last updated: 2026-04-01 (Chat 0)
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Apply plot style automatically
from plot_style import PALETTE, set_academic_style, save_fig, despine_all, add_panel_label

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

@dataclass
class CDSPaths:
    """All project paths — managed from a single place."""

    cds_root: Path = Path("/content/drive/MyDrive/0000_Kaan_CDS")
    ml_root: Path = Path("/content/drive/MyDrive/0000_ML_Models")

    @property
    def original_data(self) -> Path:
        return self.ml_root / "data"

    @property
    def frozen_models(self) -> Path:
        return self.cds_root / "frozen_models"

    @property
    def src(self) -> Path:
        return self.cds_root / "src"

    @property
    def probabilities(self) -> Path:
        return self.cds_root / "data" / "probabilities"

    def module_dir(self, module: str, sub: str = "") -> Path:
        p = self.cds_root / "modules" / module
        if sub:
            p = p / sub
        return p

    @property
    def reports(self) -> Path:
        return self.cds_root / "reports"

    @property
    def pub_figures(self) -> Path:
        return self.cds_root / "reports" / "publication" / "figures"

    @property
    def pub_tables(self) -> Path:
        return self.cds_root / "reports" / "publication" / "tables"

    @property
    def cds_per_sample(self) -> Path:
        return self.cds_root / "reports" / "cds_per_sample"

    @property
    def notebooks(self) -> Path:
        return self.cds_root / "notebooks"


PATHS = CDSPaths()


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentInfo:
    """Information for a single experiment."""
    name: str
    stage: int
    mode: str
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0

    @property
    def dir(self) -> Path:
        return PATHS.frozen_models / self.name

    @property
    def model_dir(self) -> Path:
        return self.dir / "AutoGluon_Model"

    @property
    def train_file(self) -> Path:
        return self.dir / f"Stage{self.stage}_TRAIN_{self.mode}.xlsx"

    @property
    def test_file(self) -> Path:
        return self.dir / f"Stage{self.stage}_TEST_{self.mode}.xlsx"

    @property
    def config_file(self) -> Path:
        return self.dir / "training_config.joblib"

    @property
    def threshold_file(self) -> Path:
        return self.dir / "optimal_threshold.joblib"

    @property
    def summary_file(self) -> Path:
        return self.dir / f"stage{self.stage}_results_summary.joblib"


# ─── 4 EXPERIMENTS ────────────────────────────────────────────────────────────

EXP_S1_CBC = ExperimentInfo(
    name="20260201_1719_Stage1_CBC_Only",
    stage=1, mode="CBC_Only",
    n_train=912, n_test=229, n_features=23
)
EXP_S2_CBC = ExperimentInfo(
    name="20260201_2035_Stage2_CBC_Only",
    stage=2, mode="CBC_Only",
    n_train=655, n_test=165, n_features=27
)
EXP_S1_BIO = ExperimentInfo(
    name="20260202_0436_Stage1_CBC_BIO",
    stage=1, mode="CBC_BIO",
    n_train=912, n_test=229, n_features=32
)
EXP_S2_BIO = ExperimentInfo(
    name="20260202_0507_Stage2_CBC_BIO",
    stage=2, mode="CBC_BIO",
    n_train=655, n_test=165, n_features=47
)

EXPERIMENTS = {
    "s1_cbc": EXP_S1_CBC,
    "s2_cbc": EXP_S2_CBC,
    "s1_bio": EXP_S1_BIO,
    "s2_bio": EXP_S2_BIO,
}

SCENARIOS = {
    "CBC_Only": {"s1": EXP_S1_CBC, "s2": EXP_S2_CBC},
    "CBC_BIO":  {"s1": EXP_S1_BIO, "s2": EXP_S2_BIO},
}


# =============================================================================
# CLASS LABELS & MAPPING
# =============================================================================

# NOTE: Internal keys preserved for model/data compatibility.
# Display mapping: IAS→AAC (Associated Anemia Causes), DAS→OAC (Other Anemia Causes),
#                  DEA→IDA, HGB_HTZ→HGB HTZ, LDH→LD.
STAGE1_CLASSES = {0: "DAS", 1: "IAS"}
STAGE1_POS_LABEL = 1

STAGE2_CLASSES = {0: "DEA", 1: "HA", 2: "HGB HTZ", 3: "Normal"}

DIAGNOSIS_TO_STAGE2 = {"IDA": 0, "HA": 1, "HGB_HTZ": 2, "NORMAL": 3}
DAS_LABELS = frozenset(["IRRELEVANT", "IRREVELANT"])
IAS_LABELS = frozenset(["IDA", "HA", "HGB_HTZ", "NORMAL"])


# =============================================================================
# COLOR MAPPING (derived from plot_style.PALETTE)
# =============================================================================

# Disease classes → PALETTE colors
CLASS_COLORS = {
    "DEA":      PALETTE["highlight"],  # Ruby Red
    "HA":       PALETTE["base1"],      # Steel Blue
    "HGB HTZ":  PALETTE["accent1"],    # Emerald Green
    "Normal":   PALETTE["accent2"],    # Muted Orange
    "DAS":      PALETTE["base2"],      # Pale Gray
    "IAS":      PALETTE["accent3"],    # Amethyst Purple
}

# Confidence zones → PALETTE colors
ZONE_COLORS = {
    "HIGH":   PALETTE["accent1"],    # Green — safe, automation
    "MEDIUM": PALETTE["accent2"],    # Orange — caution, technician
    "LOW":    PALETTE["highlight"],  # Red — danger, expert
}

# Scenario colors
SCENARIO_COLORS = {
    "CBC_Only": PALETTE["base1"],    # Steel Blue
    "CBC_BIO":  PALETTE["highlight"],# Ruby Red
}

# Stage colors
STAGE_COLORS = {
    "Stage 1": PALETTE["base1"],
    "Stage 2": PALETTE["accent3"],
}


# =============================================================================
# THESIS BASELINE METRICS (Table 8 + Table 9)
# =============================================================================

THESIS_BASELINE = {
    "CBC_Only": {
        "stage1": {
            "accuracy": 0.764, "precision": 0.782, "f1": 0.850,
            "sensitivity": 0.933, "specificity": 0.328,
            "roc_auc": 0.710, "pr_auc": 0.839,
            "threshold": 0.500,
        },
        "stage2": {
            "accuracy": 0.667,
            "macro_precision": 0.665, "macro_sensitivity": 0.664,
            "macro_f1": 0.662, "weighted_f1": 0.666,
            "micro_f1": 0.667,
            "roc_auc_ovr": 0.883, "roc_auc_ovo": 0.883,
        },
    },
    "CBC_BIO": {
        "stage1": {
            "accuracy": 0.838, "precision": 0.860, "f1": 0.892,
            "sensitivity": 0.927, "specificity": 0.609,
            "roc_auc": 0.869, "pr_auc": 0.943,
            "threshold": 0.500,
        },
        "stage2": {
            "accuracy": 0.770,
            "macro_precision": 0.768, "macro_sensitivity": 0.763,
            "macro_f1": 0.764, "weighted_f1": 0.770,
            "micro_f1": 0.770,
            "roc_auc_ovr": 0.929, "roc_auc_ovo": 0.930,
        },
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prob_filename(stage: int, mode: str, split: str, fmt: str = "parquet") -> str:
    """Standard file name: stage1_cbc_only_oof_probs.parquet"""
    mode_short = mode.lower()
    return f"stage{stage}_{mode_short}_{split}_probs.{fmt}"


def setup_notebook(notebook_name: str = ""):
    """Called at the start of every notebook."""
    import warnings
    warnings.filterwarnings("ignore")
    set_academic_style()

    print(f"{'=' * 60}")
    print(f"CDS Pipeline — {notebook_name}")
    print(f"{'=' * 60}")
    print(f"  Project root : {PATHS.cds_root}")
    print(f"  Frozen models: {PATHS.frozen_models}")

    assert PATHS.cds_root.exists(), f"CDS root not found: {PATHS.cds_root}"
    assert PATHS.frozen_models.exists(), f"Frozen models not found"
    for key, exp in EXPERIMENTS.items():
        assert exp.dir.exists(), f"{key} not found: {exp.dir}"

    print(f"  Plot style   : Tufte Academic ({PALETTE['highlight']} highlight)")
    print(f"  ✓ 4 experiments verified")
    print(f"{'=' * 60}\n")


def load_experiment_config(exp: "ExperimentInfo") -> dict:
    import joblib
    return joblib.load(exp.config_file)


def load_threshold(exp: "ExperimentInfo") -> float:
    import joblib
    return joblib.load(exp.threshold_file)
