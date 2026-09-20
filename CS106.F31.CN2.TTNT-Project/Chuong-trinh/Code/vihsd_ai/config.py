"""Configuration and per-run paths; no training or data access at import time."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

VIHSD_COMMIT = "70fe50c9c94145e3b9be5614a048cbd5b480fa54"
VIHSD_URL = (
    f"https://raw.githubusercontent.com/sonlam1102/vihsd/{VIHSD_COMMIT}/data/vihsd.zip"
)
VIHSD_SHA256 = "1d823c4c86a8aede83555c64da4e1be24146967aac6b30cdaf8603d4910f328d"
SOURCE_TO_BINARY = {"CLEAN": "SAFE", "OFFENSIVE": "TOXIC", "HATE": "TOXIC"}
ID_TO_SOURCE = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
BINARY_LABELS = ["SAFE", "TOXIC"]


@dataclass(frozen=True)
class RunConfig:
    run_mode: str = "FULL_WITH_NEURAL"
    seed: int = 42
    neural_epochs: int = 3
    data_path: Path | None = None
    output_root: Path | None = None

    def __post_init__(self):
        if self.run_mode not in {"SMOKE", "FULL", "FULL_WITH_NEURAL"}:
            raise ValueError(f"Unknown run mode: {self.run_mode}")
        if self.neural_epochs < 1:
            raise ValueError("neural_epochs must be >= 1")


@dataclass(frozen=True)
class RunContext:
    config: RunConfig
    run_id: str
    data_path: Path
    output_dir: Path


def create_run(config: RunConfig) -> RunContext:
    """Create a new output directory; existing experiment artifacts remain intact."""
    colab = Path("/content").exists()
    data_path = (
        Path(config.data_path)
        if config.data_path is not None
        else Path("/content/vihsd.zip" if colab else "vihsd.zip")
    )
    output_root = (
        Path(config.output_root)
        if config.output_root is not None
        else Path("/content/vihsd_ai_outputs" if colab else "outputs/notebook_run")
    )
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "_"
        + config.run_mode.lower()
    )
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    return RunContext(config, run_id, data_path, output_dir)
