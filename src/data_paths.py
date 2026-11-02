"""Shared locations for reference data and generated analysis outputs."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
REFERENCE_DATA_DIR = DATA_DIR / "reference_data"
OUTPUTS_DIR = DATA_DIR / "outputs"


def analysis_output_path(run_dir: str | Path) -> Path:
    """Keep each measurement run's reconstructed fields in its own folder."""
    return OUTPUTS_DIR / Path(run_dir).resolve().name / "integrated_fields.npz"


def fit_output_path(input_path: str | Path, dimension: str) -> Path:
    """Keep fits beside generated inputs, or create a folder for other inputs."""
    source = Path(input_path).resolve()
    output_root = OUTPUTS_DIR.resolve()
    directory = source.parent if source.is_relative_to(output_root) else output_root / source.stem
    return directory / f"{source.stem}_fit_{dimension}.npz"
