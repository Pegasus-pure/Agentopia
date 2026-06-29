from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Optional, Set


def generate_run_id(now: Optional[datetime] = None) -> str:
    """Return run_id in MMDDHHMM from local time (8 digits)."""
    dt = now or datetime.now()
    return dt.strftime("%m%d%H%M")


def get_run_data_dir(base_world: str, run_id: str) -> str:
    """Compose data directory name with run suffix."""
    run_id = str(run_id).strip()
    if len(run_id) != 8 or not run_id.isdigit():
        raise ValueError(f"invalid run_id: {run_id}")
    return f"{base_world}_{run_id}"


def ensure_run_world_data(base_world: str, run_id: str) -> Path:
    """Ensure data/{base_world}_{run_id} exists; copy from data/{base_world} if missing.

    Notes:
    - To ensure this run's map follows the preset (e.g. preset=llm_campus_town with persist=run),
      the top-level locations.json is explicitly skipped when copying the base world directory.
      This lets LocationStore generate or reuse the correct map file per config/preset,
      rather than inheriting the default map from the base directory.

    Returns destination path.
    """
    src = Path("data") / base_world
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(src.as_posix())

    dst = Path("data") / get_run_data_dir(base_world, run_id)
    if dst.exists():
        return dst

    def _ignore(dirpath: str, names: list[str]) -> Set[str]:
        """Skip locations.json - each run should generate its own or use shared_map."""
        p = Path(dirpath)
        ignore: Set[str] = set()
        if p == src and "locations.json" in names:
            ignore.add("locations.json")
        return ignore

    shutil.copytree(src, dst, ignore=_ignore)

    return dst


def save_run_config(data_dir: str, config: dict) -> None:
    """Save the effective config (with CLI overrides applied) to the run directory."""
    import json

    dst = Path("data") / data_dir / "config.json"
    with dst.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
