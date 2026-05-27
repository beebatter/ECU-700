from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_if_present(start_dir: str | Path | None = None) -> None:
    env_path = _find_env_file(start_dir)
    if env_path is None:
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def env(name: str, default: str | None = None) -> str | None:
    load_dotenv_if_present()
    return os.getenv(name, default)


def bool_env(name: str, default: bool = False) -> bool:
    value = env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _find_env_file(start_dir: str | Path | None = None) -> Path | None:
    current = Path(start_dir).expanduser().resolve() if start_dir else Path.cwd().resolve()
    for path in (current, *current.parents):
        candidate = path / ".env"
        if candidate.exists():
            return candidate
    return None

