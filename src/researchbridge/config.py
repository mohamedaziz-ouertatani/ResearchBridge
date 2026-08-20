from __future__ import annotations

from dotenv import load_dotenv


def load_config() -> None:
    """Load .env into the process environment, if present. Safe to call multiple times."""
    load_dotenv(override=False)
