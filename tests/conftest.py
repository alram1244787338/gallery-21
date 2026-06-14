from __future__ import annotations

import datetime
import sys
from pathlib import Path

# Polyfill datetime.UTC for Python < 3.11 (the project requires 3.12+ but CI
# runners or developer laptops may have older interpreters during quick checks).
if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]  # noqa: UP017

# The registry scripts use sibling imports like ``from _utils.io import ...``.
# Ensure the scripts directory is on sys.path so pytest can resolve them.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "components" / "registry" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
