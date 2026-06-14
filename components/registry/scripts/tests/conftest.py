"""Test configuration and fixtures for pipeline tests.

Requires Python 3.11+ (datetime.UTC). Tests are automatically skipped on older versions.
"""

from __future__ import annotations

import sys

# On Python < 3.11, datetime.UTC doesn't exist. We set a flag that tests check.
CAN_RUN_TESTS = sys.version_info >= (3, 11)
