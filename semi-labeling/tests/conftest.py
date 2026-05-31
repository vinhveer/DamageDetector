"""Make the semi-labeling package root importable for tests.

Tests run as `python -m pytest semi-labeling/tests`; ensure the semi-labeling
directory (containing shared/, steps/, tools/) is on sys.path.
"""
import sys
from pathlib import Path

SEMI_LABELING_DIR = Path(__file__).resolve().parents[1]
if str(SEMI_LABELING_DIR) not in sys.path:
    sys.path.insert(0, str(SEMI_LABELING_DIR))
