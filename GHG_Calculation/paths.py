"""Unified input/output paths shared by every script in this repository.

Rules
-----
* every raw data file is read from   <repo>/input/
* every result file is written to    <repo>/output/
* an intermediate file produced by one script and consumed by another one
  also lives in <repo>/output/, and the downstream script reads it from there

Paths are resolved relative to this file, so the scripts can be launched from
any working directory.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def inp(*parts):
    """Path of a raw input file under input/."""
    return os.path.join(INPUT_DIR, *parts)


def out(*parts):
    """Path of a result file under output/, creating its parent directory."""
    path = os.path.join(OUTPUT_DIR, *parts)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def out_dir(*parts):
    """Path of a result directory under output/, created if missing."""
    path = os.path.join(OUTPUT_DIR, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def require_input(*parts):
    """Path of a required input file, with a clear error when it is absent."""
    path = inp(*parts)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing input file: {path}\n"
            f"place it under {INPUT_DIR} before running this script")
    return path
