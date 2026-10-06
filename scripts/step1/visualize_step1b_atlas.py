#!/usr/bin/env python3
"""Backward-compatible wrapper for the Step 1 atlas visualization script."""

from __future__ import annotations

from _run_step1_script import run_step1_script


if __name__ == "__main__":
    run_step1_script("qc_review/visualize_step1b_atlas.py")
