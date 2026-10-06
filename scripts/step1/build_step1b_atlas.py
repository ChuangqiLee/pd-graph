#!/usr/bin/env python3
"""Backward-compatible wrapper for the Step 1 atlas build pipeline."""

from __future__ import annotations

from _run_step1_script import run_step1_script


if __name__ == "__main__":
    run_step1_script("atlas_build/build_step1b_atlas.py")
