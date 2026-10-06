#!/usr/bin/env python3
"""Backward-compatible wrapper for staging the GSE253462 nigra holdout."""

from __future__ import annotations

from _run_step1_script import run_step1_script


if __name__ == "__main__":
    run_step1_script("dataset_build/stage_gse253462_nigra_holdout.py")
