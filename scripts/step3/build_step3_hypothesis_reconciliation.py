#!/usr/bin/env python3.12
"""Reconcile the original hypothesis with observed Step 2 and Step 3 results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown


DEFAULT_OUTPUT_DIR = Path("data/processed/step3/08_hypothesis_reconciliation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 3 hypothesis reconciliation.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked_pathways = pd.read_csv(project_path("data/processed/step2/08_interpretation/step2_ranked_pathways.csv"))
    ranked_pairs = pd.read_csv(project_path("data/processed/step2/08_interpretation/step2_ranked_lr_pairs.csv"))
    confounds = pd.read_csv(project_path("data/processed/step3/07_confounds/step3_promotion_status.csv"))

    def pathway_rank(pathway: str) -> str:
        match = ranked_pathways.index[ranked_pathways["pathway_family"].eq(pathway)]
        return str(int(match[0] + 1)) if len(match) else "NA"

    def pair_rank(pair: str) -> str:
        match = ranked_pairs.index[ranked_pairs["pair_key"].eq(pair)]
        return str(int(match[0] + 1)) if len(match) else "NA"

    rows = [
        {
            "original_expected_mechanism": "cytokine signaling",
            "observed_step2_support": f"pathway rank {pathway_rank('cytokine signaling')}; IL18->IL1RAPL1 pair rank {pair_rank('IL18->IL1RAPL1')}",
            "current_step3_status": "secondary support",
            "interpretation": "The inflammatory interface is present, but it was not the dominant discriminator.",
        },
        {
            "original_expected_mechanism": "stress and APP signaling",
            "observed_step2_support": f"pathway rank {pathway_rank('stress response module')}; APOE->APP pair rank {pair_rank('APOE->APP')}",
            "current_step3_status": "core supported mechanism",
            "interpretation": "The model strongly prioritized APP-centered stress and vulnerability signaling.",
        },
        {
            "original_expected_mechanism": "glutamatergic signaling",
            "observed_step2_support": f"pathway rank {pathway_rank('Glutamate')}; SLC1A3|GLS->GRIK2 pair rank {pair_rank('SLC1A3|GLS->GRIK2')}",
            "current_step3_status": "high-priority interpretation target",
            "interpretation": "A coherent receptor cluster supports a real program rather than an isolated pair.",
        },
        {
            "original_expected_mechanism": "guidance and adhesion signaling",
            "observed_step2_support": f"SLIT pathway rank {pathway_rank('SLIT')}; DSCAM->DSCAM pair rank {pair_rank('DSCAM->DSCAM')}",
            "current_step3_status": "expanded beyond original expectation",
            "interpretation": "Guidance and synaptic adhesion emerged more strongly than expected from the original inflammatory framing.",
        },
        {
            "original_expected_mechanism": "trophic maintenance signaling",
            "observed_step2_support": f"NRG pathway rank {pathway_rank('NRG')}; ADAM17->ERBB4 pair rank {pair_rank('ADAM17->ERBB4')}",
            "current_step3_status": "aging-associated secondary program",
            "interpretation": "ERBB4-centered support is present, but looks more maintenance-like than disease-gain driven.",
        },
    ]
    reconciliation = pd.DataFrame(rows)
    reconciliation.to_csv(output_dir / "step3_hypothesis_reconciliation.csv", index=False)

    report = "\n".join(
        [
            "# Step 3 Hypothesis Reconciliation",
            "",
            "## Reconciliation Table",
            markdown_table(reconciliation),
            "",
            "## What the model prioritized",
            "- APP-centered stress signaling, SLIT/ROBO guidance, glutamatergic receptor programs, and synaptic/contact recognition programs.",
            "",
            "## What the model did not prioritize",
            "- Classical cytokine signaling was retained as context, but it was not the main signal captured by the current Step 2 run.",
        ]
    )
    write_markdown(output_dir / "step3_hypothesis_reconciliation.md", report)


if __name__ == "__main__":
    main()

