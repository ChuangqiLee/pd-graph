# Microglia–Dopaminergic Neuron Communication Programs in Parkinson's Disease

This repository contains the analysis code for the paper:

> **Graph-based discovery and external validation of microglia–dopaminergic neuron communication programs in Parkinson's disease**

## Overview

This study integrates multi-cohort single-cell RNA-seq data from the human substantia nigra and midbrain to discover intercellular communication programs linking microglia and dopaminergic neurons in Parkinson's disease (PD). A four-step computational pipeline constructs a donor-level cell communication atlas, trains heterogeneous graph transformer models for age and disease classification, refines biologically interpretable communication programs, and validates top candidates through independent holdout cohorts, spatial transcriptomics (Visium), and genetic triangulation (GTEx cis-eQTLs and PD GWAS).

The two promoted mechanistic axes are:
- **APOE–lipoprotein receptor signaling** (lead candidate; spatial, genetic, and internal concordance)
- **SLIT–ROBO direct guidance** (priority replication candidate; partial spatial and genetic support)

## Pipeline Architecture

The pipeline is organized into four sequential steps:

### Step 1 — Atlas Construction
Builds a multi-cohort single-cell atlas from public substantia nigra and midbrain datasets (Agarwal et al. 2020; Adams et al. 2024; Lee et al. 2023), with quality control, batch integration via Harmony, and standardized cell type annotations.

### Step 2 — Graph-Based Modeling
Constructs cell-type-specific bipartite communication graphs (sender–receiver) per donor, where nodes represent cells and edges encode ligand–receptor interaction scores. Trains:
- An **age regression model** (graph encoder → continuous age prediction)
- An **age-conditioned PD classification model** (heterogeneous graph transformer with domain-adversarial source invariance)

Model architecture: `HeteroGraphTransformerEncoder` with edge-role attention, context-conditioned FiLM modulation, and donor-grouped cross-validation with Platt calibration.

### Step 3 — Program Refinement and State Mapping
Refines model-derived ligand–receptor pair rankings into biologically coherent communication programs using pathway family assignment, gene family heuristics, and multi-task support scoring. Maps programs onto cell lineage states (e.g., microglial stress-response state, dopaminergic synaptic-organizer states) and evaluates donor-level trajectories.

### Step 4 — External Validation
Validates top programs through three independent evidence layers:
- **Internal holdout** — Smajic et al. 2022 (midbrain) and a withheld substantia nigra companion cohort
- **Spatial transcriptomics** — Ma et al. 2025 substantia nigra Visium cohort (10 sections)
- **Genetic triangulation** — GTEx v8 cis-eQTLs in brain tissues + PD GWAS summary statistics (FinnGen, GCST90275127)

Final promotion decisions integrate all layers into a structured evidence matrix with prespecified gate logic.

## Repository Structure

```
code-release/
├── README.md
├── scripts/
│   ├── step1/                     # Atlas construction
│   │   ├── build_pd_definitive_manifest.py
│   │   ├── download_pd_definitive.py
│   │   ├── audit_pd_definitive.py
│   │   ├── build_step1b_atlas.py
│   │   ├── repair_step1b_atlas.py
│   │   ├── visualize_step1b_atlas.py
│   │   ├── joint_check_step1b_atlas.py
│   │   └── stage_gse253462_nigra_holdout.py
│   ├── step2/                     # Graph modeling
│   │   ├── common.py              # Shared helpers, I/O, project paths
│   │   ├── graph_models.py        # GNN and Transformer architectures
│   │   ├── training_utils.py      # CV splits, calibration, weighting
│   │   ├── resources.py           # NicheNet/CellChat LR database loaders
│   │   ├── build_step2_backbone_manifest.py
│   │   ├── build_step2_comm_gene_panel.py
│   │   ├── apply_step2_dopaminergic_rescue.py
│   │   ├── rehydrate_step2_expression.py
│   │   ├── build_step2_cohorts.py
│   │   ├── build_step2_lr_universe.py
│   │   ├── build_step2_baselines.py
│   │   ├── build_step2_graphs.py
│   │   ├── train_step2_age_model.py
│   │   ├── train_step2_pd_model.py
│   │   ├── run_step2_robustness.py
│   │   ├── interpret_step2_models.py
│   │   ├── export_step2_results.py
│   │   ├── visualize_step2_results.py
│   │   ├── prepare_step2_experiment.py
│   │   └── summarize_step2_experiments.py
│   ├── step3/                     # Program refinement
│   │   ├── common.py              # Program definitions, lineage modules, heuristics
│   │   ├── freeze_step3_inputs.py
│   │   ├── build_step3_candidate_registry.py
│   │   ├── refine_step3_programs.py
│   │   ├── compute_step3_pair_scores.py
│   │   ├── model_step3_donor_trajectories.py
│   │   ├── map_step3_task_support.py
│   │   ├── build_step3_lineage_states.py
│   │   ├── map_step3_state_support.py
│   │   ├── extract_step3_exemplars.py
│   │   ├── render_step3_figures.py
│   │   ├── run_step3_confound_checks.py
│   │   ├── build_step3_hypothesis_reconciliation.py
│   │   ├── rank_step3_programs.py
│   │   ├── export_step3_handoff.py
│   │   └── run_step3_pipeline.py
│   ├── step4/                     # External validation
│   │   ├── common.py              # Validation registry, anchor semantics, gate logic
│   │   ├── freeze_step4_inputs.py
│   │   ├── build_step4_validation_registry.py
│   │   ├── build_step4_genetic_resources.py
│   │   ├── run_step4_gene_level_analyses.py
│   │   ├── run_step4_program_level_genetics.py
│   │   ├── ingest_step4_primary_visium.py
│   │   ├── map_step4_states_to_spatial.py
│   │   ├── score_step4_spatial_programs.py
│   │   ├── run_step4_primary_spatial_tests.py
│   │   ├── run_step4_secondary_spatial_replication.py
│   │   ├── run_step4_internal_robustness.py
│   │   ├── integrate_step4_evidence.py
│   │   ├── build_step4_reports.py
│   │   └── run_step4_pipeline.py
│   ├── archive_full_refresh_outputs.py
│   └── summarize_full_refresh_archives.py
├── configs/
│   ├── step1b/                    # Atlas QC and metadata schema
│   │   ├── step1b_qc.yaml
│   │   └── step1b_metadata_schema.yaml
│   ├── step2/                     # Graph construction and model hyperparameters
│   │   ├── step2_backbone.yaml
│   │   ├── step2_comm_panel.yaml
│   │   ├── step2_cohorts.yaml
│   │   ├── step2_dopaminergic_rescue.yaml
│   │   ├── step2_graphs.yaml
│   │   ├── step2_lr_universe.yaml
│   │   ├── step2_model_age.yaml
│   │   ├── step2_model_pd.yaml
│   │   └── step2_interpretation.yaml
│   └── step3/                     # Program refinement and handoff
│       ├── step3_inputs.yaml
│       ├── step3_candidate_registry.yaml
│       ├── step3_program_refinement.yaml
│       ├── step3_donor_models.yaml
│       ├── step3_task_support.yaml
│       ├── step3_states.yaml
│       ├── step3_confounds.yaml
│       ├── step3_figures.yaml
│       ├── step3_handoff.yaml
│       └── step3_handoff_exploratory_candidates.yaml
├── jobs/                          # Slurm batch scripts (HPC execution)
│   ├── step2_transformer_single_gpu.sbatch
│   ├── step2_pd_refresh.sbatch
│   ├── step34_refresh.sbatch
│   ├── holdout_refresh.sbatch
│   ├── step1_rebuild_atlas_cpu.sbatch
│   ├── step2_refresh_cpu.sbatch
│   ├── archive_full_refresh.sbatch
│   ├── submit_step2.sh
│   ├── submit_step2_cycle.sh
│   ├── submit_step34.sh
│   ├── submit_holdout_refresh.sh
│   ├── submit_full_refresh_chain.sh
│   └── submit_seed_sweep.sh
├── figures/                       # Figure generation
│   ├── generate_paper_figures.py
│   └── figure_abbreviations.txt
└── tables/                        # Table generation
    ├── build_manuscript_tables.py
    └── format_submission_tables.py
```

## Key Model Architecture

The core graph model (`scripts/step2/graph_models.py`) implements:

- **`HeteroGraphTransformerEncoder`** — A heterogeneous graph transformer with:
  - Edge-role multi-head attention (sender/receiver role embeddings)
  - Edge-conditioned attention bias from ligand–receptor interaction features
  - FiLM-style context modulation (source, cohort, region, sex embeddings + graph topology numerics)
  - Role-aware pooling (separate sender/receiver/global/edge mean and max pools)

- **`AgeConditionedPDModel`** — PD classifier that:
  - Encodes the communication graph via the transformer
  - Conditions disease classification on a learned age representation
  - Includes an auxiliary age regression head
  - Applies domain-adversarial gradient reversal for source invariance

- **`AgeModel`** — Standalone age regression for pretraining the graph encoder

Training uses donor-grouped stratified K-fold cross-validation, donor-grouped Platt calibration, balanced sample weighting (donor + source), and mixed-precision (AMP).

## Data Sources

### Discovery cohorts
| Source | GEO/CxG ID | Region | Reference |
|--------|-----------|--------|-----------|
| Agarwal et al. | GSE143353 | Substantia nigra | *Nat. Commun.* 11, 4183 (2020) |
| Adams et al. | GSE270553 | Midbrain | *Nat. Aging* 4, 364–378 (2024) |
| Lee et al. | GSE202210 | Midbrain | *Sci. Adv.* 9, eabo2467 (2023) |

### Holdout cohorts
| Source | GEO ID | Region | Reference |
|--------|--------|--------|-----------|
| Smajic et al. | GSE157783 | Midbrain | *Brain* 145, 964–978 (2022) |
| Adams et al. (companion) | GSE253462 | Substantia nigra | Same study family as discovery |

### Spatial validation
| Source | Region | Reference |
|--------|--------|-----------|
| Ma et al. | Substantia nigra (Visium) | *Nat. Commun.* 16, 7146 (2025) |

### Genetic resources
| Resource | Description |
|----------|-------------|
| GTEx v8 | Brain-tissue cis-eQTLs (substantia nigra, caudate, putamen, frontal cortex, cortex) |
| FinnGen / GCST90275127 | PD GWAS summary statistics (Kim et al. *Nat. Genet.* 56, 27–36, 2024) |

## Requirements

### Python dependencies
```
python >= 3.12
numpy
pandas
scipy
scikit-learn
torch >= 2.0
anndata
scanpy
matplotlib
seaborn
networkx
pyyaml
requests
pyreadr
rdata
```

### Optional (for figure generation)
```
pubfig
adjustText
graphviz
```

### Hardware
- Steps 1, 3, 4: CPU-only (64 GB+ RAM recommended)
- Step 2 training: Single GPU (NVIDIA A100 or equivalent; mixed-precision enabled)

## Execution Order

The pipeline is designed to run sequentially. Within each step, scripts should be executed in the order implied by their naming convention:

```bash
# Step 1 — Atlas
python scripts/step1/build_pd_definitive_manifest.py
python scripts/step1/download_pd_definitive.py
python scripts/step1/build_step1b_atlas.py
python scripts/step1/repair_step1b_atlas.py
python scripts/step1/visualize_step1b_atlas.py
python scripts/step1/joint_check_step1b_atlas.py
python scripts/step1/stage_gse253462_nigra_holdout.py

# Step 2 — Graph modeling
python scripts/step2/build_step2_backbone_manifest.py
python scripts/step2/build_step2_comm_gene_panel.py
python scripts/step2/apply_step2_dopaminergic_rescue.py
python scripts/step2/rehydrate_step2_expression.py
python scripts/step2/build_step2_cohorts.py
python scripts/step2/build_step2_lr_universe.py
python scripts/step2/build_step2_baselines.py
python scripts/step2/build_step2_graphs.py
python scripts/step2/train_step2_age_model.py
python scripts/step2/train_step2_pd_model.py
python scripts/step2/run_step2_robustness.py
python scripts/step2/interpret_step2_models.py
python scripts/step2/export_step2_results.py

# Step 3 — Program refinement
python scripts/step3/run_step3_pipeline.py   # orchestrates all Step 3 sub-scripts

# Step 4 — External validation
python scripts/step4/run_step4_pipeline.py   # orchestrates all Step 4 sub-scripts

# Figures and tables
python figures/generate_paper_figures.py
python tables/build_manuscript_tables.py
```

For HPC execution, see the Slurm scripts in `jobs/`. Before running, configure:
- Replace `<YOUR_ACCOUNT>` in `.sbatch` files with your Slurm allocation account
- Set `PDNPJ_STEP_CPU_VENV` to point to your CPU Python 3.12 virtual environment
- Set `PDNPJ_STEP4_VENV` to point to your Step 4 host virtual environment (for spatial scoring)
- Adjust `module load` lines for your cluster's module system

## Configuration

All pipeline behavior is controlled by YAML configuration files in `configs/`. Key parameters:

- **Model architecture** (`configs/step2/step2_model_pd.yaml`): hidden_dim=96, num_layers=3, num_heads=4, dropout=0.15, hetero_transformer family
- **Graph construction** (`configs/step2/step2_graphs.yaml`): k_neighbors=10, top_lr_features_per_edge=16, harmony_dims=24
- **Cross-validation** (`configs/step2/step2_model_pd.yaml`): 3-fold donor-grouped stratified CV, 40% calibration split, Platt scaling with L2=2.0
- **Training** (`configs/step2/step2_model_pd.yaml`): 100 epochs, lr=7e-4, patience=15, label_smoothing=0.02, gradient reversal lambda=0.25

## Reproducibility

- Random seed: 13 (default across all steps)
- All cross-validation splits are donor-grouped to prevent data leakage
- Source invariance training (gradient reversal) mitigates batch effects across GEO sources
- Full pipeline reruns are archived with configs for reproducibility auditing

