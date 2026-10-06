#!/usr/bin/env python3

import csv
import os


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(BASE, "tab-paper", "manuscript_tables_csv")
OUT_DIR = os.path.join(BASE, "tab-paper", "manuscript_tables_csv_submission_ready")
NA = "NA"


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def read_csv_rows(path):
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, NA) for field in fieldnames})


def transform_rows(rows, columns):
    transformed = []
    fieldnames = [new_name for new_name, _ in columns]
    for row in rows:
        out = {}
        for new_name, old_name in columns:
            value = row.get(old_name, NA)
            if value in ("", None):
                value = NA
            out[new_name] = value
        transformed.append(out)
    return transformed, fieldnames


TABLE_MAPPINGS = [
    {
        "source": "Table01_human_cohorts_and_external_resources.csv",
        "output": "Table01_submission_ready.csv",
        "table_id": "Table01",
        "columns": [
            ("Source/accession", "source_accession"),
            ("Resource label", "resource_label"),
            ("Resource category", "resource_category"),
            ("Modality", "modality"),
            ("Tissue/region", "tissue_region"),
            ("Disease groups", "disease_groups"),
            ("Donors by disease", "donor_count_by_disease"),
            ("Samples by disease", "sample_count_by_disease"),
            ("N donors", "n_donors"),
            ("N samples", "n_samples"),
            ("N sections", "n_sections"),
            ("N cases", "n_cases"),
            ("N controls", "n_controls"),
            ("Age availability", "age_availability"),
            ("Sex availability", "sex_availability"),
            ("Pipeline role", "pipeline_role"),
            ("Discovery-atlas inclusion status", "discovery_atlas_inclusion_status"),
            ("Independence notes", "independence_notes"),
        ],
    },
    {
        "source": "Table02_task_coverage_and_evaluation_design.csv",
        "output": "Table02_submission_ready.csv",
        "table_id": "Table02",
        "columns": [
            ("Task", "task_name"),
            ("Sender cell type", "sender_cell_type"),
            ("Receiver cell type", "receiver_cell_type"),
            ("Included cohorts", "included_cohorts"),
            ("Included sources", "included_sources"),
            ("Disease groups", "disease_groups"),
            ("N donors", "n_donors"),
            ("N graph units", "n_graph_units"),
            ("Sender cell threshold", "sender_cell_count_threshold"),
            ("Receiver cell threshold", "receiver_cell_count_threshold"),
            ("Receiver threshold source overrides", "receiver_threshold_source_overrides"),
            ("Donor-grouped CV grouping", "donor_grouped_cv_grouping"),
            ("Calibration grouping", "calibration_grouping"),
            ("LBD bridge status", "bridge_cohort_status"),
            ("Holdout-source designation", "holdout_source_designation"),
        ],
    },
    {
        "source": "Table03_pd_model_performance_and_calibration.csv",
        "output": "Table03_submission_ready.csv",
        "table_id": "Table03",
        "columns": [
            ("Entry type", "row_type"),
            ("Fold", "fold"),
            ("Validation source", "source_id"),
            ("AUROC", "auroc"),
            ("Balanced accuracy", "balanced_accuracy"),
            ("Brier score", "brier_score"),
            ("Log-loss", "log_loss"),
            ("Decision threshold", "decision_threshold"),
            ("Calibration intercept", "calibration_intercept"),
            ("Calibration slope", "calibration_slope"),
            ("N fit groups", "n_fit_groups"),
            ("N calibration groups", "n_calibration_groups"),
            ("N validation groups", "n_val_groups"),
            ("N grouped units", "n_groups"),
            ("N graphs", "n_graphs"),
            ("Metric scope", "metric_scope"),
        ],
    },
    {
        "source": "Table04_baseline_and_ablation_comparison.csv",
        "output": "Table04_submission_ready.csv",
        "table_id": "Table04",
        "columns": [
            ("Record type", "record_type"),
            ("Comparator ID", "comparison_id"),
            ("Task", "task_name"),
            ("Analysis", "analysis"),
            ("Model family", "model_family"),
            ("Mean absolute error", "mean_absolute_error"),
            ("Mean AUROC", "mean_auroc"),
            ("Mean balanced accuracy", "mean_balanced_accuracy"),
            ("Mean Brier score", "mean_brier_score"),
            ("Mean log-loss", "mean_log_loss"),
            ("Top claim tier", "top_claim_tier"),
            ("Top promoted program", "top_promoted_program"),
            ("Final Step 4 score", "final_step4_score"),
            ("Bridge-cohort mean PD probability", "bridge_mean_pd_probability"),
            ("Comparison scope note", "comparison_scope_note"),
        ],
    },
    {
        "source": "Table05_step3_refined_program_landscape.csv",
        "output": "Table05_submission_ready.csv",
        "table_id": "Table05",
        "columns": [
            ("Refined program", "refined_program_name"),
            ("Step 3 program score", "step3_program_score"),
            ("Anchor mean score", "anchor_pair_score_mean"),
            ("Anchor max score", "anchor_pair_score_max"),
            ("Program coherence", "program_coherence"),
            ("Trajectory coherence", "donor_trajectory_coherence"),
            ("Validation readiness", "validation_readiness"),
            ("Dominant task", "dominant_task"),
            ("Anchor pairs", "anchor_pairs"),
            ("Dominant sender state", "dominant_sender_state"),
            ("Dominant receiver state", "dominant_receiver_state"),
            ("Interpretation class", "interpretation_class"),
            ("Caveats", "caveat_summary"),
            ("Final Step 3 tier", "final_step3_tier"),
            ("Step 4 priority", "step4_validation_priority"),
            ("Exemplar graph IDs", "exemplar_graph_ids"),
            ("Refined program ID", "refined_program_id"),
        ],
    },
    {
        "source": "Table06_state_mapping_and_donor_trajectory_summary.csv",
        "output": "Table06_submission_ready.csv",
        "table_id": "Table06",
        "columns": [
            ("Program", "program_name"),
            ("Dominant sender state ID", "dominant_sender_state_id"),
            ("Dominant sender state label", "dominant_sender_state_label"),
            ("Dominant receiver state ID", "dominant_receiver_state_id"),
            ("Dominant receiver state label", "dominant_receiver_state_label"),
            ("Ligand expression support", "ligand_expression_support"),
            ("Receptor expression support", "receptor_expression_support"),
            ("High-contribution state-pair enrichment", "high_contribution_state_pair_enrichment"),
            ("Control-age beta", "control_age_beta"),
            ("PD shift size", "pd_shift_size"),
            ("PD-age interaction beta", "pd_age_interaction_beta"),
            ("LBD bridge class", "lbd_bridge_class"),
            ("Trajectory class", "trajectory_class"),
            ("Holdout direction concordance", "holdout_direction_concordance"),
            ("Holdout source concordance", "holdout_source_concordance"),
            ("Trajectory source note", "trajectory_source"),
            ("State-support note", "state_support_note"),
            ("Program ID", "program_id"),
        ],
    },
    {
        "source": "Table07_primary_spatial_donor_level_results.csv",
        "output": "Table07_submission_ready.csv",
        "table_id": "Table07",
        "columns": [
            ("Program", "program_name"),
            ("Expected PD direction", "expected_pd_direction"),
            ("N PD donors", "n_pd_donors"),
            ("N control donors", "n_control_donors"),
            ("SNpc adjacency beta", "snpc_adjacency_beta"),
            ("SNpc adjacency P value", "snpc_adjacency_p_value"),
            ("SNpc adjacency FDR", "snpc_adjacency_fdr"),
            ("SNpc-surrounding beta", "contrast_beta"),
            ("SNpc-surrounding P value", "contrast_p_value"),
            ("SNpc-surrounding FDR", "contrast_fdr"),
            ("Same-spot beta", "same_spot_beta"),
            ("Same-spot P value", "same_spot_p_value"),
            ("Aligned metric count", "aligned_metric_count"),
            ("Directional P (SNpc adjacency)", "directional_p_snpc"),
            ("Directional P (SNpc-surrounding)", "directional_p_contrast"),
            ("Directional P (same spot)", "directional_p_same_spot"),
            ("Composite directional P", "composite_directional_p_value"),
            ("Direction-match flag", "direction_match"),
            ("Primary support bin", "primary_spatial_support_bin"),
            ("Program ID", "program_id"),
        ],
    },
    {
        "source": "Table08_primary_visium_section_qc_summary.csv",
        "output": "Table08_submission_ready.csv",
        "table_id": "Table08",
        "columns": [
            ("Section accession", "section_accession"),
            ("Donor ID", "donor_id"),
            ("Disease", "disease_group"),
            ("Sex", "sex"),
            ("Age text", "age_text"),
            ("Age years", "age_years"),
            ("Raw spot count", "raw_spot_count"),
            ("Kept spot count", "kept_spot_count"),
            ("Reconstructed SNpc spot count", "reconstructed_snpc_spot_count"),
            ("Reconstructed surrounding_tissue spot count", "reconstructed_surrounding_tissue_spot_count"),
            ("Mean total counts", "mean_total_counts_kept"),
            ("Mean genes", "mean_n_genes_kept"),
            ("Mitochondrial percentage", "mean_pct_mito_kept"),
            ("Mean dopaminergic score", "mean_dopaminergic_score_kept"),
            ("QC thresholds", "qc_thresholds"),
            ("Compartment-definition rule", "compartment_definition_rule"),
        ],
    },
    {
        "source": "Table09_secondary_spatial_replication_summary.csv",
        "output": "Table09_submission_ready.csv",
        "table_id": "Table09",
        "columns": [
            ("Program", "program_name"),
            ("Scoring gene source", "scoring_gene_source"),
            ("Total-program score", "total_program_score"),
            ("Total genes present", "total_program_n_genes"),
            ("Mean quadrant score", "mean_quadrant_score"),
            ("Direction-consistent quadrants", "n_quadrants_direction_consistent"),
            ("Secondary support bin", "secondary_spatial_support_bin"),
            ("Program ID", "program_id"),
        ],
    },
    {
        "source": "Table10_internal_holdout_robustness_summary.csv",
        "output": "Table10_submission_ready.csv",
        "table_id": "Table10",
        "columns": [
            ("Program", "program_name"),
            ("Holdout source", "holdout_source_id"),
            ("Weighted holdout PD-minus-control effect", "program_weighted_holdout_pd_minus_control"),
            ("Source mean PD-minus-control effect", "source_mean_holdout_pd_minus_control"),
            ("Expected internal direction", "expected_internal_direction"),
            ("Support tier", "holdout_support_tier"),
            ("Weighted match fraction", "holdout_weighted_match_fraction"),
            ("Holdout sources available", "n_holdout_sources_available"),
            ("Holdout sources matching direction", "n_holdout_sources_direction_match"),
            ("Source direction match", "source_direction_match"),
            ("Program ID", "program_id"),
        ],
    },
    {
        "source": "Table11_genetic_support_summary.csv",
        "output": "Table11_submission_ready.csv",
        "table_id": "Table11",
        "columns": [
            ("Entry level", "row_level"),
            ("Program", "program_name"),
            ("Gene symbol", "gene_symbol"),
            ("Instrumentability", "instrumentability_status"),
            ("Best tissue", "best_tissue"),
            ("Best SNP", "best_snp_id"),
            ("MR method", "best_primary_mr_method"),
            ("FinnGen beta", "finngen_mr_beta"),
            ("FinnGen P value", "finngen_mr_p_value"),
            ("FinnGen FDR", "finngen_mr_fdr"),
            ("GCST90275127 beta", "gcst90275127_mr_beta"),
            ("GCST90275127 P value", "gcst90275127_mr_p_value"),
            ("GCST90275127 FDR", "gcst90275127_mr_fdr"),
            ("Mean F", "mean_F"),
            ("Coloc H4 proxy", "coloc_h4_proxy"),
            ("Cross-outcome direction consistent", "cross_outcome_direction_consistent"),
            ("Gene support bin", "gene_support_bin"),
            ("N genes", "n_genes"),
            ("N strong gene support", "n_strong_gene_support"),
            ("N moderate gene support", "n_moderate_gene_support"),
            ("MAGMA-like P value", "magma_like_p_value"),
            ("MAGMA-like FDR", "magma_like_fdr"),
            ("Empirical permutation P value", "empirical_p_value"),
            ("Empirical permutation FDR", "empirical_fdr"),
            ("Locus-overlap count", "locus_overlap_count"),
            ("Program support bin", "program_support_bin"),
            ("Program ID", "program_id"),
        ],
    },
    {
        "source": "Table12_integrated_step4_evidence_matrix.csv",
        "output": "Table12_submission_ready.csv",
        "table_id": "Table12",
        "columns": [
            ("Program", "program_name"),
            ("Program ID", "program_id"),
            ("Batch", "program_batch"),
            ("Anchor pairs", "anchor_pairs"),
            ("Gene members", "gene_members"),
            ("Sender state", "sender_state_id"),
            ("Receiver state", "receiver_state_id"),
            ("Expected PD direction", "expected_pd_direction"),
            ("Caveat flags", "caveat_flags"),
            ("Primary spatial support bin", "primary_spatial_support_bin"),
            ("SNpc adjacency beta", "snpc_adjacency_beta"),
            ("SNpc adjacency P value", "snpc_adjacency_p_value"),
            ("SNpc adjacency FDR", "snpc_adjacency_fdr"),
            ("SNpc-surrounding beta", "contrast_beta"),
            ("SNpc-surrounding P value", "contrast_p_value"),
            ("SNpc-surrounding FDR", "contrast_fdr"),
            ("Secondary spatial support bin", "secondary_spatial_support_bin"),
            ("Total-program score", "total_program_score"),
            ("Mean quadrant score", "mean_quadrant_score"),
            ("Direction-consistent quadrants", "n_quadrants_direction_consistent"),
            ("Program genetic support bin", "program_genetic_support_bin"),
            ("N strong gene support", "n_strong_gene_support"),
            ("N moderate gene support", "n_moderate_gene_support"),
            ("Empirical permutation P value", "empirical_p_value"),
            ("Empirical permutation FDR", "empirical_fdr"),
            ("Holdout PD-minus-control effect", "mean_holdout_pd_minus_control"),
            ("Holdout direction match", "holdout_direction_match"),
            ("Holdout support tier", "holdout_support_tier"),
            ("Holdout sources available", "n_holdout_sources_available"),
            ("Holdout sources matching direction", "n_holdout_sources_direction_match"),
            ("Source-dominant flag", "source_dominant_flag"),
            ("Negative-holdout flag", "negative_holdout_flag"),
            ("Source-flip flag", "source_flip_flag"),
            ("Region-flip flag", "region_flip_flag"),
            ("Cell-count-link flag", "cell_count_link_flag"),
            ("Primary spatial component", "primary_spatial_component"),
            ("Secondary spatial component", "secondary_spatial_component"),
            ("Program-level genetic component", "program_level_genetic_component"),
            ("Cross-dataset consistency component", "cross_dataset_consistency_component"),
            ("Total penalty", "total_penalty"),
            ("Final Step 4 score", "final_step4_score"),
            ("Spatial primary gate", "spatial_primary_gate"),
            ("Spatial replication gate", "spatial_replication_gate"),
            ("Gene-level genetics gate", "genetic_gene_level_gate"),
            ("Program-level genetics gate", "genetic_program_level_gate"),
            ("Genetic consensus gate", "genetic_consensus_gate"),
            ("Spatial overall gate", "spatial_overall_gate"),
            ("Internal stability gate", "internal_stability_gate"),
            ("Claim tier", "claim_tier"),
            ("Promotion decision", "promotion_decision"),
            ("Final evidence class", "final_evidence_class"),
            ("Decision rationale", "decision_rationale"),
        ],
    },
    {
        "source": "Table13_candidate_decision_log.csv",
        "output": "Table13_submission_ready.csv",
        "table_id": "Table13",
        "columns": [
            ("Program", "program_name"),
            ("Program ID", "program_id"),
            ("Spatial primary gate", "spatial_primary_gate"),
            ("Spatial replication gate", "spatial_replication_gate"),
            ("Gene-level genetics gate", "genetic_gene_level_gate"),
            ("Program-level genetics gate", "genetic_program_level_gate"),
            ("Internal stability gate", "internal_stability_gate"),
            ("Claim tier", "claim_tier"),
            ("Promotion decision", "promotion_decision"),
            ("Final Step 4 score", "final_step4_score"),
            ("Final evidence class", "final_evidence_class"),
            ("Source-dominant flag", "source_dominant_flag"),
            ("Source-flip flag", "source_flip_flag"),
            ("Region-flip flag", "region_flip_flag"),
            ("Negative-holdout flag", "negative_holdout_flag"),
            ("Cell-count-link flag", "cell_count_link_flag"),
            ("Blocking issue flag", "blocking_issue_flag"),
            ("Rerun/hold-back decision", "rerun_or_holdback_decision"),
            ("Failure rationale", "failure_rationale"),
        ],
    },
    {
        "source": "Table14_sensitivity_and_confound_summary.csv",
        "output": "Table14_submission_ready.csv",
        "table_id": "Table14",
        "columns": [
            ("Refined program", "refined_program_name"),
            ("Refined program ID", "refined_program_id"),
            ("Dominant task", "dominant_task"),
            ("Label-confidence sensitivity counts", "label_confidence_sensitivity_counts"),
            ("Cell-count robustness correlations", "cell_count_robustness_correlations"),
            ("Task specificity score", "task_specificity_score"),
            ("Receiver specificity task", "receiver_specificity_task"),
            ("Task consistency status", "task_consistency_status"),
            ("Sender-state mapping sensitivity", "sender_state_mapping_sensitivity"),
            ("Receiver-state mapping sensitivity", "receiver_state_mapping_sensitivity"),
            ("Source majority share", "source_majority_share"),
            ("Source direction flip", "source_direction_flip"),
            ("Region direction flip", "region_direction_flip"),
            ("Sex-sensitivity status", "sex_sensitivity_status"),
            ("Holdout sign behavior", "holdout_direction_concordance"),
            ("Holdout support metric", "holdout_support_metric"),
            ("Promotion/concern status", "promotion_status"),
            ("Caveat summary", "caveat_summary"),
        ],
    },
]


def main():
    ensure_dir(OUT_DIR)
    index_rows = []
    for mapping in TABLE_MAPPINGS:
        src_path = os.path.join(SOURCE_DIR, mapping["source"])
        out_path = os.path.join(OUT_DIR, mapping["output"])
        rows = read_csv_rows(src_path)
        transformed, fieldnames = transform_rows(rows, mapping["columns"])
        write_csv(out_path, transformed, fieldnames)
        index_rows.append(
            {
                "Table ID": mapping["table_id"],
                "Source file": mapping["source"],
                "Submission-ready file": mapping["output"],
                "Row count": len(transformed),
            }
        )

    write_csv(
        os.path.join(OUT_DIR, "submission_ready_index.csv"),
        index_rows,
        ["Table ID", "Source file", "Submission-ready file", "Row count"],
    )


if __name__ == "__main__":
    main()
