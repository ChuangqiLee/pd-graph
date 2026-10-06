#!/usr/bin/env python3

import csv
import gzip
import json
import os
from collections import defaultdict
from statistics import mean


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(BASE, "MANUSCRIPT_WRITING_PACKAGE_20260402")
OUT_DIR = os.path.join(BASE, "tab-paper", "manuscript_tables_csv")
NA = "NA"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_nonempty(value) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text not in ("", "NA", "nan", "NaN", "None", "null")


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, "r", newline="")


def read_csv_rows(path: str):
    with open_text(path) as handle:
        return list(csv.DictReader(handle))


def parse_float(value):
    if not is_nonempty(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value):
    if not is_nonempty(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def mean_or_na(values):
    clean = [value for value in values if value is not None]
    if not clean:
        return NA
    return mean(clean)


def join_sorted(values):
    clean = sorted({str(value) for value in values if is_nonempty(value)})
    return "|".join(clean) if clean else NA


def availability_status(n_present: int, n_total: int) -> str:
    if n_total <= 0:
        return "not_staged"
    if n_present <= 0:
        return "not_available"
    if n_present == n_total:
        return "complete"
    return "partial"


def bool_text(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    text = str(value).strip()
    if text in ("True", "False"):
        return text
    if text in ("true", "false"):
        return text.title()
    return text if text else NA


def format_value(value):
    if value is None:
        return NA
    if isinstance(value, float):
        return "{:.12g}".format(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    text = str(value)
    return text if is_nonempty(text) else NA


def write_csv(path: str, rows, fieldnames) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fieldnames})


def parse_json_field(value):
    if not is_nonempty(value):
        return []
    return json.loads(value)


def parse_markdown_tables(path: str):
    with open(path, "r") as handle:
        lines = [line.rstrip("\n") for line in handle]
    tables = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("|"):
            block = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                block.append(lines[idx].strip())
                idx += 1
            if len(block) >= 2 and set(block[1].replace("|", "").strip()) <= set("-: "):
                header = [cell.strip() for cell in block[0].strip("|").split("|")]
                rows = []
                for raw_row in block[2:]:
                    values = [cell.strip() for cell in raw_row.strip("|").split("|")]
                    if len(values) < len(header):
                        values.extend([""] * (len(header) - len(values)))
                    rows.append(dict(zip(header, values)))
                tables.append(rows)
            continue
        idx += 1
    return tables


def donor_set_update(target, group_key, donor, sample):
    if is_nonempty(donor):
        target["donors"].add(donor)
        target["donors_by_disease"][group_key].add(donor)
    if is_nonempty(sample):
        target["samples"].add(sample)
        target["samples_by_disease"][group_key].add(sample)


def make_source_stat():
    return {
        "donors": set(),
        "samples": set(),
        "donors_by_disease": defaultdict(set),
        "samples_by_disease": defaultdict(set),
        "donors_with_age": set(),
        "donors_with_sex": set(),
        "modalities": set(),
        "tissues": set(),
    }


def build_source_stats():
    stats = defaultdict(make_source_stat)

    atlas_obs = os.path.join(BASE, "data", "processed", "step1b", "pd_definitive_first_pass_atlas_obs.csv.gz")
    with open_text(atlas_obs) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_id = row["source_id"]
            donor = row.get("donor_id") or row.get("sample_id")
            sample = row.get("sample_id") or row.get("sample_accession") or donor
            disease = row.get("disease_group") or "unknown"
            entry = stats[source_id]
            donor_set_update(entry, disease, donor, sample)
            if is_nonempty(row.get("age_years")):
                entry["donors_with_age"].add(donor)
            if is_nonempty(row.get("sex_standardized")):
                entry["donors_with_sex"].add(donor)
            if is_nonempty(row.get("modality")):
                entry["modalities"].add(row["modality"])
            if is_nonempty(row.get("brain_region_standardized")):
                entry["tissues"].add(row["brain_region_standardized"])
            elif is_nonempty(row.get("tissue_original")):
                entry["tissues"].add(row["tissue_original"])

    graph_manifest = os.path.join(BASE, "data", "processed", "step3", "00_inputs", "step2_core", "step2_graph_manifest.csv")
    for row in read_csv_rows(graph_manifest):
        source_id = row["source_id"]
        donor = row.get("donor_id") or row.get("split_group_id")
        sample = row.get("sample_id") or donor
        disease = row.get("disease_group") or "unknown"
        entry = stats[source_id]
        donor_set_update(entry, disease, donor, sample)
        if is_nonempty(row.get("age_years")):
            entry["donors_with_age"].add(donor)
        if is_nonempty(row.get("sex_standardized")):
            entry["donors_with_sex"].add(donor)
        entry["tissues"].add(row.get("brain_region_standardized"))

    return stats


def summarize_source_entry(entry):
    donors = len(entry["donors"])
    samples = len(entry["samples"])
    donor_counts = []
    sample_counts = []
    for disease in sorted(entry["donors_by_disease"]):
        donor_counts.append("%s=%d" % (disease, len(entry["donors_by_disease"][disease])))
    for disease in sorted(entry["samples_by_disease"]):
        sample_counts.append("%s=%d" % (disease, len(entry["samples_by_disease"][disease])))
    return {
        "n_donors": donors if donors else NA,
        "n_samples": samples if samples else NA,
        "donor_count_by_disease": "|".join(donor_counts) if donor_counts else NA,
        "sample_count_by_disease": "|".join(sample_counts) if sample_counts else NA,
        "age_availability": availability_status(len(entry["donors_with_age"]), donors),
        "sex_availability": availability_status(len(entry["donors_with_sex"]), donors),
        "modality": join_sorted(entry["modalities"]),
        "tissue_region": join_sorted(entry["tissues"]),
        "disease_groups": join_sorted(entry["donors_by_disease"].keys()),
    }


def build_table1():
    source_stats = build_source_stats()
    gse157783 = read_csv_rows(os.path.join(BASE, "data", "processed", "step2", "07_robustness", "gse157783_holdout", "gse157783_patient_celltype_counts.csv"))
    gse253462 = read_csv_rows(os.path.join(BASE, "data", "processed", "step2", "07_robustness", "gse253462_holdout", "gse253462_patient_celltype_counts.csv"))
    primary_visium = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "primary_visium_manifest.csv"))
    geomx_scores = read_csv_rows(os.path.join(BASE, "data", "processed", "step4", "03_spatial", "05_secondary_replication", "geomx_roi_program_scores.csv"))
    resource_manifest = {row["resource_id"]: row for row in read_csv_rows(os.path.join(BASE, "data", "processed", "step4", "02_genetics", "00_resources", "resource_manifest.csv"))}

    special_stats = {}

    def build_patient_stat(rows, donor_key, sample_key, disease_key):
        entry = make_source_stat()
        for row in rows:
            donor = row.get(donor_key)
            sample = row.get(sample_key) or donor
            disease = row.get(disease_key) or "unknown"
            donor_set_update(entry, disease, donor, sample)
        return summarize_source_entry(entry)

    special_stats["GSE157783"] = build_patient_stat(gse157783, "patient", "patient", "disease_group")
    special_stats["GSE157783"]["modality"] = "snRNA-seq"
    special_stats["GSE157783"]["tissue_region"] = "midbrain"
    special_stats["GSE157783"]["age_availability"] = "not_staged"
    special_stats["GSE157783"]["sex_availability"] = "not_staged"

    special_stats["GSE253462"] = build_patient_stat(gse253462, "donor_id", "gsm_accession", "disease_group")
    special_stats["GSE253462"]["modality"] = "snRNA-seq"
    special_stats["GSE253462"]["tissue_region"] = "substantia_nigra"
    special_stats["GSE253462"]["age_availability"] = "complete"
    special_stats["GSE253462"]["sex_availability"] = "complete"

    visium_entry = make_source_stat()
    for row in primary_visium:
        donor = row.get("donor_id")
        sample = row.get("sample_accession")
        disease = row.get("disease_group")
        donor_set_update(visium_entry, disease, donor, sample)
        if is_nonempty(row.get("age_years")) or is_nonempty(row.get("age")):
            visium_entry["donors_with_age"].add(donor)
        if is_nonempty(row.get("sex")):
            visium_entry["donors_with_sex"].add(donor)
        visium_entry["modalities"].add("Visium spatial transcriptomics")
        visium_entry["tissues"].add("substantia_nigra")
    special_stats["GSE253975"] = summarize_source_entry(visium_entry)
    special_stats["GSE253975"]["n_sections"] = len(primary_visium)

    geomx_blocks = sorted({row["sheet_name"] for row in geomx_scores if row.get("sheet_name")})
    special_stats["geomx_secondary"] = {
        "n_donors": NA,
        "n_samples": 1,
        "n_sections": len(geomx_blocks),
        "donor_count_by_disease": NA,
        "sample_count_by_disease": "pd_vs_control_workbook=1",
        "age_availability": "not_staged",
        "sex_availability": "not_staged",
        "modality": "GeoMx DSP processed DEG workbook",
        "tissue_region": "substantia_nigra_pars_compacta",
        "disease_groups": "control|pd",
    }

    source_metadata = [
        {
            "source_accession": "b0f0b447-ac37-45b0-b1bf-5c0b7d871120",
            "resource_label": "CELLxGENE SNpc disease collection",
            "resource_category": "single_cell_discovery",
            "pipeline_role": "primary discovery atlas cohort",
            "discovery_atlas_inclusion_status": "included_in_step1_atlas",
            "independence_notes": "Integrated into discovery atlas; used heavily in Step 2 and Step 3.",
        },
        {
            "source_accession": "GSE193688",
            "resource_label": "Paired human midbrain age and PD multiome",
            "resource_category": "single_cell_discovery",
            "pipeline_role": "primary discovery atlas cohort",
            "discovery_atlas_inclusion_status": "included_in_step1_atlas",
            "independence_notes": "Integrated into discovery atlas; midbrain discovery coverage.",
        },
        {
            "source_accession": "GSE140231",
            "resource_label": "Normal substantia nigra snRNA-seq atlas",
            "resource_category": "single_cell_reference",
            "pipeline_role": "normal reference for discovery calibration",
            "discovery_atlas_inclusion_status": "included_in_step1_atlas",
            "independence_notes": "Integrated into discovery atlas as normal substantia nigra reference.",
        },
        {
            "source_accession": "GSE126836",
            "resource_label": "Normal substantia nigra donor matrices",
            "resource_category": "single_cell_reference",
            "pipeline_role": "reference atlas and unknown-age auxiliary donor support",
            "discovery_atlas_inclusion_status": "included_in_step1_atlas",
            "independence_notes": "Integrated into discovery atlas; also supports auxiliary donor coverage.",
        },
        {
            "source_accession": "GSE148434",
            "resource_label": "Substantia nigra PD/control multi-omics",
            "resource_category": "single_cell_auxiliary",
            "pipeline_role": "auxiliary donor expansion and dopaminergic rescue",
            "discovery_atlas_inclusion_status": "excluded_from_original_step1_atlas",
            "independence_notes": "Kept out of original atlas; used as auxiliary coverage rather than discovery-atlas core.",
        },
        {
            "source_accession": "GSE157783",
            "resource_label": "Human midbrain PD/control snRNA-seq",
            "resource_category": "single_cell_holdout",
            "pipeline_role": "independent internal holdout",
            "discovery_atlas_inclusion_status": "excluded_holdout_source",
            "independence_notes": "Independent midbrain holdout used for internal robustness, not discovery integration.",
        },
        {
            "source_accession": "GSE253462",
            "resource_label": "Same-study substantia nigra snRNA-seq companion cohort",
            "resource_category": "single_cell_holdout",
            "pipeline_role": "same-study holdout and calibration-only source",
            "discovery_atlas_inclusion_status": "excluded_same_study_holdout",
            "independence_notes": "Held out because it is the same-study snRNA-seq companion to GSE253975.",
        },
        {
            "source_accession": "GSE253975",
            "resource_label": "Primary human Visium substantia nigra cohort",
            "resource_category": "spatial_validation",
            "pipeline_role": "primary spatial validation",
            "discovery_atlas_inclusion_status": "not_applicable",
            "independence_notes": "Primary external spatial validation dataset; SNpc compartment reconstructed in-project.",
        },
        {
            "source_accession": "PD_HC_total_and_quadrant.xlsx",
            "resource_label": "Shin et al. SNpc GeoMx differential-expression workbook",
            "resource_category": "spatial_replication",
            "pipeline_role": "secondary spatial replication",
            "discovery_atlas_inclusion_status": "not_applicable",
            "independence_notes": "Processed public DEG workbook used for signed enrichment replication, not donor-level mixed models.",
        },
        {
            "source_accession": "gtex_v8_brain_eqtl_api",
            "resource_label": "GTEx v8 brain cis-eQTL API",
            "resource_category": "genetics_exposure",
            "pipeline_role": "genetic exposure resource",
            "discovery_atlas_inclusion_status": "not_applicable",
            "independence_notes": "Used for instrumentability and tissue selection in gene-level genetics analyses.",
        },
        {
            "source_accession": "finngen_r10_g6_parkinson",
            "resource_label": "FinnGen R10 G6_PARKINSON",
            "resource_category": "genetics_outcome",
            "pipeline_role": "primary PD genetics outcome",
            "discovery_atlas_inclusion_status": "not_applicable",
            "independence_notes": "Primary PD GWAS outcome for MR and program-level enrichment.",
        },
        {
            "source_accession": "gcst90275127_multi_ancestry_pd",
            "resource_label": "GCST90275127 multi-ancestry PD meta-analysis",
            "resource_category": "genetics_outcome",
            "pipeline_role": "secondary PD genetics outcome",
            "discovery_atlas_inclusion_status": "not_applicable",
            "independence_notes": "Secondary PD GWAS outcome used for cross-outcome transfer and replication.",
        },
    ]

    rows = []
    for meta in source_metadata:
        source_id = meta["source_accession"]
        if source_id in resource_manifest:
            resource = resource_manifest[source_id]
            row = dict(meta)
            row.update(
                {
                    "modality": resource.get("resource_type"),
                    "tissue_region": resource.get("trait_or_molecule"),
                    "disease_groups": resource.get("trait_or_molecule"),
                    "n_donors": NA,
                    "n_samples": resource.get("n_samples_total") or NA,
                    "n_sections": NA,
                    "n_cases": resource.get("n_cases") or NA,
                    "n_controls": resource.get("n_controls") or NA,
                    "donor_count_by_disease": NA,
                    "sample_count_by_disease": NA,
                    "age_availability": "not_applicable",
                    "sex_availability": "not_applicable",
                }
            )
            rows.append(row)
            continue

        if source_id == "PD_HC_total_and_quadrant.xlsx":
            source_summary = special_stats["geomx_secondary"]
        elif source_id == "GSE253975":
            source_summary = special_stats["GSE253975"]
        elif source_id in special_stats:
            source_summary = special_stats[source_id]
        else:
            source_summary = summarize_source_entry(source_stats[source_id])

        row = dict(meta)
        row.update(
            {
                "modality": source_summary.get("modality", NA),
                "tissue_region": source_summary.get("tissue_region", NA),
                "disease_groups": source_summary.get("disease_groups", NA),
                "n_donors": source_summary.get("n_donors", NA),
                "n_samples": source_summary.get("n_samples", NA),
                "n_sections": source_summary.get("n_sections", NA),
                "n_cases": NA,
                "n_controls": NA,
                "donor_count_by_disease": source_summary.get("donor_count_by_disease", NA),
                "sample_count_by_disease": source_summary.get("sample_count_by_disease", NA),
                "age_availability": source_summary.get("age_availability", NA),
                "sex_availability": source_summary.get("sex_availability", NA),
            }
        )
        rows.append(row)

    fieldnames = [
        "source_accession",
        "resource_label",
        "resource_category",
        "modality",
        "tissue_region",
        "disease_groups",
        "donor_count_by_disease",
        "sample_count_by_disease",
        "n_donors",
        "n_samples",
        "n_sections",
        "n_cases",
        "n_controls",
        "age_availability",
        "sex_availability",
        "pipeline_role",
        "discovery_atlas_inclusion_status",
        "independence_notes",
    ]
    path = os.path.join(OUT_DIR, "Table01_human_cohorts_and_external_resources.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_table2():
    graph_manifest = read_csv_rows(os.path.join(BASE, "data", "processed", "step3", "00_inputs", "step2_core", "step2_graph_manifest.csv"))
    task_groups = defaultdict(lambda: {"sources": set(), "cohorts": set(), "diseases": set(), "donors": set(), "graphs": 0})
    for row in graph_manifest:
        task = row["task_name"]
        entry = task_groups[task]
        entry["sources"].add(row["source_id"])
        entry["cohorts"].add(row["cohort_name"])
        entry["diseases"].add(row["disease_group"])
        entry["donors"].add(row["donor_id"] or row["split_group_id"])
        entry["graphs"] += 1

    task_meta = {
        "microglia_to_dopaminergic_neuron": {
            "sender_cell_type": "microglia",
            "receiver_cell_type": "dopaminergic neuron",
            "sender_cell_count_threshold": 20,
            "receiver_cell_count_threshold": 10,
            "receiver_threshold_source_overrides": "GSE140231=5|GSE148434=5|GSE193688=5",
        },
        "microglia_to_neuron": {
            "sender_cell_type": "microglia",
            "receiver_cell_type": "neuron",
            "sender_cell_count_threshold": 20,
            "receiver_cell_count_threshold": 30,
            "receiver_threshold_source_overrides": NA,
        },
        "astrocyte_to_dopaminergic_neuron": {
            "sender_cell_type": "astrocyte",
            "receiver_cell_type": "dopaminergic neuron",
            "sender_cell_count_threshold": 15,
            "receiver_cell_count_threshold": 10,
            "receiver_threshold_source_overrides": NA,
        },
        "endothelial_to_dopaminergic_neuron": {
            "sender_cell_type": "endothelial cell",
            "receiver_cell_type": "dopaminergic neuron",
            "sender_cell_count_threshold": 15,
            "receiver_cell_count_threshold": 10,
            "receiver_threshold_source_overrides": NA,
        },
    }

    cohort_order = ["control_known_age", "pd_known_age", "lbd_bridge", "unknown_age_auxiliary"]
    rows = []
    for task_name in sorted(task_groups):
        entry = task_groups[task_name]
        ordered_cohorts = [cohort for cohort in cohort_order if cohort in entry["cohorts"]]
        row = dict(task_meta[task_name])
        row.update(
            {
                "task_name": task_name,
                "included_cohorts": "|".join(ordered_cohorts),
                "included_sources": join_sorted(entry["sources"]),
                "disease_groups": join_sorted(entry["diseases"]),
                "n_donors": len(entry["donors"]),
                "n_graph_units": entry["graphs"],
                "donor_grouped_cv_grouping": "split_group_id=source_id::donor_id",
                "calibration_grouping": "donor-grouped Platt scaling on held-out calibration donors",
                "bridge_cohort_status": "present" if "lbd_bridge" in entry["cohorts"] else "absent",
                "holdout_source_designation": "GSE157783=independent_midbrain_holdout|GSE253462=same_study_calibration_holdout",
            }
        )
        rows.append(row)

    fieldnames = [
        "task_name",
        "sender_cell_type",
        "receiver_cell_type",
        "included_cohorts",
        "included_sources",
        "disease_groups",
        "n_donors",
        "n_graph_units",
        "sender_cell_count_threshold",
        "receiver_cell_count_threshold",
        "receiver_threshold_source_overrides",
        "donor_grouped_cv_grouping",
        "calibration_grouping",
        "bridge_cohort_status",
        "holdout_source_designation",
    ]
    path = os.path.join(OUT_DIR, "Table02_task_coverage_and_evaluation_design.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_table3():
    fold_metrics = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "pd_model_fold_metrics.csv"))
    fold_metadata = {row["fold"]: row for row in read_csv_rows(os.path.join(BASE, "data", "processed", "step2", "06_models", "pd_model", "pd_model_fold_metadata.csv"))}
    source_metrics = read_csv_rows(os.path.join(BASE, "data", "processed", "step2", "06_models", "pd_model", "pd_model_source_metrics.csv"))

    rows = []
    for row in fold_metrics:
        meta = fold_metadata[row["fold"]]
        rows.append(
            {
                "row_type": "fold",
                "fold": row["fold"],
                "source_id": NA,
                "decision_threshold": row["decision_threshold"],
                "auroc": row["auroc"],
                "balanced_accuracy": row["balanced_accuracy"],
                "brier_score": row["brier_score"],
                "log_loss": row["log_loss"],
                "calibration_intercept": meta["calibration_intercept"],
                "calibration_slope": meta["calibration_slope"],
                "n_fit_groups": meta["n_fit_groups"],
                "n_calibration_groups": meta["n_calibration_groups"],
                "n_val_groups": meta["n_val_groups"],
                "n_groups": meta["n_val_groups"],
                "n_graphs": NA,
                "metric_scope": "donor_grouped_fold",
            }
        )

    rows.append(
        {
            "row_type": "mean_summary",
            "fold": "mean",
            "source_id": NA,
            "decision_threshold": mean_or_na([parse_float(row["decision_threshold"]) for row in fold_metrics]),
            "auroc": mean_or_na([parse_float(row["auroc"]) for row in fold_metrics]),
            "balanced_accuracy": mean_or_na([parse_float(row["balanced_accuracy"]) for row in fold_metrics]),
            "brier_score": mean_or_na([parse_float(row["brier_score"]) for row in fold_metrics]),
            "log_loss": mean_or_na([parse_float(row["log_loss"]) for row in fold_metrics]),
            "calibration_intercept": mean_or_na([parse_float(row["calibration_intercept"]) for row in fold_metadata.values()]),
            "calibration_slope": mean_or_na([parse_float(row["calibration_slope"]) for row in fold_metadata.values()]),
            "n_fit_groups": NA,
            "n_calibration_groups": NA,
            "n_val_groups": NA,
            "n_groups": NA,
            "n_graphs": NA,
            "metric_scope": "mean_across_folds",
        }
    )

    for row in source_metrics:
        rows.append(
            {
                "row_type": "source_validation",
                "fold": row["fold"],
                "source_id": row["source_id"],
                "decision_threshold": fold_metadata[row["fold"]]["decision_threshold"],
                "auroc": row["auroc"],
                "balanced_accuracy": row["balanced_accuracy"],
                "brier_score": row["brier_score"],
                "log_loss": row["log_loss"],
                "calibration_intercept": fold_metadata[row["fold"]]["calibration_intercept"],
                "calibration_slope": fold_metadata[row["fold"]]["calibration_slope"],
                "n_fit_groups": NA,
                "n_calibration_groups": NA,
                "n_val_groups": NA,
                "n_groups": row["n_groups"],
                "n_graphs": row["n_graphs"],
                "metric_scope": "source_stratified_validation",
            }
        )

    fieldnames = [
        "row_type",
        "fold",
        "source_id",
        "decision_threshold",
        "auroc",
        "balanced_accuracy",
        "brier_score",
        "log_loss",
        "calibration_intercept",
        "calibration_slope",
        "n_fit_groups",
        "n_calibration_groups",
        "n_val_groups",
        "n_groups",
        "n_graphs",
        "metric_scope",
    ]
    path = os.path.join(OUT_DIR, "Table03_pd_model_performance_and_calibration.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_table4():
    baseline = read_csv_rows(os.path.join(BASE, "data", "processed", "step2", "04_baselines", "step2_baseline_metrics.csv"))
    experiments = read_csv_rows(os.path.join(BASE, "data", "processed", "step2", "experiment_summaries", "step2_experiment_leaderboard.csv"))
    full_refresh = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "full_refresh_leaderboard.csv"))
    fold_metrics = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "pd_model_fold_metrics.csv"))
    lbd_bridge = {row["cohort_name"]: row for row in read_csv_rows(os.path.join(BASE, "data", "processed", "step2", "07_robustness", "step2_lbd_bridge_summary.csv"))}
    manuscript_results = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "step4_results_table_for_manuscript.csv"))

    main_row = None
    for row in manuscript_results:
        if row.get("promotion_decision") == "main_figure_candidate":
            main_row = row
            break
    if main_row is None:
        main_row = manuscript_results[0]

    scope_note = "in_study_comparators_only; no defensible external SOTA comparator in staged materials"
    rows = [
        {
            "record_type": "current_gnn_model",
            "comparison_id": "current_pd_model",
            "task_name": "microglia_to_dopaminergic_neuron",
            "analysis": "pd_deviation",
            "model_family": "hetero_transformer",
            "mean_absolute_error": NA,
            "mean_auroc": mean_or_na([parse_float(row["auroc"]) for row in fold_metrics]),
            "mean_balanced_accuracy": mean_or_na([parse_float(row["balanced_accuracy"]) for row in fold_metrics]),
            "mean_brier_score": mean_or_na([parse_float(row["brier_score"]) for row in fold_metrics]),
            "mean_log_loss": mean_or_na([parse_float(row["log_loss"]) for row in fold_metrics]),
            "top_claim_tier": main_row["claim_tier"],
            "top_promoted_program": main_row["program_name"],
            "final_step4_score": main_row["final_step4_score"],
            "bridge_mean_pd_probability": lbd_bridge["lbd_bridge"]["mean_pd_probability"],
            "comparison_scope_note": scope_note,
        }
    ]

    for row in baseline:
        if is_nonempty(row.get("mae_elastic_net")):
            rows.append(
                {
                    "record_type": "baseline_model_family",
                    "comparison_id": "%s_%s_elastic_net" % (row["task_name"], row["analysis"]),
                    "task_name": row["task_name"],
                    "analysis": row["analysis"],
                    "model_family": "elastic_net",
                    "mean_absolute_error": row["mae_elastic_net"],
                    "mean_auroc": NA,
                    "mean_balanced_accuracy": NA,
                    "mean_brier_score": NA,
                    "mean_log_loss": NA,
                    "top_claim_tier": NA,
                    "top_promoted_program": NA,
                    "final_step4_score": NA,
                    "bridge_mean_pd_probability": NA,
                    "comparison_scope_note": scope_note,
                }
            )
        if is_nonempty(row.get("auroc_logistic")) or is_nonempty(row.get("balanced_accuracy_logistic")):
            rows.append(
                {
                    "record_type": "baseline_model_family",
                    "comparison_id": "%s_%s_logistic" % (row["task_name"], row["analysis"]),
                    "task_name": row["task_name"],
                    "analysis": row["analysis"],
                    "model_family": "logistic_regression",
                    "mean_absolute_error": NA,
                    "mean_auroc": row["auroc_logistic"],
                    "mean_balanced_accuracy": row["balanced_accuracy_logistic"],
                    "mean_brier_score": NA,
                    "mean_log_loss": NA,
                    "top_claim_tier": NA,
                    "top_promoted_program": NA,
                    "final_step4_score": NA,
                    "bridge_mean_pd_probability": NA,
                    "comparison_scope_note": scope_note,
                }
            )
        if is_nonempty(row.get("mae_random_forest")) or is_nonempty(row.get("auroc_random_forest")):
            rows.append(
                {
                    "record_type": "baseline_model_family",
                    "comparison_id": "%s_%s_random_forest" % (row["task_name"], row["analysis"]),
                    "task_name": row["task_name"],
                    "analysis": row["analysis"],
                    "model_family": "random_forest",
                    "mean_absolute_error": row.get("mae_random_forest") or NA,
                    "mean_auroc": row.get("auroc_random_forest") or NA,
                    "mean_balanced_accuracy": row.get("balanced_accuracy_random_forest") or NA,
                    "mean_brier_score": NA,
                    "mean_log_loss": NA,
                    "top_claim_tier": NA,
                    "top_promoted_program": NA,
                    "final_step4_score": NA,
                    "bridge_mean_pd_probability": NA,
                    "comparison_scope_note": scope_note,
                }
            )

    for row in experiments:
        rows.append(
            {
                "record_type": "step2_ablation_experiment",
                "comparison_id": row["experiment"],
                "task_name": "microglia_to_dopaminergic_neuron",
                "analysis": "pd_deviation",
                "model_family": "hetero_transformer_ablation",
                "mean_absolute_error": NA,
                "mean_auroc": row["mean_auroc"],
                "mean_balanced_accuracy": row["mean_balanced_accuracy"],
                "mean_brier_score": row["mean_brier_score"],
                "mean_log_loss": row["mean_log_loss"],
                "top_claim_tier": NA,
                "top_promoted_program": NA,
                "final_step4_score": NA,
                "bridge_mean_pd_probability": row["bridge_mean_pd_probability"],
                "comparison_scope_note": scope_note,
            }
        )

    for row in full_refresh:
        rows.append(
            {
                "record_type": "full_refresh_archive",
                "comparison_id": row["archive"],
                "task_name": "microglia_to_dopaminergic_neuron",
                "analysis": "step4_refresh",
                "model_family": "pipeline_refresh_archive",
                "mean_absolute_error": NA,
                "mean_auroc": row["mean_pd_auroc"],
                "mean_balanced_accuracy": row["mean_pd_balanced_accuracy"],
                "mean_brier_score": NA,
                "mean_log_loss": NA,
                "top_claim_tier": row["top_claim_tier"],
                "top_promoted_program": row["top_program_id"],
                "final_step4_score": row["top_final_step4_score"],
                "bridge_mean_pd_probability": NA,
                "comparison_scope_note": scope_note,
            }
        )

    fieldnames = [
        "record_type",
        "comparison_id",
        "task_name",
        "analysis",
        "model_family",
        "mean_absolute_error",
        "mean_auroc",
        "mean_balanced_accuracy",
        "mean_brier_score",
        "mean_log_loss",
        "top_claim_tier",
        "top_promoted_program",
        "final_step4_score",
        "bridge_mean_pd_probability",
        "comparison_scope_note",
    ]
    path = os.path.join(OUT_DIR, "Table04_baseline_and_ablation_comparison.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_table5(program_names):
    top10 = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "step3_top10_refined_programs.csv"))
    exemplars = read_csv_rows(os.path.join(BASE, "data", "processed", "step3", "06_exemplars", "step3_exemplar_subgraph_manifest.csv"))
    exemplar_map = defaultdict(list)
    for row in exemplars:
        exemplar_map[row["refined_program_id"]].append(row["graph_id"])
    rows = []
    for row in top10:
        rows.append(
            {
                "refined_program_id": row["refined_program_id"],
                "refined_program_name": row["refined_program_name"],
                "step3_program_score": row["step3_program_score"],
                "anchor_pair_score_mean": row["anchor_pair_score_mean"],
                "anchor_pair_score_max": row["anchor_pair_score_max"],
                "program_coherence": row["program_coherence"],
                "donor_trajectory_coherence": row["donor_trajectory_coherence"],
                "validation_readiness": row["validation_readiness"],
                "dominant_task": row["dominant_task"],
                "anchor_pairs": row["anchor_pairs"],
                "dominant_sender_state": row["dominant_sender_state"],
                "dominant_receiver_state": row["dominant_receiver_state"],
                "interpretation_class": row["interpretation_class"],
                "caveat_summary": row["caveat_summary"],
                "final_step3_tier": row["final_step3_tier"],
                "step4_validation_priority": row["step4_validation_priority"],
                "exemplar_graph_ids": join_sorted(exemplar_map.get(row["refined_program_id"], [])),
            }
        )
        program_names.setdefault(row["refined_program_id"], row["refined_program_name"])

    fieldnames = [
        "refined_program_id",
        "refined_program_name",
        "step3_program_score",
        "anchor_pair_score_mean",
        "anchor_pair_score_max",
        "program_coherence",
        "donor_trajectory_coherence",
        "validation_readiness",
        "dominant_task",
        "anchor_pairs",
        "dominant_sender_state",
        "dominant_receiver_state",
        "interpretation_class",
        "caveat_summary",
        "final_step3_tier",
        "step4_validation_priority",
        "exemplar_graph_ids",
    ]
    path = os.path.join(OUT_DIR, "Table05_step3_refined_program_landscape.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_state_label_map():
    mapping = {}
    for row in read_csv_rows(os.path.join(BASE, "data", "processed", "step3", "05_states", "step3_state_marker_summary.csv")):
        mapping[row["state_id"]] = row["state_label"]
    return mapping


def build_table6(program_names):
    state_labels = build_state_label_map()
    state_support = read_csv_rows(os.path.join(BASE, "data", "processed", "step3", "05_states", "step3_candidate_state_support.csv"))
    trajectory = read_csv_rows(os.path.join(BASE, "data", "processed", "step3", "03_donor_patterns", "step3_trajectory_summary.csv"))
    registry_rows = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "step4_validation_registry.csv"))
    holdout = {row["program_id"]: row for row in read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "holdout_annex.csv"))}
    holdout_by_source = defaultdict(list)
    for row in read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "holdout_annex_by_source.csv")):
        holdout_by_source[row["program_id"]].append(row)

    program_support_rows = defaultdict(list)
    pair_support_rows = defaultdict(list)
    for row in state_support:
        if row["candidate_level"] == "program":
            program_support_rows[row["candidate_id"]].append(row)
        elif row["candidate_level"] == "pair":
            pair_support_rows[row["candidate_label"]].append(row)

    trajectory_map = {}
    for row in trajectory:
        if row["candidate_level"] == "program" and is_nonempty(row.get("refined_program_id")):
            trajectory_map[row["refined_program_id"]] = row

    rows = []
    for registry in registry_rows:
        program_id = registry["program_id"]
        program_name = registry["program_name"]
        sender_state = registry["sender_state_id"]
        receiver_state = registry["receiver_state_id"]
        support_note = ""

        support_rows = []
        if program_id == "apoe_lipoprotein_receptor_signaling":
            support_rows = sorted(program_support_rows.get("program::apoe_lipoprotein_receptor_signaling", []), key=lambda row: parse_int(row["state_pair_rank"]) or 999)
        else:
            anchors = [item["pair_key"] for item in parse_json_field(registry["anchor_pairs_json"])]
            exact_rows = []
            fallback_rows = []
            for anchor in anchors:
                for row in pair_support_rows.get(anchor, []):
                    fallback_rows.append(row)
                    if row["sender_state_id"] == sender_state and row["receiver_state_id"] == receiver_state:
                        exact_rows.append(row)
            support_rows = exact_rows if exact_rows else sorted(fallback_rows, key=lambda row: parse_float(row["high_contribution_edge_enrichment"]) or -1, reverse=True)[:1]
            if exact_rows:
                support_note = "state_support_from_split_anchor_pairs"
            else:
                support_note = "fallback_to_best_available_anchor_pair_state_support"

        ligand_support = mean_or_na([parse_float(row["ligand_expression_support"]) for row in support_rows])
        receptor_support = mean_or_na([parse_float(row["receptor_expression_support"]) for row in support_rows])
        enrichment = mean_or_na([parse_float(row["high_contribution_edge_enrichment"]) for row in support_rows])

        traj_key = program_id
        traj_note = "program_specific_trajectory"
        if program_id in ("slit_robo_guidance_canonical", "slit_robo_guidance_context"):
            traj_key = "slit_robo_guidance"
            traj_note = "trajectory_inherited_from_parent_slit_robo_guidance"
        traj = trajectory_map.get(traj_key, {})

        by_source = holdout_by_source.get(program_id, [])
        source_concordance = []
        for row in sorted(by_source, key=lambda item: item["holdout_source_id"]):
            source_concordance.append("%s=%s" % (row["holdout_source_id"], row["direction_match"]))

        rows.append(
            {
                "program_id": program_id,
                "program_name": program_name,
                "dominant_sender_state_id": sender_state,
                "dominant_sender_state_label": state_labels.get(sender_state, NA),
                "dominant_receiver_state_id": receiver_state,
                "dominant_receiver_state_label": state_labels.get(receiver_state, NA),
                "ligand_expression_support": ligand_support,
                "receptor_expression_support": receptor_support,
                "high_contribution_state_pair_enrichment": enrichment,
                "control_age_beta": traj.get("control_age_beta", NA),
                "pd_shift_size": traj.get("pd_shift_size", NA),
                "pd_age_interaction_beta": traj.get("pd_age_interaction_beta", NA),
                "lbd_bridge_class": traj.get("lbd_bridge_class", NA),
                "trajectory_class": traj.get("trajectory_class", NA),
                "holdout_direction_concordance": holdout.get(program_id, {}).get("holdout_direction_match", NA),
                "holdout_source_concordance": "|".join(source_concordance) if source_concordance else NA,
                "trajectory_source": traj_note,
                "state_support_note": support_note or "program_level_state_support",
            }
        )
        program_names.setdefault(program_id, program_name)

    fieldnames = [
        "program_id",
        "program_name",
        "dominant_sender_state_id",
        "dominant_sender_state_label",
        "dominant_receiver_state_id",
        "dominant_receiver_state_label",
        "ligand_expression_support",
        "receptor_expression_support",
        "high_contribution_state_pair_enrichment",
        "control_age_beta",
        "pd_shift_size",
        "pd_age_interaction_beta",
        "lbd_bridge_class",
        "trajectory_class",
        "holdout_direction_concordance",
        "holdout_source_concordance",
        "trajectory_source",
        "state_support_note",
    ]
    path = os.path.join(OUT_DIR, "Table06_state_mapping_and_donor_trajectory_summary.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_table7():
    rows = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "program_primary_spatial_results.csv"))
    fieldnames = [
        "program_id",
        "program_name",
        "expected_pd_direction",
        "n_pd_donors",
        "n_control_donors",
        "snpc_adjacency_beta",
        "snpc_adjacency_p_value",
        "snpc_adjacency_fdr",
        "contrast_beta",
        "contrast_p_value",
        "contrast_fdr",
        "same_spot_beta",
        "same_spot_p_value",
        "aligned_metric_count",
        "directional_p_snpc",
        "directional_p_contrast",
        "directional_p_same_spot",
        "composite_directional_p_value",
        "direction_match",
        "primary_spatial_support_bin",
    ]
    path = os.path.join(OUT_DIR, "Table07_primary_spatial_donor_level_results.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_table8():
    manifest = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "primary_visium_manifest.csv"))
    qc_tables = parse_markdown_tables(os.path.join(PKG, "04_KEY_TABLES", "primary_visium_qc_report.md"))
    qc_rows = []
    for table in qc_tables:
        if table and "sample_accession" in table[0]:
            qc_rows = table
            break

    qc_by_sample = defaultdict(list)
    for row in qc_rows:
        qc_by_sample[row["sample_accession"]].append(row)

    rows = []
    thresholds = "total_counts>=200|n_genes_by_counts>=150"
    rule = "SNpc=donor-specific dopaminergic score >= 65th percentile; remaining kept spots=surrounding_tissue"

    for row in manifest:
        disease = row["disease_group"]
        compartment_rows = [
            item
            for item in qc_by_sample.get(row["sample_accession"], [])
            if item.get("disease_group") == disease and parse_int(item.get("n_spots")) not in (None, 0)
        ]
        total_spots = sum(parse_int(item["n_spots"]) or 0 for item in compartment_rows)

        def weighted(metric):
            values = []
            weights = []
            for item in compartment_rows:
                value = parse_float(item.get(metric))
                weight = parse_int(item.get("n_spots"))
                if value is not None and weight:
                    values.append(value * weight)
                    weights.append(weight)
            if not weights:
                return NA
            return sum(values) / float(sum(weights))

        snpc_spots = 0
        surrounding_spots = 0
        for item in compartment_rows:
            if item["anatomical_compartment"] == "SNpc":
                snpc_spots = parse_int(item["n_spots"]) or 0
            elif item["anatomical_compartment"] == "surrounding_tissue":
                surrounding_spots = parse_int(item["n_spots"]) or 0

        rows.append(
            {
                "section_accession": row["sample_accession"],
                "donor_id": row["donor_id"],
                "disease_group": row["disease_group"],
                "sex": row["sex"],
                "age_text": row["age"],
                "age_years": row["age_years"] or NA,
                "raw_spot_count": row["raw_spot_count"],
                "kept_spot_count": row["kept_spot_count"],
                "reconstructed_snpc_spot_count": snpc_spots,
                "reconstructed_surrounding_tissue_spot_count": surrounding_spots,
                "mean_total_counts_kept": weighted("mean_total_counts"),
                "mean_n_genes_kept": weighted("mean_n_genes"),
                "mean_pct_mito_kept": weighted("mean_pct_mito"),
                "mean_dopaminergic_score_kept": weighted("mean_da_score"),
                "qc_thresholds": thresholds,
                "compartment_definition_rule": rule,
            }
        )

    fieldnames = [
        "section_accession",
        "donor_id",
        "disease_group",
        "sex",
        "age_text",
        "age_years",
        "raw_spot_count",
        "kept_spot_count",
        "reconstructed_snpc_spot_count",
        "reconstructed_surrounding_tissue_spot_count",
        "mean_total_counts_kept",
        "mean_n_genes_kept",
        "mean_pct_mito_kept",
        "mean_dopaminergic_score_kept",
        "qc_thresholds",
        "compartment_definition_rule",
    ]
    path = os.path.join(OUT_DIR, "Table08_primary_visium_section_qc_summary.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_table9():
    summary = {row["program_id"]: row for row in read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "secondary_spatial_support_summary.csv"))}
    roi_rows = read_csv_rows(os.path.join(BASE, "data", "processed", "step4", "03_spatial", "05_secondary_replication", "geomx_roi_program_scores.csv"))

    grouped = defaultdict(list)
    for row in roi_rows:
        grouped[row["program_id"]].append(row)

    rows = []
    for program_id, rows_for_program in grouped.items():
        total_rows = [row for row in rows_for_program if row["sheet_name"].lower() == "total"]
        quadrant_rows = [row for row in rows_for_program if row["sheet_name"].lower() != "total"]
        total_row = total_rows[0] if total_rows else {}
        summary_row = summary.get(program_id, {})
        rows.append(
            {
                "program_id": program_id,
                "program_name": summary_row.get("program_name", rows_for_program[0]["program_name"]),
                "scoring_gene_source": summary_row.get("scoring_gene_source", total_row.get("scoring_gene_source", NA)),
                "total_program_score": summary_row.get("total_program_score", total_row.get("program_score", NA)),
                "total_program_n_genes": summary_row.get("total_program_n_genes", total_row.get("n_genes_present", NA)),
                "mean_quadrant_score": summary_row.get("mean_quadrant_score", mean_or_na([parse_float(row["program_score"]) for row in quadrant_rows])),
                "n_quadrants_direction_consistent": summary_row.get(
                    "n_quadrants_direction_consistent",
                    sum(1 for row in quadrant_rows if row.get("direction_match") == "True"),
                ),
                "secondary_spatial_support_bin": summary_row.get("secondary_spatial_support_bin", NA),
            }
        )

    fieldnames = [
        "program_id",
        "program_name",
        "scoring_gene_source",
        "total_program_score",
        "total_program_n_genes",
        "mean_quadrant_score",
        "n_quadrants_direction_consistent",
        "secondary_spatial_support_bin",
    ]
    path = os.path.join(OUT_DIR, "Table09_secondary_spatial_replication_summary.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_table10():
    annex = {row["program_id"]: row for row in read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "holdout_annex.csv"))}
    by_source = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "holdout_annex_by_source.csv"))
    rows = []
    for row in by_source:
        overall = annex[row["program_id"]]
        rows.append(
            {
                "program_id": row["program_id"],
                "program_name": row["program_name"],
                "holdout_source_id": row["holdout_source_id"],
                "program_weighted_holdout_pd_minus_control": overall["mean_holdout_pd_minus_control"],
                "source_mean_holdout_pd_minus_control": row["mean_holdout_pd_minus_control"],
                "expected_internal_direction": overall["expected_internal_direction"],
                "holdout_support_tier": overall["holdout_support_tier"],
                "holdout_weighted_match_fraction": overall["holdout_weighted_match_fraction"],
                "n_holdout_sources_available": overall["n_holdout_sources_available"],
                "n_holdout_sources_direction_match": overall["n_holdout_sources_direction_match"],
                "source_direction_match": row["direction_match"],
            }
        )

    fieldnames = [
        "program_id",
        "program_name",
        "holdout_source_id",
        "program_weighted_holdout_pd_minus_control",
        "source_mean_holdout_pd_minus_control",
        "expected_internal_direction",
        "holdout_support_tier",
        "holdout_weighted_match_fraction",
        "n_holdout_sources_available",
        "n_holdout_sources_direction_match",
        "source_direction_match",
    ]
    path = os.path.join(OUT_DIR, "Table10_internal_holdout_robustness_summary.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_table11(program_names):
    gene_rows = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "gene_level_support_summary.csv"))
    program_rows = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "program_genetic_evidence_matrix.csv"))
    for row in program_rows:
        program_names.setdefault(row["program_id"], row["program_name"])

    rows = []
    for row in gene_rows:
        memberships = parse_json_field(row["program_ids_json"])
        if not memberships:
            memberships = [NA]
        for program_id in memberships:
            rows.append(
                {
                    "row_level": "gene",
                    "program_id": program_id,
                    "program_name": program_names.get(program_id, NA),
                    "gene_symbol": row["gene_symbol"],
                    "instrumentability_status": row["instrumentability_status"],
                    "best_tissue": row["best_tissue"],
                    "best_snp_id": row["best_snp_id"],
                    "best_primary_mr_method": row["best_primary_mr_method"],
                    "finngen_mr_beta": row["best_primary_mr_beta"],
                    "finngen_mr_p_value": row["best_primary_mr_p_value"],
                    "finngen_mr_fdr": row["best_primary_mr_p_adj_fdr"],
                    "gcst90275127_mr_beta": row["best_secondary_mr_beta"],
                    "gcst90275127_mr_p_value": row["best_secondary_mr_p_value"],
                    "gcst90275127_mr_fdr": row["best_secondary_mr_p_adj_fdr"],
                    "mean_F": row["best_primary_mean_F"],
                    "coloc_h4_proxy": row["best_primary_coloc_h4_proxy"],
                    "cross_outcome_direction_consistent": row["cross_outcome_direction_consistent"],
                    "gene_support_bin": row["gene_support_bin"],
                    "n_genes": NA,
                    "n_strong_gene_support": NA,
                    "n_moderate_gene_support": NA,
                    "magma_like_p_value": NA,
                    "magma_like_fdr": NA,
                    "empirical_p_value": NA,
                    "empirical_fdr": NA,
                    "locus_overlap_count": NA,
                    "program_support_bin": NA,
                }
            )

    for row in program_rows:
        rows.append(
            {
                "row_level": "program",
                "program_id": row["program_id"],
                "program_name": row["program_name"],
                "gene_symbol": NA,
                "instrumentability_status": NA,
                "best_tissue": NA,
                "best_snp_id": NA,
                "best_primary_mr_method": NA,
                "finngen_mr_beta": NA,
                "finngen_mr_p_value": NA,
                "finngen_mr_fdr": NA,
                "gcst90275127_mr_beta": NA,
                "gcst90275127_mr_p_value": NA,
                "gcst90275127_mr_fdr": NA,
                "mean_F": NA,
                "coloc_h4_proxy": NA,
                "cross_outcome_direction_consistent": NA,
                "gene_support_bin": NA,
                "n_genes": row["n_genes"],
                "n_strong_gene_support": row["n_strong_gene_support"],
                "n_moderate_gene_support": row["n_moderate_gene_support"],
                "magma_like_p_value": row["magma_like_p_value"],
                "magma_like_fdr": row["magma_like_fdr"],
                "empirical_p_value": row["empirical_p_value"],
                "empirical_fdr": row["empirical_fdr"],
                "locus_overlap_count": row["locus_overlap_count"],
                "program_support_bin": row["program_genetic_support_bin"],
            }
        )

    fieldnames = [
        "row_level",
        "program_id",
        "program_name",
        "gene_symbol",
        "instrumentability_status",
        "best_tissue",
        "best_snp_id",
        "best_primary_mr_method",
        "finngen_mr_beta",
        "finngen_mr_p_value",
        "finngen_mr_fdr",
        "gcst90275127_mr_beta",
        "gcst90275127_mr_p_value",
        "gcst90275127_mr_fdr",
        "mean_F",
        "coloc_h4_proxy",
        "cross_outcome_direction_consistent",
        "gene_support_bin",
        "n_genes",
        "n_strong_gene_support",
        "n_moderate_gene_support",
        "magma_like_p_value",
        "magma_like_fdr",
        "empirical_p_value",
        "empirical_fdr",
        "locus_overlap_count",
        "program_support_bin",
    ]
    path = os.path.join(OUT_DIR, "Table11_genetic_support_summary.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_table12():
    final_rows = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "step4_final_evidence_matrix.csv"))
    rows = []
    for row in final_rows:
        anchor_pairs = [item["pair_key"] for item in parse_json_field(row["anchor_pairs_json"])]
        gene_members = parse_json_field(row["gene_members_json"])
        caveat_flags = parse_json_field(row["caveat_flags_json"])
        rows.append(
            {
                "program_id": row["program_id"],
                "program_name": row["program_name"],
                "program_batch": row["program_batch"],
                "anchor_pairs": "|".join(anchor_pairs) if anchor_pairs else NA,
                "gene_members": "|".join(gene_members) if gene_members else NA,
                "sender_state_id": row["sender_state_id"],
                "receiver_state_id": row["receiver_state_id"],
                "expected_pd_direction": row["expected_pd_direction"],
                "caveat_flags": "|".join(caveat_flags) if caveat_flags else NA,
                "primary_spatial_support_bin": row["primary_spatial_support_bin"],
                "snpc_adjacency_beta": row["snpc_adjacency_beta"],
                "snpc_adjacency_p_value": row["snpc_adjacency_p_value"],
                "snpc_adjacency_fdr": row["snpc_adjacency_fdr"],
                "contrast_beta": row["contrast_beta"],
                "contrast_p_value": row["contrast_p_value"],
                "contrast_fdr": row["contrast_fdr"],
                "secondary_spatial_support_bin": row["secondary_spatial_support_bin"],
                "total_program_score": row["total_program_score"],
                "mean_quadrant_score": row["mean_quadrant_score"],
                "n_quadrants_direction_consistent": row["n_quadrants_direction_consistent"],
                "program_genetic_support_bin": row["program_genetic_support_bin"],
                "n_strong_gene_support": row["n_strong_gene_support"],
                "n_moderate_gene_support": row["n_moderate_gene_support"],
                "empirical_p_value": row["empirical_p_value"],
                "empirical_fdr": row["empirical_fdr"],
                "mean_holdout_pd_minus_control": row["mean_holdout_pd_minus_control"],
                "holdout_direction_match": row["holdout_direction_match"],
                "holdout_support_tier": row["holdout_support_tier"],
                "n_holdout_sources_available": row["n_holdout_sources_available"],
                "n_holdout_sources_direction_match": row["n_holdout_sources_direction_match"],
                "source_dominant_flag": row["source_dominant_flag"],
                "negative_holdout_flag": row["negative_holdout_flag"],
                "source_flip_flag": row["source_flip_flag"],
                "region_flip_flag": row["region_flip_flag"],
                "cell_count_link_flag": row["cell_count_link_flag"],
                "primary_spatial_component": row["primary_spatial_component"],
                "secondary_spatial_component": row["secondary_spatial_component"],
                "program_level_genetic_component": row["program_level_genetic_component"],
                "cross_dataset_consistency_component": row["cross_dataset_consistency_component"],
                "total_penalty": row["total_penalty"],
                "final_step4_score": row["final_step4_score"],
                "spatial_primary_gate": row["spatial_primary_gate"],
                "spatial_replication_gate": row["spatial_replication_gate"],
                "genetic_gene_level_gate": row["genetic_gene_level_gate"],
                "genetic_program_level_gate": row["genetic_program_level_gate"],
                "genetic_consensus_gate": row["genetic_consensus_gate"],
                "spatial_overall_gate": row["spatial_overall_gate"],
                "internal_stability_gate": row["internal_stability_gate"],
                "claim_tier": row["claim_tier"],
                "promotion_decision": row["promotion_decision"],
                "final_evidence_class": row["final_evidence_class"],
                "decision_rationale": row["decision_rationale"],
            }
        )

    fieldnames = [
        "program_id",
        "program_name",
        "program_batch",
        "anchor_pairs",
        "gene_members",
        "sender_state_id",
        "receiver_state_id",
        "expected_pd_direction",
        "caveat_flags",
        "primary_spatial_support_bin",
        "snpc_adjacency_beta",
        "snpc_adjacency_p_value",
        "snpc_adjacency_fdr",
        "contrast_beta",
        "contrast_p_value",
        "contrast_fdr",
        "secondary_spatial_support_bin",
        "total_program_score",
        "mean_quadrant_score",
        "n_quadrants_direction_consistent",
        "program_genetic_support_bin",
        "n_strong_gene_support",
        "n_moderate_gene_support",
        "empirical_p_value",
        "empirical_fdr",
        "mean_holdout_pd_minus_control",
        "holdout_direction_match",
        "holdout_support_tier",
        "n_holdout_sources_available",
        "n_holdout_sources_direction_match",
        "source_dominant_flag",
        "negative_holdout_flag",
        "source_flip_flag",
        "region_flip_flag",
        "cell_count_link_flag",
        "primary_spatial_component",
        "secondary_spatial_component",
        "program_level_genetic_component",
        "cross_dataset_consistency_component",
        "total_penalty",
        "final_step4_score",
        "spatial_primary_gate",
        "spatial_replication_gate",
        "genetic_gene_level_gate",
        "genetic_program_level_gate",
        "genetic_consensus_gate",
        "spatial_overall_gate",
        "internal_stability_gate",
        "claim_tier",
        "promotion_decision",
        "final_evidence_class",
        "decision_rationale",
    ]
    path = os.path.join(OUT_DIR, "Table12_integrated_step4_evidence_matrix.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_table13():
    final_rows = {row["program_id"]: row for row in read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "step4_final_evidence_matrix.csv"))}
    decision_log = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "step4_candidate_decision_log.csv"))
    rows = []
    for row in decision_log:
        details = final_rows[row["program_id"]]
        rows.append(
            {
                "program_id": row["program_id"],
                "program_name": row["program_name"],
                "spatial_primary_gate": row["spatial_primary_gate"],
                "spatial_replication_gate": row["spatial_replication_gate"],
                "genetic_gene_level_gate": row["genetic_gene_level_gate"],
                "genetic_program_level_gate": row["genetic_program_level_gate"],
                "internal_stability_gate": row["internal_stability_gate"],
                "claim_tier": row["claim_tier"],
                "promotion_decision": row["promotion_decision"],
                "final_step4_score": row["final_step4_score"],
                "final_evidence_class": row["final_evidence_class"],
                "source_dominant_flag": details["source_dominant_flag"],
                "source_flip_flag": details["source_flip_flag"],
                "region_flip_flag": details["region_flip_flag"],
                "negative_holdout_flag": details["negative_holdout_flag"],
                "cell_count_link_flag": details["cell_count_link_flag"],
                "blocking_issue_flag": details["blocking_issue_flag"],
                "rerun_or_holdback_decision": row["promotion_decision"],
                "failure_rationale": row["decision_rationale"],
            }
        )

    fieldnames = [
        "program_id",
        "program_name",
        "spatial_primary_gate",
        "spatial_replication_gate",
        "genetic_gene_level_gate",
        "genetic_program_level_gate",
        "internal_stability_gate",
        "claim_tier",
        "promotion_decision",
        "final_step4_score",
        "final_evidence_class",
        "source_dominant_flag",
        "source_flip_flag",
        "region_flip_flag",
        "negative_holdout_flag",
        "cell_count_link_flag",
        "blocking_issue_flag",
        "rerun_or_holdback_decision",
        "failure_rationale",
    ]
    path = os.path.join(OUT_DIR, "Table13_candidate_decision_log.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def build_table14():
    top10 = read_csv_rows(os.path.join(PKG, "04_KEY_TABLES", "step3_top10_refined_programs.csv"))
    confounds = {}
    for row in read_csv_rows(os.path.join(BASE, "data", "processed", "step3", "07_confounds", "step3_confound_summary.csv")):
        if row["candidate_level"] == "program":
            confounds[row["candidate_id"]] = row
    task_support = {}
    for row in read_csv_rows(os.path.join(BASE, "data", "processed", "step3", "04_task_support", "step3_task_support_matrix.csv")):
        if row["candidate_level"] == "program":
            task_support[row["candidate_id"]] = row
    trajectory = {}
    for row in read_csv_rows(os.path.join(BASE, "data", "processed", "step3", "03_donor_patterns", "step3_trajectory_summary.csv")):
        if row["candidate_level"] == "program":
            trajectory[row["candidate_id"]] = row
    label_sensitivity_rows = read_csv_rows(os.path.join(BASE, "data", "processed", "step2", "07_robustness", "step2_label_confidence_sensitivity.csv"))
    state_tables = parse_markdown_tables(os.path.join(PKG, "04_KEY_TABLES", "state_mapping_sensitivity_report.md"))
    state_map_rows = {}
    for row in state_tables[0] if state_tables else []:
        state_map_rows[row["state_id"]] = row
    cell_count_rows = read_csv_rows(os.path.join(BASE, "data", "processed", "step2", "07_robustness", "step2_cell_count_robustness.csv"))

    sensitivity_by_task = defaultdict(lambda: defaultdict(int))
    for row in label_sensitivity_rows:
        sensitivity_by_task[row["task_name"]][row["confidence_threshold"]] += parse_int(row["n_graph_units"]) or 0

    cell_count_summary = []
    for row in cell_count_rows:
        if row["analysis"] == "pd_model":
            cell_count_summary.append("%s=%s" % (row["metric"], row["spearman_rho"]))
    cell_count_text = "|".join(cell_count_summary) if cell_count_summary else NA

    rows = []
    for row in top10:
        candidate_id = "program::%s" % row["refined_program_id"]
        confound = confounds.get(candidate_id, {})
        support = task_support.get(candidate_id, {})
        traj = trajectory.get(candidate_id, {})
        task = row["dominant_task"]
        threshold_parts = []
        for threshold in sorted(sensitivity_by_task.get(task, {}), key=lambda value: float(value)):
            threshold_parts.append("%s=%s" % (threshold, sensitivity_by_task[task][threshold]))
        rows.append(
            {
                "refined_program_id": row["refined_program_id"],
                "refined_program_name": row["refined_program_name"],
                "dominant_task": task,
                "label_confidence_sensitivity_counts": "|".join(threshold_parts) if threshold_parts else NA,
                "cell_count_robustness_correlations": cell_count_text,
                "task_specificity_score": support.get("task_specificity_score", NA),
                "receiver_specificity_task": support.get("receiver_specificity_task", NA),
                "task_consistency_status": support.get("task_consistency_status", NA),
                "sender_state_mapping_sensitivity": state_map_rows.get(row["dominant_sender_state"], {}).get("primary_sensitivity_correlation", NA),
                "receiver_state_mapping_sensitivity": state_map_rows.get(row["dominant_receiver_state"], {}).get("primary_sensitivity_correlation", NA),
                "source_majority_share": confound.get("source_majority_share", NA),
                "source_direction_flip": confound.get("source_direction_flip", NA),
                "region_direction_flip": confound.get("region_direction_flip", NA),
                "sex_sensitivity_status": confound.get("sex_sensitivity_status", NA),
                "holdout_direction_concordance": traj.get("holdout_direction_concordance", NA),
                "holdout_support_metric": confound.get("holdout_support", NA),
                "promotion_status": confound.get("promotion_status", NA),
                "caveat_summary": confound.get("caveat_summary", row.get("caveat_summary", NA)),
            }
        )

    fieldnames = [
        "refined_program_id",
        "refined_program_name",
        "dominant_task",
        "label_confidence_sensitivity_counts",
        "cell_count_robustness_correlations",
        "task_specificity_score",
        "receiver_specificity_task",
        "task_consistency_status",
        "sender_state_mapping_sensitivity",
        "receiver_state_mapping_sensitivity",
        "source_majority_share",
        "source_direction_flip",
        "region_direction_flip",
        "sex_sensitivity_status",
        "holdout_direction_concordance",
        "holdout_support_metric",
        "promotion_status",
        "caveat_summary",
    ]
    path = os.path.join(OUT_DIR, "Table14_sensitivity_and_confound_summary.csv")
    write_csv(path, rows, fieldnames)
    return path, len(rows)


def write_index(index_rows):
    fieldnames = ["table_id", "output_file", "row_count"]
    path = os.path.join(OUT_DIR, "table_output_index.csv")
    write_csv(path, index_rows, fieldnames)


def main():
    ensure_dir(OUT_DIR)
    program_names = {}
    index_rows = []

    builders = [
        ("Table01", build_table1),
        ("Table02", build_table2),
        ("Table03", build_table3),
        ("Table04", build_table4),
        ("Table05", lambda: build_table5(program_names)),
        ("Table06", lambda: build_table6(program_names)),
        ("Table07", build_table7),
        ("Table08", build_table8),
        ("Table09", build_table9),
        ("Table10", build_table10),
        ("Table11", lambda: build_table11(program_names)),
        ("Table12", build_table12),
        ("Table13", build_table13),
        ("Table14", build_table14),
    ]

    for table_id, builder in builders:
        path, row_count = builder()
        index_rows.append({"table_id": table_id, "output_file": os.path.basename(path), "row_count": row_count})

    write_index(index_rows)


if __name__ == "__main__":
    main()
