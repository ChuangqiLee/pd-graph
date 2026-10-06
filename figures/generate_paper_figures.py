import argparse
from functools import lru_cache
import json
from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import pubfig as pf
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

FIG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = FIG_DIR.parent
PACKAGE_DIR = PROJECT_DIR / 'MANUSCRIPT_WRITING_PACKAGE_20260402'
PACKAGE_TABLE_DIR = PACKAGE_DIR / '04_KEY_TABLES'
RENDERED_DIR = FIG_DIR / 'rendered'
STEP2_DIR = PROJECT_DIR / 'data' / 'processed' / 'step2'
STEP3_DIR = PROJECT_DIR / 'data' / 'processed' / 'step3'
STEP4_DIR = PROJECT_DIR / 'data' / 'processed' / 'step4'
PAPER_PALETTE = ['#90A4C4', '#ADC9E4', '#C0D8E8', '#DBDDEF', '#EBF0FC', '#FCEDE9', '#F7DAD4', '#F5C9B6', '#F3B8AC', '#F29F95']
PAPER_COOL_DARK = PAPER_PALETTE[0]
PAPER_COOL_MID = PAPER_PALETTE[1]
PAPER_COOL_LIGHT = PAPER_PALETTE[2]
PAPER_COOL_FADE = PAPER_PALETTE[3]
PAPER_COOL_PALE = PAPER_PALETTE[4]
PAPER_WARM_PALE = PAPER_PALETTE[5]
PAPER_WARM_LIGHT = PAPER_PALETTE[6]
PAPER_WARM_SOFT = PAPER_PALETTE[7]
PAPER_WARM_MID = PAPER_PALETTE[8]
PAPER_WARM_DARK = PAPER_PALETTE[9]
PAPER_BLUE_SOFT = PAPER_COOL_MID
PAPER_BLUE_LIGHT = PAPER_COOL_LIGHT
PAPER_TEAL_LIGHT = PAPER_COOL_FADE
PAPER_RED = PAPER_WARM_DARK
PAPER_BLUE = PAPER_COOL_DARK
PAPER_TEAL = PAPER_WARM_MID
PAPER_LINE_PALETTE = [PAPER_COOL_DARK, PAPER_COOL_MID, PAPER_COOL_LIGHT, PAPER_WARM_SOFT, PAPER_WARM_MID, PAPER_WARM_DARK]
FIGURE2_FOLD_PALETTE = [PAPER_BLUE, PAPER_RED, PAPER_WARM_SOFT, PAPER_COOL_MID, PAPER_WARM_LIGHT, PAPER_COOL_LIGHT]
FIGURE2_CALIBRATION_COLORS = {'raw': PAPER_WARM_SOFT, 'calibrated': PAPER_BLUE}
FIGURE2_PARAMETER_STYLES = {
    'decision_threshold': {'color': PAPER_BLUE, 'marker': 'o', 'label': 'Threshold'},
    'calibration_intercept': {'color': PAPER_RED, 'marker': 's', 'label': 'Intercept'},
    'calibration_slope': {'color': PAPER_COOL_MID, 'marker': 'D', 'label': 'Slope'},
}
FIGURE2_BASELINE_COLORS = {'Logistic': PAPER_COOL_MID, 'Random forest': PAPER_WARM_SOFT, 'Graph model': PAPER_BLUE}
FIGURE2_DISTRIBUTION_COLORS = {'Discovery control': PAPER_BLUE, 'LBD bridge': PAPER_WARM_SOFT, 'Discovery PD': PAPER_RED}
FIGURE3_PATHWAY_PALETTE = [
    PAPER_WARM_DARK,
    PAPER_BLUE,
    PAPER_WARM_SOFT,
    PAPER_COOL_MID,
    PAPER_WARM_LIGHT,
    PAPER_COOL_LIGHT,
    PAPER_WARM_PALE,
    PAPER_COOL_FADE,
    PAPER_BLUE_SOFT,
    PAPER_TEAL_LIGHT,
]
FIGURE3_PAIR_HIGHLIGHT_COLORS = {
    'APOE->APP': PAPER_WARM_DARK,
    'APOE->LRP1': PAPER_BLUE,
    'SLIT2->ROBO1': PAPER_WARM_SOFT,
    'SLIT2->ROBO2': PAPER_COOL_MID,
}
FIGURE3_BUBBLE_RANK_COLORS = {1: PAPER_WARM_DARK, 2: PAPER_BLUE, 3: PAPER_WARM_SOFT}
FIGURE5_PROGRAM_ID = 'apoe_lipoprotein_receptor_signaling'
FIGURE5_REPRESENTATIVE_SAMPLES = {'control': 'GSM8031366', 'pd': 'GSM8031361'}
FIGURE5_GROUP_ORDER = ['control', 'pd']
FIGURE5_GROUP_LABELS = {'control': 'Control', 'pd': 'PD'}
FIGURE5_GROUP_COLORS = {'control': PAPER_BLUE, 'pd': PAPER_RED}
FIGURE5_SOURCE_COLORS = {'GSE157783': PAPER_WARM_SOFT, 'GSE253462': PAPER_RED}
FIGURE5_MATCH_COLOR = PAPER_TEAL
FIGURE5_HOTSPOT_COLOR = PAPER_BLUE_LIGHT
FIGURE6_PROGRAM_IDS = {
    'canonical': 'slit_robo_guidance_canonical',
    'context': 'slit_robo_guidance_context',
}
FIGURE6_PALETTE = [PAPER_COOL_DARK, PAPER_COOL_MID, PAPER_COOL_LIGHT, PAPER_COOL_FADE, PAPER_WARM_SOFT, PAPER_WARM_DARK]
FIGURE6_BRANCH_COLORS = {'canonical': PAPER_BLUE, 'context': PAPER_WARM_SOFT}
FIGURE6_DISEASE_COLORS = {'control': PAPER_BLUE_LIGHT, 'pd': PAPER_RED}
FIGURE6_COMPONENT_COLORS = {
    'primary': PAPER_BLUE,
    'geomx': PAPER_WARM_SOFT,
    'genetics': PAPER_RED,
    'holdout': PAPER_BLUE_SOFT,
    'final': PAPER_TEAL_LIGHT,
    'highlight': PAPER_BLUE_LIGHT,
}
FIGURE7_PROGRAM_ORDER = [
    'apoe_lipoprotein_receptor_signaling',
    'slit_robo_guidance_canonical',
    'slit_robo_guidance_context',
]
FIGURE7_PROGRAM_COLORS = {
    'apoe_lipoprotein_receptor_signaling': PAPER_WARM_DARK,
    'slit_robo_guidance_canonical': PAPER_BLUE,
    'slit_robo_guidance_context': PAPER_WARM_SOFT,
}
FIGURE7_SUPPORT_COLOR_SCALE = ['#FFFFFF', PAPER_COOL_PALE, PAPER_COOL_MID, PAPER_WARM_LIGHT, PAPER_WARM_DARK]
DISCOVERY_ATLAS_SOURCES = [
    'b0f0b447-ac37-45b0-b1bf-5c0b7d871120',
    'GSE193688',
    'GSE140231',
    'GSE126836',
]
AUXILIARY_SOURCES = ['GSE148434']
HOLDOUT_SOURCES = ['GSE157783', 'GSE253462']
SOURCE_LABELS = {
    'b0f0b447-ac37-45b0-b1bf-5c0b7d871120': 'CELLxGENE SNpc',
    'GSE193688': 'GSE193688',
    'GSE140231': 'GSE140231',
    'GSE126836': 'GSE126836',
    'GSE148434': 'GSE148434',
    'GSE157783': 'GSE157783',
    'GSE253462': 'GSE253462',
    'GSE253975': 'GSE253975',
    'GeoMx': 'GeoMx SNpc',
    'GTEx': 'GTEx v8',
    'FinnGen': 'FinnGen R10',
    'GCST90275127': 'GCST90275127',
}
SOURCE_COLORS = {
    'b0f0b447-ac37-45b0-b1bf-5c0b7d871120': PAPER_BLUE,
    'GSE193688': PAPER_BLUE_SOFT,
    'GSE140231': PAPER_BLUE_LIGHT,
    'GSE126836': PAPER_TEAL_LIGHT,
    'GSE148434': PAPER_WARM_LIGHT,
    'GSE157783': PAPER_WARM_SOFT,
    'GSE253462': PAPER_RED,
}
COHORT_COLORS = {
    'control_known_age': PAPER_BLUE,
    'pd_known_age': PAPER_RED,
    'lbd_bridge': PAPER_BLUE_SOFT,
    'unknown_age_auxiliary': PAPER_WARM_LIGHT,
    'control': PAPER_BLUE,
    'pd': PAPER_RED,
}
INTERPRETATION_COLORS = {
    'pd_amplified_aging': PAPER_RED,
    'pd_specific': PAPER_BLUE,
    'normal_aging_associated': PAPER_WARM_SOFT,
    'unstable_or_source_sensitive': PAPER_TEAL_LIGHT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=str(RENDERED_DIR))
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    pf.set_default_theme('nature')
    mpl.rcParams.update(
        {
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'font.size': 6,
            'axes.titlesize': 7,
            'axes.labelsize': 6,
            'xtick.labelsize': 5,
            'ytick.labelsize': 5,
        }
    )


def export_figure(fig: plt.Figure, stem: str, out_dir: Path, width_mm: float, height_mm: float) -> list[Path]:
    ensure_dir(out_dir)
    fig.set_size_inches(width_mm / 25.4, height_mm / 25.4, forward=True)
    paths = []
    for ext in ('png', 'pdf', 'svg'):
        target = out_dir / f'{stem}.{ext}'
        fig.savefig(target, dpi=600, bbox_inches=None)
        paths.append(target)
    plt.close(fig)
    return paths


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight='bold', ha='left', va='bottom', clip_on=False)


def set_left_title(ax: plt.Axes, title: str, pad: float = 4.5) -> None:
    ax.set_title(title, loc='left', pad=pad, fontsize=7)


def figure5_panel_label(ax: plt.Axes, label: str, *, x: float = -0.10, y: float = 1.07) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9.5, fontweight='bold', ha='left', va='bottom', clip_on=False)


def floating_title(ax: plt.Axes, title: str, subtitle: str | None = None, *, y: float = 1.13, subtitle_y: float = 1.08) -> None:
    ax.text(0.0, y, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=7, color='#2B2F33', clip_on=False)
    if subtitle:
        ax.text(0.98, subtitle_y, subtitle, transform=ax.transAxes, ha='right', va='bottom', fontsize=4.4, color='#667085', clip_on=False)


def blend_with_white(color: str, amount: float = 0.84) -> str:
    rgb = np.array(mpl.colors.to_rgb(color), dtype=float)
    mixed = rgb * (1.0 - amount) + amount
    return mpl.colors.to_hex(np.clip(mixed, 0.0, 1.0))


def blend_with_black(color: str, amount: float = 0.18) -> str:
    rgb = np.array(mpl.colors.to_rgb(color), dtype=float)
    mixed = rgb * (1.0 - amount)
    return mpl.colors.to_hex(np.clip(mixed, 0.0, 1.0))


def color_with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    return mpl.colors.to_rgba(color, alpha)


def text_color_for_fill(color: str) -> str:
    r, g, b = mpl.colors.to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'white' if luminance < 0.68 else '#1F2937'


def format_compact_stat(value: float) -> str:
    return f'{value:.2g}' if abs(value) < 0.1 else f'{value:.3g}'


def draw_mini_donut(
    ax: plt.Axes,
    *,
    value: float,
    max_value: float,
    color: str,
    center_text: str,
    ring_width: float = 0.36,
) -> None:
    fraction = float(np.clip(value / max_value if max_value > 0 else 0.0, 0.0, 1.0))
    ax.pie(
        [fraction, 1.0 - fraction],
        startangle=90,
        counterclock=False,
        colors=[color, blend_with_white(color, 0.80)],
        wedgeprops={'width': ring_width, 'edgecolor': 'white', 'linewidth': 0.8},
    )
    ax.text(0.0, 0.02, center_text, ha='center', va='center', fontsize=6.0, fontweight='bold', color='#344054')
    ax.set_aspect('equal')
    ax.set_axis_off()


def compact_geomx_label(value: str) -> str:
    return {
        'Dorsolateral': 'Dorsal lat.',
        'Dorsomedial': 'Dorsal med.',
        'Ventrolateral': 'Ventral lat.',
        'Ventromedial': 'Ventral med.',
        'total_down': 'Total down',
        'total_up': 'Total up',
    }.get(str(value), str(value))


def wrap_tile_label(value: str) -> str:
    label = compact_geomx_label(value)
    if ' ' in label:
        first, second = label.split(' ', 1)
        return f'{first}\n{second}'
    return label


def add_round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = '#D6DEE6',
    linewidth: float = 0.8,
    radius: float = 0.04,
) -> mpl.patches.FancyBboxPatch:
    patch = mpl.patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f'round,pad=0.012,rounding_size={radius}',
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    return patch


def pretty_source(source_id: str) -> str:
    return SOURCE_LABELS.get(str(source_id), str(source_id))


def pretty_cohort(value: str) -> str:
    mapping = {
        'control_known_age': 'Control',
        'pd_known_age': 'PD',
        'lbd_bridge': 'LBD bridge',
        'unknown_age_auxiliary': 'Auxiliary',
        'control': 'Control',
        'pd': 'PD',
    }
    return mapping.get(str(value), str(value).replace('_', ' ').title())


def pretty_task(task_name: str) -> str:
    mapping = {
        'microglia_to_dopaminergic_neuron': 'Microglia -> DA neuron',
        'astrocyte_to_dopaminergic_neuron': 'Astrocyte -> DA neuron',
        'endothelial_to_dopaminergic_neuron': 'Endothelial -> DA neuron',
        'microglia_to_neuron': 'Microglia -> neuron',
    }
    return mapping.get(str(task_name), str(task_name).replace('_', ' '))


def short_program_name(value: str, width: int = 22) -> str:
    mapping = {
        'APOE lipoprotein receptor signaling': 'APOE-lipoprotein',
        'SLIT-ROBO direct guidance': 'Canonical SLIT-ROBO',
        'ROBO guidance context organizers': 'Guidance context',
        'SLIT and ROBO guidance': 'Unsplit guidance',
        'APP centered stress signaling': 'APP-centered stress',
        'Adhesion organizer network': 'Adhesion organizer',
        'Glutamatergic ionotropic receptor stress': 'Ionotropic glutamate',
        'Glutamatergic metabotropic receptor stress': 'Metabotropic glutamate',
    }
    text = mapping.get(str(value), str(value))
    return fill(text, width=width)


def compact_archive_label(value: str, width: int = 18) -> str:
    text = str(value).replace('20260402_', '')
    replacements = {
        'multisource': 'multi-src',
        'downweight_b0f0': 'downwt b0f0',
        'current_mechanistic': 'current mech',
        'revert_graph_only': 'graph-only',
        'softsrc': 'soft-src',
        'strongsrc': 'strong-src',
        'ultrasoftsrc': 'ultra-soft',
        'nosrcbal': 'no-srcbal',
        'exploratory_candidates': 'exploratory',
        'exploratory_step4_candidates': 'exploratory',
        'exclude_b0f0': 'exclude b0f0',
        'default': 'default',
        'seed13': 's13',
        'seed29': 's29',
        'seed7': 's7',
        'w010': 'w010',
        'w050': 'w050',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return fill(text.replace('_', ' '), width=width)


def pretty_claim_tier(value: str) -> str:
    mapping = {
        'external_mechanistic_candidate': 'External mechanistic',
        'externally_supported_program': 'External program',
        'genetic_support_only': 'Genetic-only',
    }
    return mapping.get(str(value), str(value).replace('_', ' '))


def pretty_promotion_decision(value: str) -> str:
    mapping = {
        'main_figure_candidate': 'Main figure candidate',
        'priority_replication_candidate': 'Priority replication',
        'rerun_spatial_validation': 'Rerun spatial validation',
    }
    return mapping.get(str(value), str(value).replace('_', ' '))


def short_state_label(value: str) -> str:
    text = str(value)
    replacements = {
        'dopaminergic_neuron': 'DA neuron',
        'microglia': 'Microglia',
        'state_01_': 'S1 ',
        'state_02_': 'S2 ',
        'state_03_': 'S3 ',
        '_synaptic_organizer': ' synaptic',
        '_stress_response': ' stress',
        '_homeostatic': ' homeostatic',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return fill(text.replace('_', ' '), width=18)


def compact_state_token(value: str) -> str:
    mapping = {
        'microglia_state_01_homeostatic': 'Mg S1',
        'microglia_state_02_homeostatic': 'Mg S2',
        'microglia_state_03_stress_response': 'Mg S3',
        'dopaminergic_neuron_state_01_synaptic_organizer': 'DA S1',
        'dopaminergic_neuron_state_02_synaptic_organizer': 'DA S2',
        'dopaminergic_neuron_state_03_synaptic_organizer': 'DA S3',
    }
    return mapping.get(str(value), str(value).replace('_', ' '))


def compact_state_pair(sender_state: str, receiver_state: str) -> str:
    return f'{compact_state_token(sender_state)} -> {compact_state_token(receiver_state)}'


def sankey_state_label(value: str) -> str:
    mapping = {
        'microglia_state_01_homeostatic': 'Mg S1\nhomeostatic',
        'microglia_state_02_homeostatic': 'Mg S2\nhomeostatic',
        'microglia_state_03_stress_response': 'Mg S3\nstress',
        'dopaminergic_neuron_state_01_synaptic_organizer': 'DA S1\nsynaptic',
        'dopaminergic_neuron_state_02_synaptic_organizer': 'DA S2\nsynaptic',
        'dopaminergic_neuron_state_03_synaptic_organizer': 'DA S3\nsynaptic',
    }
    return mapping.get(str(value), short_state_label(value))


def sankey_program_label(value: str) -> str:
    mapping = {
        'Glutamatergic ionotropic receptor stress': 'Ionotropic\nglutamate',
        'Glutamatergic metabotropic receptor stress': 'Metabotropic\nglutamate',
        'Immune neuronal IL18 interface': 'IL18\ninterface',
        'SLIT and ROBO guidance': 'Unsplit\nguidance',
        'APP centered stress signaling': 'APP-centered\nstress',
        'APOE lipoprotein receptor signaling': 'APOE-\nlipoprotein',
    }
    return mapping.get(str(value), short_program_name(str(value), width=12))


def parse_json_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith('['):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return [str(item) for item in parsed]
    return [item for item in text.split('|') if item]


def parse_pipe_flags(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {token for token in str(value).split('|') if token}


def parse_markdown_tables(path: Path) -> list[pd.DataFrame]:
    lines = path.read_text(encoding='utf-8').splitlines()
    tables: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            current.append(stripped)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)

    frames: list[pd.DataFrame] = []
    for table in tables:
        if len(table) < 2:
            continue
        header = [cell.strip() for cell in table[0].strip('|').split('|')]
        rows = []
        for raw in table[2:]:
            cells = [cell.strip() for cell in raw.strip('|').split('|')]
            if len(cells) == len(header):
                rows.append(cells)
        if rows:
            frames.append(pd.DataFrame(rows, columns=header))
    return frames


@lru_cache(maxsize=1)
def package_pd_definitive_manifest() -> pd.DataFrame:
    return pd.read_csv(PACKAGE_TABLE_DIR / 'pd_definitive_manifest.csv')


@lru_cache(maxsize=1)
def package_primary_visium_manifest() -> pd.DataFrame:
    return pd.read_csv(PACKAGE_TABLE_DIR / 'primary_visium_manifest.csv')


@lru_cache(maxsize=1)
def package_step2_source_task_coverage() -> pd.DataFrame:
    return pd.read_csv(PACKAGE_TABLE_DIR / 'step2_source_task_coverage.csv')


@lru_cache(maxsize=1)
def package_pd_model_fold_metrics() -> pd.DataFrame:
    return pd.read_csv(PACKAGE_TABLE_DIR / 'pd_model_fold_metrics.csv')


@lru_cache(maxsize=1)
def package_step3_top10() -> pd.DataFrame:
    return pd.read_csv(PACKAGE_TABLE_DIR / 'step3_top10_refined_programs.csv')


@lru_cache(maxsize=1)
def package_step3_candidate_evidence() -> pd.DataFrame:
    return pd.read_csv(PACKAGE_TABLE_DIR / 'step3_candidate_evidence_matrix.csv')


@lru_cache(maxsize=1)
def package_gene_support_summary() -> pd.DataFrame:
    return pd.read_csv(PACKAGE_TABLE_DIR / 'gene_level_support_summary.csv')


@lru_cache(maxsize=1)
def package_program_genetics() -> pd.DataFrame:
    return pd.read_csv(PACKAGE_TABLE_DIR / 'program_genetic_evidence_matrix.csv')


@lru_cache(maxsize=1)
def package_step4_results_table() -> pd.DataFrame:
    return pd.read_csv(PACKAGE_TABLE_DIR / 'step4_results_table_for_manuscript.csv')


@lru_cache(maxsize=1)
def package_step4_decision_log() -> pd.DataFrame:
    return pd.read_csv(PACKAGE_TABLE_DIR / 'step4_candidate_decision_log.csv')


@lru_cache(maxsize=1)
def package_full_refresh_leaderboard() -> pd.DataFrame:
    return pd.read_csv(PACKAGE_TABLE_DIR / 'full_refresh_leaderboard.csv')


@lru_cache(maxsize=1)
def package_state_mapping_sensitivity() -> pd.DataFrame:
    frame = parse_markdown_tables(PACKAGE_TABLE_DIR / 'state_mapping_sensitivity_report.md')[0].copy()
    for column in ['primary_sensitivity_correlation', 'mean_primary_in_snpc', 'mean_primary_in_surrounding']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return frame


@lru_cache(maxsize=1)
def package_gse253462_donor_counts() -> pd.DataFrame:
    frame = parse_markdown_tables(PACKAGE_TABLE_DIR / 'gse253462_nigra_sample_summary.md')[1].copy()
    frame['n_donors'] = pd.to_numeric(frame['n_donors'], errors='coerce')
    return frame


@lru_cache(maxsize=1)
def step2_graph_manifest() -> pd.DataFrame:
    return pd.read_csv(STEP2_DIR / '02_cohorts' / 'step2_graph_manifest.csv')


@lru_cache(maxsize=1)
def step2_pd_predictions() -> pd.DataFrame:
    return pd.read_csv(STEP2_DIR / '06_models' / 'pd_model' / 'pd_model_predictions.csv')


@lru_cache(maxsize=1)
def step2_pd_lbd_scores() -> pd.DataFrame:
    return pd.read_csv(STEP2_DIR / '06_models' / 'pd_model' / 'pd_model_lbd_bridge_scores.csv')


@lru_cache(maxsize=1)
def step2_baseline_metrics() -> pd.DataFrame:
    return pd.read_csv(STEP2_DIR / '04_baselines' / 'step2_baseline_metrics.csv')


@lru_cache(maxsize=1)
def step2_ranked_pathways() -> pd.DataFrame:
    return pd.read_csv(STEP2_DIR / '08_interpretation' / 'step2_ranked_pathways.csv')


@lru_cache(maxsize=1)
def step2_ranked_lr_pairs() -> pd.DataFrame:
    return pd.read_csv(STEP2_DIR / '08_interpretation' / 'step2_ranked_lr_pairs.csv')


@lru_cache(maxsize=1)
def step2_step3_candidates() -> pd.DataFrame:
    return pd.read_csv(STEP2_DIR / '09_exports' / 'step2_top_candidates_for_step3.csv')


@lru_cache(maxsize=1)
def step2_step4_candidates() -> pd.DataFrame:
    return pd.read_csv(STEP2_DIR / '09_exports' / 'step2_top_candidates_for_step4.csv')


@lru_cache(maxsize=1)
def step2_holdout_baseline_scores() -> pd.DataFrame:
    return pd.read_csv(STEP2_DIR / '07_robustness' / 'gse157783_holdout' / 'gse157783_mgda_baseline_scores.csv')


@lru_cache(maxsize=1)
def step2_holdout_patient_counts() -> pd.DataFrame:
    return pd.read_csv(STEP2_DIR / '07_robustness' / 'gse157783_holdout' / 'gse157783_patient_celltype_counts.csv')


@lru_cache(maxsize=1)
def step2_label_confidence_sensitivity() -> pd.DataFrame:
    return pd.read_csv(STEP2_DIR / '07_robustness' / 'step2_label_confidence_sensitivity.csv')


@lru_cache(maxsize=1)
def step2_cell_count_robustness() -> pd.DataFrame:
    return pd.read_csv(STEP2_DIR / '07_robustness' / 'step2_cell_count_robustness.csv')


@lru_cache(maxsize=1)
def step3_program_scores_by_donor() -> pd.DataFrame:
    return pd.read_csv(STEP3_DIR / '03_donor_patterns' / 'step3_program_scores_by_donor.csv', low_memory=False)


@lru_cache(maxsize=1)
def step3_pair_scores_by_donor() -> pd.DataFrame:
    return pd.read_csv(STEP3_DIR / '03_donor_patterns' / 'step3_pair_scores_by_donor.csv', low_memory=False)


@lru_cache(maxsize=1)
def step3_candidate_state_support() -> pd.DataFrame:
    return pd.read_csv(STEP3_DIR / '05_states' / 'step3_candidate_state_support.csv')


@lru_cache(maxsize=1)
def step3_exemplar_edges() -> pd.DataFrame:
    return pd.read_csv(STEP3_DIR / '06_exemplars' / 'step3_exemplar_edges.csv')


@lru_cache(maxsize=1)
def step3_confound_summary() -> pd.DataFrame:
    return pd.read_csv(STEP3_DIR / '07_confounds' / 'step3_confound_summary.csv')


@lru_cache(maxsize=1)
def figure2_donor_level_predictions() -> pd.DataFrame:
    pred = step2_pd_predictions()
    donor = (
        pred.loc[pred['split'].eq('val')]
        .groupby(['fold', 'split_group_id', 'source_id', 'disease_target'], as_index=False)
        .agg(
            pd_probability=('pd_probability', 'mean'),
            pd_probability_raw=('pd_probability_raw', 'mean'),
            decision_threshold=('decision_threshold', 'mean'),
            calibration_intercept=('calibration_intercept', 'mean'),
            calibration_slope=('calibration_slope', 'mean'),
            n_graphs=('graph_id', 'nunique'),
        )
    )
    meta = (
        step2_graph_manifest()[['split_group_id', 'cohort_name', 'disease_group', 'age_years']]
        .drop_duplicates(subset=['split_group_id'])
    )
    donor = donor.merge(meta, on='split_group_id', how='left')
    donor['source_label'] = donor['source_id'].map(pretty_source)
    return donor


@lru_cache(maxsize=1)
def figure2_bridge_predictions() -> pd.DataFrame:
    bridge = (
        step2_pd_lbd_scores()
        .groupby(['fold', 'split_group_id', 'source_id'], as_index=False)
        .agg(pd_probability=('pd_probability', 'mean'))
    )
    bridge['cohort_name'] = 'lbd_bridge'
    bridge['disease_group'] = 'lbd'
    return bridge


def cohort_color(value: str) -> str:
    return COHORT_COLORS.get(str(value), PAPER_BLUE_LIGHT)


@lru_cache(maxsize=1)
def figure5_primary_results() -> pd.Series:
    frame = pd.read_csv(STEP4_DIR / '03_spatial' / '04_primary_tests' / 'program_primary_spatial_results.csv')
    return frame.loc[frame['program_id'] == FIGURE5_PROGRAM_ID].iloc[0]


@lru_cache(maxsize=1)
def figure5_donor_metrics() -> pd.DataFrame:
    frame = pd.read_csv(
        STEP4_DIR / '03_spatial' / '04_primary_tests' / 'program_donor_level_spatial_summary.csv',
        usecols=[
            'program_id',
            'sample_accession',
            'disease_group',
            'anatomical_compartment',
            'adjacency_excess_score',
            'same_spot_program_score',
            'n_spots',
        ],
    )
    frame = frame.loc[frame['program_id'] == FIGURE5_PROGRAM_ID].copy()
    snpc = frame.loc[
        frame['anatomical_compartment'] == 'SNpc',
        ['sample_accession', 'disease_group', 'adjacency_excess_score', 'same_spot_program_score', 'n_spots'],
    ].copy()
    contrast = (
        frame.pivot_table(
            index=['sample_accession', 'disease_group'],
            columns='anatomical_compartment',
            values='adjacency_excess_score',
            aggfunc='mean',
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    contrast['snpc_minus_surround'] = contrast['SNpc'] - contrast['surrounding_tissue']
    snpc = snpc.merge(contrast[['sample_accession', 'snpc_minus_surround']], on='sample_accession', how='left')
    snpc['disease_group'] = pd.Categorical(snpc['disease_group'], categories=FIGURE5_GROUP_ORDER, ordered=True)
    return snpc.sort_values(['disease_group', 'sample_accession']).reset_index(drop=True)


@lru_cache(maxsize=1)
def figure5_spot_scores() -> pd.DataFrame:
    frame = pd.read_csv(
        STEP4_DIR / '03_spatial' / '03_anchor_scoring' / 'program_spatial_scores.csv',
        usecols=[
            'sample_accession',
            'disease_group',
            'anatomical_compartment',
            'imageX',
            'imageY',
            'program_id',
            'adjacency_excess_score',
        ],
    )
    frame = frame.loc[frame['program_id'] == FIGURE5_PROGRAM_ID].copy()
    frame['disease_group'] = pd.Categorical(frame['disease_group'], categories=FIGURE5_GROUP_ORDER, ordered=True)
    return frame


@lru_cache(maxsize=1)
def figure5_geomx_scores() -> pd.DataFrame:
    frame = pd.read_csv(STEP4_DIR / '03_spatial' / '05_secondary_replication' / 'geomx_roi_program_scores.csv')
    frame = frame.loc[frame['program_id'] == FIGURE5_PROGRAM_ID].copy()
    block_order = ['total_down', 'total_up', 'Dorsolateral', 'Dorsomedial', 'Ventrolateral', 'Ventromedial']
    frame['block_name'] = pd.Categorical(frame['block_name'], categories=block_order, ordered=True)
    return frame.sort_values('block_name').reset_index(drop=True)


@lru_cache(maxsize=1)
def figure5_holdout_scores() -> pd.DataFrame:
    frame = pd.read_csv(STEP4_DIR / '04_internal_robustness' / 'holdout_annex_by_source.csv')
    frame = frame.loc[frame['program_id'] == FIGURE5_PROGRAM_ID].copy()
    source_order = ['GSE253462', 'GSE157783']
    frame['holdout_source_id'] = pd.Categorical(frame['holdout_source_id'], categories=source_order, ordered=True)
    return frame.sort_values('holdout_source_id').reset_index(drop=True)


@lru_cache(maxsize=1)
def figure5_gpnmb_support() -> pd.Series:
    frame = pd.read_csv(STEP4_DIR / '02_genetics' / '01_gene_level' / 'gene_level_support_summary.csv')
    return frame.loc[frame['gene_symbol'] == 'GPNMB'].iloc[0]


@lru_cache(maxsize=1)
def figure6_primary_results() -> pd.DataFrame:
    frame = pd.read_csv(STEP4_DIR / '03_spatial' / '04_primary_tests' / 'program_primary_spatial_results.csv')
    order = [FIGURE6_PROGRAM_IDS['context'], FIGURE6_PROGRAM_IDS['canonical']]
    frame = frame.loc[frame['program_id'].isin(order)].copy()
    frame['program_id'] = pd.Categorical(frame['program_id'], categories=order, ordered=True)
    return frame.sort_values('program_id').reset_index(drop=True)


@lru_cache(maxsize=1)
def figure6_donor_metrics() -> pd.DataFrame:
    frame = pd.read_csv(
        STEP4_DIR / '03_spatial' / '04_primary_tests' / 'program_donor_level_spatial_summary.csv',
        usecols=[
            'program_id',
            'sample_accession',
            'disease_group',
            'anatomical_compartment',
            'adjacency_excess_score',
        ],
    )
    frame = frame.loc[
        frame['program_id'].isin(FIGURE6_PROGRAM_IDS.values()) & (frame['anatomical_compartment'] == 'SNpc')
    ].copy()
    frame['disease_group'] = pd.Categorical(frame['disease_group'], categories=FIGURE5_GROUP_ORDER, ordered=True)
    return frame.sort_values(['program_id', 'disease_group', 'sample_accession']).reset_index(drop=True)


@lru_cache(maxsize=1)
def figure6_branch_evidence() -> pd.DataFrame:
    frame = pd.read_csv(STEP4_DIR / '05_integration' / 'step4_final_evidence_matrix.csv')
    order = [FIGURE6_PROGRAM_IDS['context'], FIGURE6_PROGRAM_IDS['canonical']]
    frame = frame.loc[frame['program_id'].isin(order)].copy()
    frame['program_id'] = pd.Categorical(frame['program_id'], categories=order, ordered=True)
    return frame.sort_values('program_id').reset_index(drop=True)


@lru_cache(maxsize=1)
def figure6_pair_metrics() -> pd.DataFrame:
    order = ['SLIT2->ROBO2', 'SLIT2->ROBO1']
    frame = pd.read_parquet(
        STEP4_DIR / '03_spatial' / '03_anchor_scoring' / 'anchor_pair_spatial_scores.parquet',
        columns=[
            'program_id',
            'pair_key',
            'disease_group',
            'anatomical_compartment',
            'same_spot_pair_score_z',
            'adjacent_pair_score_z',
            'adjacency_excess_score_z',
        ],
    )
    frame = frame.loc[
        (frame['program_id'] == FIGURE6_PROGRAM_IDS['canonical'])
        & (frame['anatomical_compartment'] == 'SNpc')
        & frame['pair_key'].isin(order)
    ].copy()
    frame['pair_key'] = pd.Categorical(frame['pair_key'], categories=order, ordered=True)
    frame['disease_group'] = pd.Categorical(frame['disease_group'], categories=FIGURE5_GROUP_ORDER, ordered=True)
    summary = (
        frame.groupby(['pair_key', 'disease_group'], observed=False)[
            ['same_spot_pair_score_z', 'adjacent_pair_score_z', 'adjacency_excess_score_z']
        ]
        .mean()
        .reset_index()
    )
    return summary.sort_values(['pair_key', 'disease_group']).reset_index(drop=True)


@lru_cache(maxsize=1)
def figure6_holdout_by_source() -> pd.DataFrame:
    frame = pd.read_csv(STEP4_DIR / '04_internal_robustness' / 'holdout_annex_by_source.csv')
    source_order = ['GSE157783', 'GSE253462']
    frame = frame.loc[frame['program_id'] == FIGURE6_PROGRAM_IDS['canonical']].copy()
    frame['holdout_source_id'] = pd.Categorical(frame['holdout_source_id'], categories=source_order, ordered=True)
    return frame.sort_values('holdout_source_id').reset_index(drop=True)


def figure5_metric_panel(
    ax: plt.Axes,
    *,
    label: str,
    metric: str,
    title: str,
    y_label: str,
    beta_key: str,
    directional_p_key: str,
    y_limits: tuple[float, float],
) -> None:
    frame = figure5_donor_metrics()
    figure5_panel_label(ax, label)
    data = [frame.loc[frame['disease_group'] == group, metric].to_numpy(dtype=float) for group in FIGURE5_GROUP_ORDER]
    positions = np.arange(len(FIGURE5_GROUP_ORDER), dtype=float)
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': '#344054', 'linewidth': 1.0},
        whiskerprops={'color': '#667085', 'linewidth': 0.9},
        capprops={'color': '#667085', 'linewidth': 0.9},
    )
    for artist, group in zip(box['boxes'], FIGURE5_GROUP_ORDER, strict=True):
        artist.set_facecolor(color_with_alpha(FIGURE5_GROUP_COLORS[group], 0.22))
        artist.set_edgecolor(FIGURE5_GROUP_COLORS[group])
        artist.set_linewidth(1.0)

    rng = np.random.default_rng(42)
    for idx, (group, values) in enumerate(zip(FIGURE5_GROUP_ORDER, data, strict=True)):
        x_coord = idx + rng.uniform(-0.08, 0.08, size=values.size)
        ax.scatter(
            x_coord,
            values,
            s=22,
            color=FIGURE5_GROUP_COLORS[group],
            alpha=0.92,
            linewidths=0.5,
            edgecolors='white',
            zorder=3,
        )
        ax.scatter(
            [idx],
            [float(values.mean())],
            marker='D',
            s=32,
            color=FIGURE5_GROUP_COLORS[group],
            edgecolors='white',
            linewidths=0.8,
            zorder=4,
        )

    summary = figure5_primary_results()
    subtitle = f"beta {summary[beta_key]:+.2f} | one-sided P {summary[directional_p_key]:.2f}"
    set_left_title(ax, title)
    ax.text(
        0.98,
        0.98,
        subtitle,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=4.0,
        color='#667085',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.3, 'alpha': 0.9},
    )
    ax.axhline(0.0, linestyle='--', linewidth=0.75, color='#D0D5DD', zorder=0)
    ax.grid(axis='y', linestyle='--', linewidth=0.55, color='#E4E7EC')
    ax.set_ylabel(y_label)
    ax.set_xticks(positions, labels=[FIGURE5_GROUP_LABELS[group] for group in FIGURE5_GROUP_ORDER])
    ax.set_ylim(*y_limits)
    ax.set_xlim(-0.55, 1.55)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def geomx_dumbbell_panel(ax: plt.Axes) -> None:
    frame = figure5_geomx_scores()
    figure5_panel_label(ax, 'f')
    ax.set_axis_off()
    floating_title(ax, 'GeoMx ROI replication', '2/6 concordant', y=1.04, subtitle_y=1.01)

    cmap = LinearSegmentedColormap.from_list(
        'figure5_geomx',
        [PAPER_RED, blend_with_white(PAPER_BLUE_LIGHT, 0.92), PAPER_BLUE],
    )
    norm = TwoSlopeNorm(vmin=-3.5, vcenter=0.0, vmax=0.5)
    tile_map = {
        'total_down': (0.03, 0.71, 0.44, 0.17),
        'total_up': (0.53, 0.71, 0.44, 0.17),
        'Dorsolateral': (0.03, 0.46, 0.44, 0.17),
        'Dorsomedial': (0.53, 0.46, 0.44, 0.17),
        'Ventrolateral': (0.03, 0.21, 0.44, 0.17),
        'Ventromedial': (0.53, 0.21, 0.44, 0.17),
    }

    for row in frame.itertuples(index=False):
        x0, y0, width, height = tile_map[str(row.block_name)]
        fill_color = cmap(norm(float(row.program_score)))
        edge_color = FIGURE5_MATCH_COLOR if bool(row.direction_match) else '#D7DDE5'
        add_round_box(ax, x0, y0, width, height, facecolor=fill_color, edgecolor=edge_color, linewidth=1.2 if row.direction_match else 0.8, radius=0.028)
        text_color = 'white' if abs(float(row.program_score)) > 1.0 else '#344054'
        ax.text(
            x0 + 0.035,
            y0 + height - 0.04,
            wrap_tile_label(str(row.block_name)),
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=4.5,
            color=text_color,
            fontweight='semibold',
            linespacing=0.95,
        )
        ax.text(x0 + 0.035, y0 + 0.06, f'{float(row.program_score):+.2f}', transform=ax.transAxes, ha='left', va='center', fontsize=6.9, color=text_color, fontweight='bold')
        ax.text(
            x0 + width - 0.04,
            y0 + 0.028,
            f'{int(row.n_genes_present)} gene' if int(row.n_genes_present) == 1 else f'{int(row.n_genes_present)} genes',
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=3.8,
            color=text_color if abs(float(row.program_score)) > 0.6 else '#667085',
        )

    ax.text(
        0.03,
        0.13,
        'Teal border = expected direction matched',
        transform=ax.transAxes,
        ha='left',
        va='center',
        fontsize=4.5,
        color=FIGURE5_MATCH_COLOR,
    )
    cax = ax.inset_axes([0.10, 0.04, 0.80, 0.05])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal', ticks=[-3.5, -2.0, -1.0, 0.0, 0.5])
    cb.set_ticklabels(['-3.5', '-2', '-1', '0', '0.5'])
    cax.tick_params(labelsize=4.0, length=0, pad=1)
    cax.set_xlabel('Program score', fontsize=4.5, labelpad=1)


def holdout_donut_panel(ax: plt.Axes) -> None:
    frame = figure5_holdout_scores()
    figure5_panel_label(ax, 'g')
    ax.set_axis_off()
    floating_title(ax, 'Internal holdout', '2/2 concordant', y=1.04, subtitle_y=1.01)
    ax.text(0.0, 0.95, 'Internal expectation calibrated as PD > control', transform=ax.transAxes, ha='left', va='top', fontsize=4.7, color='#667085')
    max_effect = float(frame['mean_holdout_pd_minus_control'].max()) * 1.05
    card_positions = [0.57, 0.25]

    for y0, row in zip(card_positions, frame.itertuples(index=False), strict=True):
        add_round_box(ax, 0.02, y0, 0.96, 0.22, facecolor='#F8FAFC', edgecolor='#D6DEE6', linewidth=0.8, radius=0.035)
        ax.text(0.07, y0 + 0.16, str(row.holdout_source_id), transform=ax.transAxes, ha='left', va='center', fontsize=5.4, color='#344054', fontweight='semibold')
        ax.text(
            0.93,
            y0 + 0.16,
            'match',
            transform=ax.transAxes,
            ha='right',
            va='center',
            fontsize=4.4,
            color=FIGURE5_MATCH_COLOR,
            fontweight='semibold',
        )
        add_round_box(
            ax,
            0.07,
            y0 + 0.07,
            0.54,
            0.05,
            facecolor=blend_with_white(PAPER_BLUE_LIGHT, 0.72),
            edgecolor=blend_with_white(PAPER_BLUE_LIGHT, 0.72),
            linewidth=0.0,
            radius=0.028,
        )
        add_round_box(
            ax,
            0.07,
            y0 + 0.07,
            0.54 * float(row.mean_holdout_pd_minus_control) / max_effect,
            0.05,
            facecolor=FIGURE5_SOURCE_COLORS[str(row.holdout_source_id)],
            edgecolor=FIGURE5_SOURCE_COLORS[str(row.holdout_source_id)],
            linewidth=0.0,
            radius=0.028,
        )
        ax.text(0.66, y0 + 0.095, f'{float(row.mean_holdout_pd_minus_control):+.2f}', transform=ax.transAxes, ha='left', va='center', fontsize=7.3, color='#2B2F33', fontweight='bold')
        ax.text(
            0.07,
            y0 + 0.025,
            f'{int(row.n_anchor_pairs_found_in_holdout)} anchor pairs recovered',
            transform=ax.transAxes,
            ha='left',
            va='center',
            fontsize=4.6,
            color='#667085',
        )
    ax.text(0.07, 0.16, 'Mean PD - control effect', transform=ax.transAxes, ha='left', va='center', fontsize=4.5, color='#667085')


def gpnmb_genetics_panel(host_ax: plt.Axes) -> None:
    row = figure5_gpnmb_support()
    figure5_panel_label(host_ax, 'd', x=-0.07)
    host_ax.axis('off')
    floating_title(host_ax, 'GPNMB genetics', 'MR + coloc', y=1.04, subtitle_y=1.01)
    beta_values = [float(row['best_primary_mr_beta']), float(row['best_secondary_mr_beta'])]
    coloc_values = [float(row['best_primary_coloc_h4_proxy']), float(row['best_secondary_coloc_h4_proxy'])]
    beta_fdr = [float(row['best_primary_mr_p_adj_fdr']), float(row['best_secondary_mr_p_adj_fdr'])]
    beta_max = max(0.12, max(beta_values) * 1.15)
    host_ax.text(0.03, 0.90, 'MR beta', transform=host_ax.transAxes, ha='left', va='center', fontsize=5.1, color='#344054', fontweight='semibold')
    host_ax.text(0.98, 0.90, f'scale 0-{beta_max:.2f}', transform=host_ax.transAxes, ha='right', va='center', fontsize=4.0, color='#98A2B3')
    host_ax.text(0.03, 0.47, 'Coloc H4', transform=host_ax.transAxes, ha='left', va='center', fontsize=5.1, color='#344054', fontweight='semibold')

    donut_specs = [
        ([0.05, 0.48, 0.40, 0.30], beta_values[0], beta_max, PAPER_BLUE, f'{beta_values[0]:+.2f}', 'Primary', f'FDR {format_compact_stat(beta_fdr[0])}'),
        ([0.55, 0.48, 0.40, 0.30], beta_values[1], beta_max, PAPER_RED, f'{beta_values[1]:+.2f}', 'Secondary', f'FDR {format_compact_stat(beta_fdr[1])}'),
        ([0.05, 0.09, 0.40, 0.28], coloc_values[0], 1.0, PAPER_BLUE_SOFT, f'{coloc_values[0]:.2f}', 'Primary', 'moderate'),
        ([0.55, 0.09, 0.40, 0.28], coloc_values[1], 1.0, PAPER_BLUE_LIGHT, f'{coloc_values[1]:.2f}', 'Secondary', 'supportive'),
    ]
    for bounds, value, max_value, color, center_text, title, note in donut_specs:
        donut_ax = host_ax.inset_axes(bounds)
        draw_mini_donut(
            donut_ax,
            value=value,
            max_value=max_value,
            color=color,
            center_text=center_text,
        )
        x0, y0, width, height = bounds
        cx = x0 + width / 2.0
        host_ax.text(cx, y0 + height + 0.050, title, transform=host_ax.transAxes, ha='center', va='center', fontsize=4.4, color='#475467')
        host_ax.text(cx, y0 + height + 0.020, note, transform=host_ax.transAxes, ha='center', va='center', fontsize=3.8, color='#667085')

    add_round_box(host_ax, 0.06, 0.02, 0.88, 0.07, facecolor='#F8FAFC', edgecolor='#D6DEE6', linewidth=0.8, radius=0.03)
    host_ax.text(
        0.50,
        0.055,
        f"rs199357 | Putamen | Mean F {float(row['best_primary_mean_F']):.1f}",
        transform=host_ax.transAxes,
        ha='center',
        va='center',
        fontsize=4.2,
        color='#344054',
        fontweight='semibold',
    )


def spatial_sections_panel(host_ax: plt.Axes) -> None:
    frame = figure5_spot_scores()
    figure5_panel_label(host_ax, 'e', x=-0.03)
    host_ax.axis('off')
    floating_title(host_ax, 'Representative spatial sections', 'Light-blue rings = top SNpc hotspots', y=1.04, subtitle_y=1.01)
    host_ax.text(0.0, 0.95, 'Adjacency excess across reconstructed compartments', transform=host_ax.transAxes, ha='left', va='top', fontsize=4.8, color='#667085')

    cmap = LinearSegmentedColormap.from_list(
        'figure5_spatial',
        [PAPER_RED, blend_with_white(PAPER_TEAL_LIGHT, 0.92), PAPER_TEAL],
    )
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    sample_axes = {
        'control': host_ax.inset_axes([0.00, 0.15, 0.46, 0.72]),
        'pd': host_ax.inset_axes([0.52, 0.15, 0.46, 0.72]),
    }

    for group, sample_ax in sample_axes.items():
        sample_id = FIGURE5_REPRESENTATIVE_SAMPLES[group]
        sample = frame.loc[frame['sample_accession'] == sample_id].copy()
        x_values = sample['imageX'].to_numpy(dtype=float)
        y_values = sample['imageY'].to_numpy(dtype=float)
        x_values = (x_values - x_values.min()) / (x_values.max() - x_values.min())
        y_values = (y_values - y_values.min()) / (y_values.max() - y_values.min())
        sample_ax.scatter(x_values, y_values, c=sample['adjacency_excess_score'], cmap=cmap, norm=norm, s=10, linewidths=0.0, alpha=0.90)
        snpc_mask = sample['anatomical_compartment'].to_numpy() == 'SNpc'
        sample_ax.scatter(x_values[snpc_mask], y_values[snpc_mask], s=28, facecolors='none', edgecolors='#5B5F66', linewidths=0.45, alpha=0.78)
        hotspot_threshold = float(sample.loc[snpc_mask, 'adjacency_excess_score'].quantile(0.95))
        hotspot_mask = snpc_mask & (sample['adjacency_excess_score'].to_numpy(dtype=float) >= hotspot_threshold)
        sample_ax.scatter(
            x_values[hotspot_mask],
            y_values[hotspot_mask],
            s=52,
            facecolors='none',
            edgecolors=FIGURE5_HOTSPOT_COLOR,
            linewidths=0.9,
        )
        sample_ax.set_title(f"{FIGURE5_GROUP_LABELS[group]} | {sample_id}", fontsize=6.4, pad=4, loc='center')
        sample_ax.set_aspect('equal')
        sample_ax.invert_yaxis()
        sample_ax.set_xticks([])
        sample_ax.set_yticks([])
        for spine in sample_ax.spines.values():
            spine.set_visible(False)

    cax = host_ax.inset_axes([0.34, 0.05, 0.32, 0.035])
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
    cax.tick_params(labelsize=4.4, length=0, pad=1)
    cax.set_xlabel('Adjacency excess', fontsize=4.6, labelpad=1)


def integrated_matrix_panel(ax: plt.Axes) -> None:
    cols = ['Primary', 'GeoMx', 'Gene\nMR', 'Prog.\ngen.', 'Cross-\nset', 'Source', 'Holdout', 'Final']
    rows = ['APOE-LRP', 'SLIT-ROBO\ncontext', 'SLIT-ROBO\ncanon.']
    data = np.array(
        [
            [0.70, 0.70, 0.19, 0.60, 0.70, 0.05, 0.00, 0.51],
            [0.00, 0.00, 0.10, 0.60, 0.00, 0.05, 0.00, 0.07],
            [1.00, 0.35, 0.08, 0.60, 0.60, 0.05, 0.00, 0.52],
        ],
        dtype=float,
    )
    pf.heatmap(
        data,
        title=None,
        colorscale='RdBu_r',
        zmin=0.0,
        zmax=1.0,
        annotate=True,
        annotate_fmt='.2f',
        cbar=False,
        cell_border_line_width=0.65,
        cell_border_color='#F3F4F6',
        tick_length=0.0,
        show_spines=False,
        ax=ax,
    )
    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(rows)), labels=rows)
    ax.tick_params(axis='x', labelsize=4.1, pad=1.5, length=0)
    ax.tick_params(axis='y', labelsize=5.0, length=0)
    set_left_title(ax, 'Integrated Step 4 evidence\nmatrix')
    ax.set_xlabel('')
    ax.set_ylabel('')


def figure6_flow_node(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    subtitle: str | None,
    facecolor: str,
    edgecolor: str,
) -> None:
    add_round_box(ax, x, y, width, height, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0, radius=0.04)
    ax.text(
        x + width * 0.06,
        y + height * 0.63,
        title,
        transform=ax.transAxes,
        ha='left',
        va='center',
        fontsize=5.2,
        color='#2B2F33',
        fontweight='semibold',
    )
    if subtitle:
        ax.text(
            x + width * 0.06,
            y + height * 0.28,
            subtitle,
            transform=ax.transAxes,
            ha='left',
            va='center',
            fontsize=4.2,
            color='#667085',
        )


def figure6_flow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    rad: float = 0.0,
    linewidth: float = 1.4,
    alpha: float = 0.95,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
) -> None:
    patch = mpl.patches.FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        connectionstyle=f'arc3,rad={rad}',
        arrowstyle='Simple,tail_width=0.55,head_width=4.7,head_length=6.4',
        linewidth=0.0,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        mutation_scale=1.0,
    )
    ax.add_patch(patch)
    if label and label_xy is not None:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=4.0,
            color=blend_with_black(color, 0.08),
        )


def figure6_split_flow_panel(host_ax: plt.Axes) -> None:
    figure5_panel_label(host_ax, 'a', x=-0.02)
    host_ax.axis('off')
    floating_title(host_ax, 'Guidance split flow', y=1.03)
    host_ax.text(
        0.0,
        0.95,
        'Parent program split into direct-pair and organizer-context branches',
        transform=host_ax.transAxes,
        ha='left',
        va='top',
        fontsize=4.6,
        color='#667085',
    )

    parent_fill = blend_with_white(FIGURE6_COMPONENT_COLORS['highlight'], 0.76)
    canonical_fill = blend_with_white(FIGURE6_BRANCH_COLORS['canonical'], 0.86)
    context_fill = blend_with_white(FIGURE6_BRANCH_COLORS['context'], 0.85)
    evidence_fill = blend_with_white(FIGURE6_COMPONENT_COLORS['genetics'], 0.86)

    figure6_flow_node(
        host_ax,
        0.03,
        0.38,
        0.16,
        0.22,
        title='Parent guidance',
        subtitle='4 anchors | mixed semantics',
        facecolor=parent_fill,
        edgecolor=FIGURE6_COMPONENT_COLORS['final'],
    )
    figure6_flow_node(
        host_ax,
        0.31,
        0.60,
        0.20,
        0.19,
        title='Canonical branch',
        subtitle='SLIT2 -> ROBO1 / ROBO2',
        facecolor=canonical_fill,
        edgecolor=FIGURE6_BRANCH_COLORS['canonical'],
    )
    figure6_flow_node(
        host_ax,
        0.31,
        0.18,
        0.20,
        0.19,
        title='Context branch',
        subtitle='OLFM2 / NCAM1 organizers',
        facecolor=context_fill,
        edgecolor=FIGURE6_BRANCH_COLORS['context'],
    )
    figure6_flow_node(
        host_ax,
        0.61,
        0.62,
        0.16,
        0.16,
        title='Primary spatial',
        subtitle='SNpc component 1.00',
        facecolor=blend_with_white(FIGURE6_COMPONENT_COLORS['primary'], 0.88),
        edgecolor=FIGURE6_COMPONENT_COLORS['primary'],
    )
    figure6_flow_node(
        host_ax,
        0.61,
        0.20,
        0.16,
        0.16,
        title='Shared support',
        subtitle='Genetics 0.60 | Holdout suggestive',
        facecolor=evidence_fill,
        edgecolor=FIGURE6_COMPONENT_COLORS['genetics'],
    )
    figure6_flow_node(
        host_ax,
        0.84,
        0.56,
        0.13,
        0.22,
        title='Promote canonical',
        subtitle='priority replicate first',
        facecolor=blend_with_white(FIGURE6_BRANCH_COLORS['canonical'], 0.78),
        edgecolor=FIGURE6_BRANCH_COLORS['canonical'],
    )
    figure6_flow_node(
        host_ax,
        0.84,
        0.15,
        0.13,
        0.22,
        title='Rerun context',
        subtitle='spatial validation',
        facecolor=blend_with_white(FIGURE6_BRANCH_COLORS['context'], 0.77),
        edgecolor=FIGURE6_BRANCH_COLORS['context'],
    )

    figure6_flow_arrow(
        host_ax,
        (0.195, 0.52),
        (0.305, 0.70),
        color=FIGURE6_BRANCH_COLORS['canonical'],
        rad=0.06,
        label='split direct pair',
        label_xy=(0.25, 0.73),
    )
    figure6_flow_arrow(
        host_ax,
        (0.195, 0.46),
        (0.305, 0.27),
        color=FIGURE6_BRANCH_COLORS['context'],
        rad=-0.06,
        label='retain organizers',
        label_xy=(0.25, 0.24),
    )
    figure6_flow_arrow(
        host_ax,
        (0.51, 0.70),
        (0.605, 0.70),
        color=FIGURE6_COMPONENT_COLORS['primary'],
        label='spatial pass',
        label_xy=(0.56, 0.78),
    )
    figure6_flow_arrow(
        host_ax,
        (0.51, 0.27),
        (0.605, 0.28),
        color=FIGURE6_COMPONENT_COLORS['genetics'],
        label='shared evidence',
        label_xy=(0.56, 0.17),
    )
    figure6_flow_arrow(
        host_ax,
        (0.77, 0.70),
        (0.835, 0.67),
        color=FIGURE6_BRANCH_COLORS['canonical'],
        rad=0.05,
    )
    figure6_flow_arrow(
        host_ax,
        (0.77, 0.28),
        (0.835, 0.26),
        color=FIGURE6_BRANCH_COLORS['context'],
        rad=-0.05,
    )
    figure6_flow_arrow(
        host_ax,
        (0.69, 0.36),
        (0.835, 0.60),
        color=FIGURE6_COMPONENT_COLORS['genetics'],
        rad=0.18,
        alpha=0.70,
        label='supports',
        label_xy=(0.79, 0.45),
    )


def figure6_branch_score_panel(ax: plt.Axes) -> None:
    rows = figure6_branch_evidence().set_index('program_id')
    figure5_panel_label(ax, 'b', x=-0.08)
    components = ['Primary', 'GeoMx', 'Genetics', 'Holdout', 'Final']
    y = np.arange(len(components))[::-1]
    context_scores = [
        float(rows.loc[FIGURE6_PROGRAM_IDS['context'], 'primary_spatial_component']),
        float(rows.loc[FIGURE6_PROGRAM_IDS['context'], 'secondary_spatial_component']),
        float(rows.loc[FIGURE6_PROGRAM_IDS['context'], 'program_level_genetic_component']),
        float(rows.loc[FIGURE6_PROGRAM_IDS['context'], 'cross_dataset_consistency_component']),
        float(rows.loc[FIGURE6_PROGRAM_IDS['context'], 'final_step4_score']),
    ]
    canonical_scores = [
        float(rows.loc[FIGURE6_PROGRAM_IDS['canonical'], 'primary_spatial_component']),
        float(rows.loc[FIGURE6_PROGRAM_IDS['canonical'], 'secondary_spatial_component']),
        float(rows.loc[FIGURE6_PROGRAM_IDS['canonical'], 'program_level_genetic_component']),
        float(rows.loc[FIGURE6_PROGRAM_IDS['canonical'], 'cross_dataset_consistency_component']),
        float(rows.loc[FIGURE6_PROGRAM_IDS['canonical'], 'final_step4_score']),
    ]

    set_left_title(ax, 'Branch component score')
    ax.grid(axis='x', linestyle='--', linewidth=0.55, color=blend_with_white(FIGURE6_COMPONENT_COLORS['primary'], 0.86))
    for y_pos, ctx, can in zip(y, context_scores, canonical_scores, strict=True):
        ax.plot(
            [min(ctx, can), max(ctx, can)],
            [y_pos, y_pos],
            color=PAPER_BLUE_SOFT,
            linewidth=1.1,
            zorder=1,
        )
    ax.scatter(context_scores, y, s=34, color=FIGURE6_BRANCH_COLORS['context'], edgecolors='white', linewidths=0.55, label='Context', zorder=3)
    ax.scatter(canonical_scores, y, s=34, color=FIGURE6_BRANCH_COLORS['canonical'], edgecolors='white', linewidths=0.55, label='Canonical', zorder=3)
    ax.set_yticks(y, labels=components)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel('Component score')
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    ax.legend(frameon=False, fontsize=4.6, loc='upper left', bbox_to_anchor=(0.00, 1.02), ncol=2, handletextpad=0.35, columnspacing=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def figure6_decision_panel(host_ax: plt.Axes) -> None:
    frame = figure6_branch_evidence().set_index('program_id')
    figure5_panel_label(host_ax, 'c', x=-0.08)
    host_ax.axis('off')
    floating_title(host_ax, 'Branch decision grid', y=1.03)

    rows = ['Primary', 'Repl.', 'Gene', 'Program', 'Stability']
    cols = ['Context', 'Canonical']
    raw = np.array(
        [
            [frame.loc[FIGURE6_PROGRAM_IDS['context'], 'spatial_primary_gate'], frame.loc[FIGURE6_PROGRAM_IDS['canonical'], 'spatial_primary_gate']],
            [frame.loc[FIGURE6_PROGRAM_IDS['context'], 'spatial_replication_gate'], frame.loc[FIGURE6_PROGRAM_IDS['canonical'], 'spatial_replication_gate']],
            [frame.loc[FIGURE6_PROGRAM_IDS['context'], 'genetic_gene_level_gate'], frame.loc[FIGURE6_PROGRAM_IDS['canonical'], 'genetic_gene_level_gate']],
            [frame.loc[FIGURE6_PROGRAM_IDS['context'], 'genetic_program_level_gate'], frame.loc[FIGURE6_PROGRAM_IDS['canonical'], 'genetic_program_level_gate']],
            [frame.loc[FIGURE6_PROGRAM_IDS['context'], 'internal_stability_gate'], frame.loc[FIGURE6_PROGRAM_IDS['canonical'], 'internal_stability_gate']],
        ],
        dtype=object,
    )
    display_map = {'pass': 'Pass', 'fail': 'Fail', 'suggestive': 'Suggest.'}
    code_map = {'fail': 0, 'suggestive': 1, 'pass': 2}
    cmap = ListedColormap([FIGURE6_COMPONENT_COLORS['genetics'], FIGURE6_COMPONENT_COLORS['final'], FIGURE6_COMPONENT_COLORS['primary']])

    ax = host_ax.inset_axes([0.08, 0.22, 0.84, 0.70])
    numeric = np.vectorize(lambda v: code_map.get(str(v), 1))(raw)
    ax.imshow(numeric, aspect='auto', cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(cols)), labels=['Context', 'Canon.'])
    ax.set_yticks(np.arange(len(rows)), labels=rows)
    ax.tick_params(axis='x', labelsize=4.9, length=0, top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis='y', labelsize=4.9, length=0)
    for i in range(len(rows)):
        for j in range(len(cols)):
            value = str(raw[i, j])
            label = display_map.get(value, str(value))
            ax.text(
                j,
                i,
                label,
                ha='center',
                va='center',
                fontsize=4.8,
                color='white' if value in {'fail', 'pass'} else '#3F3A23',
                fontweight='bold',
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.0)
    ax.tick_params(which='minor', bottom=False, left=False)

    add_round_box(
        host_ax,
        0.08,
        0.04,
        0.36,
        0.10,
        facecolor=blend_with_white(FIGURE6_BRANCH_COLORS['context'], 0.78),
        edgecolor=FIGURE6_BRANCH_COLORS['context'],
        linewidth=0.8,
        radius=0.03,
    )
    host_ax.text(0.26, 0.09, 'Context: rerun spatial', transform=host_ax.transAxes, ha='center', va='center', fontsize=4.3, color=blend_with_black(FIGURE6_BRANCH_COLORS['context'], 0.32), fontweight='semibold')
    add_round_box(
        host_ax,
        0.56,
        0.04,
        0.36,
        0.10,
        facecolor=blend_with_white(FIGURE6_BRANCH_COLORS['canonical'], 0.80),
        edgecolor=FIGURE6_BRANCH_COLORS['canonical'],
        linewidth=0.8,
        radius=0.03,
    )
    host_ax.text(0.74, 0.09, 'Canon.: replicate first', transform=host_ax.transAxes, ha='center', va='center', fontsize=4.3, color=blend_with_black(FIGURE6_BRANCH_COLORS['canonical'], 0.25), fontweight='semibold')


def figure6_adjacency_panel(ax: plt.Axes, *, label: str, program_id: str, title: str) -> None:
    frame = figure6_donor_metrics()
    summary = figure6_primary_results().set_index('program_id').loc[program_id]
    sub = frame.loc[frame['program_id'] == program_id].copy()
    figure5_panel_label(ax, label, x=-0.08)
    data = [sub.loc[sub['disease_group'] == group, 'adjacency_excess_score'].to_numpy(dtype=float) for group in FIGURE5_GROUP_ORDER]
    palette = [FIGURE6_DISEASE_COLORS[group] for group in FIGURE5_GROUP_ORDER]
    pf.raincloud(
        data,
        category_names=[FIGURE5_GROUP_LABELS[group] for group in FIGURE5_GROUP_ORDER],
        color_palette=palette,
        half='upper',
        show_violin=True,
        show_box=True,
        show_points=True,
        violin_width=0.64,
        violin_face_alpha=0.22,
        violin_edge_line_width=0.95,
        box_width=0.16,
        box_facecolor='white',
        box_edgecolor=FIGURE6_COMPONENT_COLORS['highlight'],
        box_line_width=0.85,
        points_jitter=0.06,
        points_size=2.35,
        points_alpha=0.76,
        points_edgecolor='white',
        points_edge_line_width=0.22,
        show_y_grid=True,
        show_full_box=False,
        ax=ax,
    )
    for xpos, values, color in zip([1.0, 1.8], data, palette, strict=True):
        ax.scatter([xpos], [float(values.mean())], marker='D', s=28, color=color, edgecolors='white', linewidths=0.7, zorder=4)

    set_left_title(ax, title)
    ax.title.set_color(FIGURE6_BRANCH_COLORS['canonical'] if program_id == FIGURE6_PROGRAM_IDS['canonical'] else FIGURE6_BRANCH_COLORS['context'])
    ax.text(
        0.98,
        0.98,
        f"beta {float(summary['snpc_adjacency_beta']):+.2f} | one-sided P {float(summary['directional_p_snpc']):.2f}",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=4.0,
        color='#667085',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.3, 'alpha': 0.9},
    )
    ax.axhline(0.0, linestyle='--', linewidth=0.75, color=FIGURE6_COMPONENT_COLORS['highlight'], zorder=0)
    ax.grid(axis='y', linestyle='--', linewidth=0.55, color=blend_with_white(FIGURE6_COMPONENT_COLORS['highlight'], 0.36))
    ax.set_ylabel('Adjacency excess')
    ax.set_xlabel('')
    upper = max(float(np.nanmax(values)) for values in data if values.size > 0) + 0.09
    lower = min(float(np.nanmin(values)) for values in data if values.size > 0) - 0.05
    ax.set_ylim(min(-0.12, lower), max(0.46, upper))
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def figure6_pair_support_panel(host_ax: plt.Axes) -> None:
    figure5_panel_label(host_ax, 'f', x=-0.02)
    host_ax.axis('off')
    floating_title(host_ax, 'Canonical pair support', y=1.03)
    host_ax.text(
        0.0,
        0.95,
        'Pair-level SNpc adjacency plus program-level holdout effect',
        transform=host_ax.transAxes,
        ha='left',
        va='top',
        fontsize=4.6,
        color='#667085',
    )

    pair_metrics = figure6_pair_metrics()
    holdout = figure6_holdout_by_source()
    pair_order = ['SLIT2->ROBO2', 'SLIT2->ROBO1']
    card_bounds = {
        'SLIT2->ROBO2': (0.03, 0.44, 0.42, 0.40),
        'SLIT2->ROBO1': (0.53, 0.44, 0.42, 0.40),
    }

    for pair_key in pair_order:
        x0, y0, width, height = card_bounds[pair_key]
        add_round_box(
            host_ax,
            x0,
            y0,
            width,
            height,
            facecolor=blend_with_white(FIGURE6_BRANCH_COLORS['canonical'], 0.92),
            edgecolor=blend_with_white(FIGURE6_BRANCH_COLORS['canonical'], 0.28),
            linewidth=0.9,
            radius=0.03,
        )
        host_ax.text(
            x0 + 0.03,
            y0 + height - 0.06,
            pair_key.replace('->', '  ->  '),
            transform=host_ax.transAxes,
            ha='left',
            va='center',
            fontsize=4.8,
            color=FIGURE6_BRANCH_COLORS['canonical'],
            fontweight='semibold',
        )
        host_ax.text(
            x0 + 0.03,
            y0 + height - 0.11,
            'SNpc adjacency excess z',
            transform=host_ax.transAxes,
            ha='left',
            va='center',
            fontsize=4.0,
            color='#667085',
        )
        card_ax = host_ax.inset_axes([x0 + 0.08, y0 + 0.07, width - 0.13, height - 0.20])
        sub = pair_metrics.loc[pair_metrics['pair_key'] == pair_key].copy()
        values = [float(sub.loc[sub['disease_group'] == group, 'adjacency_excess_score_z'].iloc[0]) for group in FIGURE5_GROUP_ORDER]
        ypos = np.array([1, 0], dtype=float)
        card_ax.barh(ypos, values, color=[FIGURE6_DISEASE_COLORS['control'], FIGURE6_DISEASE_COLORS['pd']], height=0.36)
        card_ax.axvline(0.0, linewidth=0.75, color=FIGURE6_COMPONENT_COLORS['highlight'])
        for y_pos, value in zip(ypos, values, strict=True):
            card_ax.text(value + 0.004, y_pos, f'{value:.2f}', va='center', ha='left', fontsize=4.0, color='#475467')
        card_ax.set_yticks(ypos, labels=['Control', 'PD'])
        card_ax.set_xlim(0.0, 0.14)
        card_ax.set_xticks([0.0, 0.05, 0.10])
        card_ax.tick_params(axis='x', labelsize=4.0, pad=1, length=0)
        card_ax.tick_params(axis='y', labelsize=4.0, length=0)
        for spine in card_ax.spines.values():
            spine.set_visible(False)
        card_ax.grid(axis='x', linestyle='--', linewidth=0.45, color=blend_with_white(FIGURE6_COMPONENT_COLORS['highlight'], 0.40))

    holdout_ax = host_ax.inset_axes([0.10, 0.08, 0.80, 0.22])
    y_pos = np.arange(holdout.shape[0], dtype=float)
    values = holdout['mean_holdout_pd_minus_control'].to_numpy(dtype=float)
    colors = [FIGURE6_BRANCH_COLORS['context'] if value > 0 else FIGURE6_BRANCH_COLORS['canonical'] for value in values]
    holdout_ax.barh(y_pos, values, color=colors, alpha=0.88, height=0.42)
    holdout_ax.axvline(0.0, linewidth=0.8, color=FIGURE6_COMPONENT_COLORS['highlight'])
    holdout_ax.set_yticks(y_pos, labels=holdout['holdout_source_id'].tolist())
    holdout_ax.set_xlabel('Holdout PD-minus-control', fontsize=4.5, labelpad=2)
    holdout_ax.tick_params(axis='x', labelsize=4.1, pad=1, length=0)
    holdout_ax.tick_params(axis='y', labelsize=3.8, length=0)
    holdout_ax.grid(axis='x', linestyle='--', linewidth=0.45, color=blend_with_white(FIGURE6_COMPONENT_COLORS['highlight'], 0.40))
    holdout_ax.set_xlim(-0.08, 1.55)
    holdout_ax.invert_yaxis()
    for value, y_entry, matched in zip(values, y_pos, holdout['direction_match'].tolist(), strict=True):
        align = 'left' if value >= 0 else 'right'
        x_text = value + 0.03 if value >= 0 else value - 0.03
        holdout_ax.text(x_text, y_entry, 'match' if matched else 'flip', ha=align, va='center', fontsize=4.0, color='#475467')
    holdout_ax.text(0.0, 1.10, 'Program holdout by source', transform=holdout_ax.transAxes, ha='left', va='bottom', fontsize=4.5, color='#667085')
    for spine in holdout_ax.spines.values():
        spine.set_visible(False)


def status_heatmap(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    order = ['Hold', 'Split', 'Weak', 'NA', 'Moderate', 'Keep', 'Strong']
    palette = [
        PAPER_RED,
        blend_with_white(PAPER_RED, 0.22),
        PAPER_BLUE_SOFT,
        blend_with_white(PAPER_BLUE_LIGHT, 0.84),
        PAPER_TEAL_LIGHT,
        PAPER_TEAL,
        PAPER_BLUE,
    ]
    cmap = ListedColormap(palette)
    code = {name: idx for idx, name in enumerate(order)}
    numeric = frame.replace(code).to_numpy(dtype=float)
    ax.imshow(numeric, aspect='auto', cmap=cmap, vmin=0, vmax=len(order) - 1)
    ax.set_xticks(np.arange(frame.shape[1]), labels=frame.columns.tolist())
    ax.set_yticks(np.arange(frame.shape[0]), labels=frame.index.tolist())
    ax.tick_params(axis='x', labelsize=4.8, length=0)
    ax.tick_params(axis='y', labelsize=4.8, length=0)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            value = str(frame.iat[i, j])
            ax.text(
                j,
                i,
                value,
                ha='center',
                va='center',
                fontsize=4.8,
                color='white' if value in {'Hold', 'Split', 'Keep', 'Strong'} else '#243B53',
                fontweight='bold' if value != 'NA' else 'normal',
            )
    set_left_title(ax, title)
    for spine in ax.spines.values():
        spine.set_visible(False)


@lru_cache(maxsize=1)
def figure1_source_summary() -> pd.DataFrame:
    mgda = step2_graph_manifest().loc[step2_graph_manifest()['task_name'] == 'microglia_to_dopaminergic_neuron'].copy()
    sequencing = (
        mgda.groupby('source_id', as_index=False)
        .agg(
            n_units=('split_group_id', 'nunique'),
            n_graphs=('graph_unit_id', 'nunique'),
            control=('disease_group', lambda s: int((s == 'control').sum())),
            pd=('disease_group', lambda s: int((s == 'pd').sum())),
            lbd=('disease_group', lambda s: int((s == 'lbd').sum())),
        )
    )

    holdout_157783 = step2_holdout_patient_counts().copy()
    holdout_157783['disease_group'] = np.where(holdout_157783['patient'].astype(str).str.upper().str.startswith('PD'), 'pd', 'control')
    holdout_157783 = holdout_157783.loc[holdout_157783['eligible_mg_da']].copy()
    sequencing = pd.concat(
        [
            sequencing,
            pd.DataFrame(
                [
                    {
                        'source_id': 'GSE157783',
                        'n_units': int(holdout_157783.shape[0]),
                        'n_graphs': int(holdout_157783.shape[0]),
                        'control': int((holdout_157783['disease_group'] == 'control').sum()),
                        'pd': int((holdout_157783['disease_group'] == 'pd').sum()),
                        'lbd': 0,
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        'source_id': 'GSE253462',
                        'n_units': int(package_gse253462_donor_counts()['n_donors'].sum()),
                        'n_graphs': np.nan,
                        'control': int(package_gse253462_donor_counts().loc[package_gse253462_donor_counts()['condition_standardized'] == 'control', 'n_donors'].sum()),
                        'pd': int(package_gse253462_donor_counts().loc[package_gse253462_donor_counts()['condition_standardized'] == 'pd', 'n_donors'].sum()),
                        'lbd': 0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    meta = (
        package_pd_definitive_manifest()
        .groupby('source_id', as_index=False)
        .agg(
            tissue_scope=('tissue_scope', lambda s: '|'.join(sorted(set(map(str, s))))),
            modality=('modality', lambda s: '|'.join(sorted(set(map(str, s))))),
        )
    )
    frame = sequencing.merge(meta, on='source_id', how='left')
    frame['group'] = np.where(frame['source_id'].isin(DISCOVERY_ATLAS_SOURCES), 'Discovery atlas', 'Outside atlas')

    def simplify_tissue(value: object) -> str:
        text = str(value).lower()
        if 'pars compacta' in text:
            return 'SNpc'
        if 'midbrain' in text:
            return 'Midbrain'
        if 'substantia nigra' in text:
            return 'Sub. nigra'
        return 'Other'

    def simplify_modality(value: object) -> str:
        text = str(value).lower()
        if 'multiome' in text:
            return 'Multiome'
        if 'matrix' in text or 'triplet' in text:
            return 'Triplets'
        if 'snrna' in text:
            return 'snRNA'
        return 'Other'

    frame['tissue_label'] = frame['tissue_scope'].map(simplify_tissue)
    frame['modality_label'] = frame['modality'].map(simplify_modality)
    return frame


@lru_cache(maxsize=1)
def figure1_role_matrix() -> pd.DataFrame:
    source_summary = figure1_source_summary()
    visium = package_primary_visium_manifest()

    role_order = ['Discovery atlas', 'Outside atlas', 'Primary spatial', 'Genetics']
    row_order = [
        'Independent units',
        'Control',
        'PD',
        'LBD',
        'SNpc',
        'Sub. nigra',
        'Midbrain',
        'snRNA',
        'Multiome',
        'Triplets',
        'Spatial',
        'Genetics',
    ]

    matrix = pd.DataFrame(0.0, index=row_order, columns=role_order)

    for role in ['Discovery atlas', 'Outside atlas']:
        sub = source_summary.loc[source_summary['group'] == role].copy()
        matrix.loc['Independent units', role] = float(sub['n_units'].sum())
        matrix.loc['Control', role] = float(sub['control'].sum())
        matrix.loc['PD', role] = float(sub['pd'].sum())
        matrix.loc['LBD', role] = float(sub['lbd'].sum())
        for label in ['SNpc', 'Sub. nigra', 'Midbrain']:
            matrix.loc[label, role] = float(sub.loc[sub['tissue_label'] == label, 'n_units'].sum())
        for label in ['snRNA', 'Multiome', 'Triplets']:
            matrix.loc[label, role] = float(sub.loc[sub['modality_label'] == label, 'n_units'].sum())

    matrix.loc['Independent units', 'Primary spatial'] = float(visium['section_id'].nunique())
    matrix.loc['Control', 'Primary spatial'] = float((visium['disease_group'] == 'control').sum())
    matrix.loc['PD', 'Primary spatial'] = float((visium['disease_group'] == 'pd').sum())
    matrix.loc['SNpc', 'Primary spatial'] = float(visium['section_id'].nunique())
    matrix.loc['Spatial', 'Primary spatial'] = float(visium['section_id'].nunique())

    matrix.loc['Independent units', 'Genetics'] = 3.0
    matrix.loc['Genetics', 'Genetics'] = 3.0
    return matrix


def draw_figure1_bubble_grid(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    column_colors: dict[str, str],
    row_groups: list[tuple[str, int, int]] | None = None,
    annotate_threshold: float = 1.0,
    max_size: float = 1500.0,
) -> None:
    n_rows, n_cols = frame.shape
    ax.set_xlim(-0.55, n_cols - 0.45)
    ax.set_ylim(n_rows - 0.5, -0.5)

    if row_groups:
        for idx, (_, start, end) in enumerate(row_groups):
            if idx % 2 == 0:
                ax.axhspan(start - 0.5, end + 0.5, color='#F8FAFC', zorder=0)
            ax.axhline(end + 0.5, color='#D8DEE7', linewidth=0.8, zorder=1)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax.add_patch(
                mpl.patches.Rectangle(
                    (col_idx - 0.47, row_idx - 0.38),
                    0.94,
                    0.76,
                    facecolor='white',
                    edgecolor='#E5E7EB',
                    linewidth=0.6,
                    zorder=1,
                )
            )

    max_value = float(frame.to_numpy().max()) if frame.size else 1.0
    for col_idx, column in enumerate(frame.columns):
        for row_idx, row_name in enumerate(frame.index):
            value = float(frame.iat[row_idx, col_idx])
            if value <= 0:
                continue
            size = max(110.0, max_size * value / max(max_value, 1.0))
            ax.scatter(
                col_idx,
                row_idx,
                s=size,
                color=column_colors[column],
                alpha=0.74,
                edgecolors='white',
                linewidths=0.8,
                zorder=3,
            )
            if value >= annotate_threshold:
                ax.text(
                    col_idx,
                    row_idx,
                    f'{int(value)}',
                    ha='center',
                    va='center',
                    fontsize=4.7,
                    color='#0F172A',
                    fontweight='bold',
                    zorder=4,
                )

    ax.set_xticks(np.arange(n_cols), labels=frame.columns.tolist())
    ax.set_yticks(np.arange(n_rows), labels=frame.index.tolist())
    ax.tick_params(axis='x', labelsize=4.8, rotation=0, length=0)
    ax.tick_params(axis='y', labelsize=4.7, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if row_groups:
        for title, start, end in row_groups:
            if start == end:
                continue
            center_y = 1.0 - (((start + end) / 2) + 0.5) / n_rows
            ax.text(
                -0.18,
                center_y,
                title,
                transform=ax.transAxes,
                ha='left',
                va='center',
                fontsize=4.2,
                color='#667085',
                fontweight='semibold',
                clip_on=False,
            )


def draw_figure1_source_chip(
    host_ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    count: int,
    color: str,
    label_fontsize: float = 4.4,
    count_fontsize: float = 4.8,
    label_xpad: float = 0.02,
    count_xpad: float = 0.02,
) -> None:
    add_round_box(
        host_ax,
        x,
        y,
        width,
        height,
        facecolor=blend_with_white(color, 0.84),
        edgecolor=blend_with_white(color, 0.20),
        linewidth=0.8,
        radius=0.020,
    )
    host_ax.text(
        x + label_xpad,
        y + height * 0.60,
        label,
        transform=host_ax.transAxes,
        fontsize=label_fontsize,
        fontweight='semibold',
        color='#1F2937',
    )
    host_ax.text(
        x + width - count_xpad,
        y + height * 0.60,
        f'{count}',
        transform=host_ax.transAxes,
        ha='right',
        fontsize=count_fontsize,
        fontweight='bold',
        color=blend_with_black(color, 0.12),
    )


def draw_compact_stacked_hbars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    colors: dict[str, str],
    title: str,
    overlay: pd.Series | None = None,
    overlay_label: str | None = None,
    x_label: str | None = None,
    legend_ncol: int = 3,
    value_label_min: float = 4.0,
) -> None:
    plot_df = frame.copy()
    ypos = np.arange(plot_df.shape[0], dtype=float)
    left = np.zeros(plot_df.shape[0], dtype=float)

    for column in plot_df.columns:
        values = plot_df[column].to_numpy(dtype=float)
        ax.barh(ypos, values, left=left, color=colors[column], height=0.58, label=column)
        for idx, value in enumerate(values):
            if value >= value_label_min:
                ax.text(left[idx] + value / 2.0, ypos[idx], f'{int(value)}', ha='center', va='center', fontsize=4.0, color=text_color_for_fill(colors[column]), fontweight='bold')
        left += values

    overlay_values = None
    if overlay is not None:
        overlay_values = overlay.reindex(plot_df.index).to_numpy(dtype=float)
        ax.scatter(overlay_values, ypos, s=24, facecolors='white', edgecolors='#344054', linewidths=0.8, zorder=4, label=overlay_label)

    totals = plot_df.sum(axis=1).to_numpy(dtype=float)
    display_end = np.maximum(totals, overlay_values) if overlay_values is not None else totals
    max_end = float(display_end.max()) if display_end.size else 1.0
    pad = max(0.7, 0.05 * max_end)
    for idx, total in enumerate(display_end):
        ax.text(total + pad * 0.35, ypos[idx], f'{int(total)}', va='center', fontsize=4.2, color='#475467', fontweight='semibold')

    ax.set_yticks(ypos, labels=plot_df.index.tolist())
    ax.invert_yaxis()
    ax.set_title(title, loc='left', fontsize=5.6, pad=3.0)
    if x_label:
        ax.set_xlabel(x_label, fontsize=4.6)
    ax.tick_params(axis='x', labelsize=4.2)
    ax.tick_params(axis='y', labelsize=4.2, length=0)
    ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')
    ax.set_axisbelow(True)
    ax.set_xlim(0.0, max_end + pad * 2.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, fontsize=3.5, ncol=legend_ncol, loc='lower right', bbox_to_anchor=(1.0, 1.01), handlelength=1.1, columnspacing=0.8, handletextpad=0.4)


def figure1_architecture_panel(host_ax: plt.Axes) -> None:
    figure5_panel_label(host_ax, 'a', x=-0.02)
    host_ax.axis('off')
    floating_title(host_ax, 'Atlas-only discovery with orthogonal validation', y=1.03)

    source_summary = figure1_source_summary().set_index('source_id')
    discovery_total = int(source_summary.loc[DISCOVERY_ATLAS_SOURCES, 'n_units'].sum())
    outside_sources = AUXILIARY_SOURCES + HOLDOUT_SOURCES
    outside_total = int(source_summary.loc[outside_sources, 'n_units'].sum())
    holdout_total = int(source_summary.loc[HOLDOUT_SOURCES, 'n_units'].sum())

    add_round_box(host_ax, 0.02, 0.46, 0.38, 0.34, facecolor=blend_with_white(PAPER_BLUE, 0.93), edgecolor=blend_with_white(PAPER_BLUE, 0.15), linewidth=0.95, radius=0.030)
    host_ax.text(0.045, 0.74, 'Discovery atlas', transform=host_ax.transAxes, fontsize=6.1, fontweight='semibold', color='#1F2937')
    host_ax.text(0.385, 0.74, f'4 sources | n={discovery_total}', transform=host_ax.transAxes, ha='right', fontsize=4.5, color='#667085')
    discovery_chips = [
        ('CELLxGENE', 'b0f0b447-ac37-45b0-b1bf-5c0b7d871120'),
        ('GSE193688', 'GSE193688'),
        ('GSE140231', 'GSE140231'),
        ('GSE126836', 'GSE126836'),
    ]
    chip_positions = [(0.05, 0.625), (0.215, 0.625), (0.05, 0.540), (0.215, 0.540)]
    for (label, source_id), (x0, y0) in zip(discovery_chips, chip_positions, strict=True):
        draw_figure1_source_chip(
            host_ax,
            x=x0,
            y=y0,
            width=0.125,
            height=0.060,
            label=label,
            count=int(source_summary.loc[source_id, 'n_units']),
            color=SOURCE_COLORS[source_id],
            label_fontsize=4.4,
            count_fontsize=4.7,
            label_xpad=0.016,
            count_xpad=0.014,
        )

    add_round_box(host_ax, 0.02, 0.13, 0.38, 0.24, facecolor=blend_with_white(PAPER_RED, 0.94), edgecolor=blend_with_white(PAPER_RED, 0.15), linewidth=0.95, radius=0.030)
    host_ax.text(0.045, 0.31, 'Outside atlas', transform=host_ax.transAxes, fontsize=6.1, fontweight='semibold', color='#1F2937')
    host_ax.text(0.385, 0.31, f'3 sources | n={outside_total}', transform=host_ax.transAxes, ha='right', fontsize=4.5, color='#667085')
    outside_chips = [
        ('148434', 'GSE148434', 0.060, 3.7),
        ('157783', 'GSE157783', 0.060, 3.7),
        ('253462', 'GSE253462', 0.034, 2.9),
    ]
    x_positions = [0.05, 0.155, 0.265]
    for (label, source_id, width, label_fontsize), x0 in zip(outside_chips, x_positions, strict=True):
        draw_figure1_source_chip(
            host_ax,
            x=x0,
            y=0.215,
            width=width + 0.016,
            height=0.060,
            label=label,
            count=int(source_summary.loc[source_id, 'n_units']),
            color=SOURCE_COLORS[source_id],
            label_fontsize=label_fontsize + 0.2,
            count_fontsize=4.2 if source_id != 'GSE253462' else 3.8,
            label_xpad=0.010,
            count_xpad=0.010,
        )

    add_round_box(host_ax, 0.49, 0.585, 0.10, 0.11, facecolor=blend_with_white(PAPER_TEAL, 0.88), edgecolor=blend_with_white(PAPER_TEAL, 0.16), linewidth=0.95, radius=0.024)
    host_ax.text(0.507, 0.646, 'Step 2', transform=host_ax.transAxes, fontsize=5.0, fontweight='semibold', color='#1F2937')
    host_ax.text(0.507, 0.612, 'ranking', transform=host_ax.transAxes, fontsize=3.9, color='#667085')

    add_round_box(host_ax, 0.49, 0.395, 0.10, 0.11, facecolor=blend_with_white(PAPER_BLUE_LIGHT, 0.86), edgecolor=blend_with_white(PAPER_BLUE_LIGHT, 0.16), linewidth=0.95, radius=0.024)
    host_ax.text(0.507, 0.456, 'Step 3', transform=host_ax.transAxes, fontsize=5.0, fontweight='semibold', color='#1F2937')
    host_ax.text(0.507, 0.422, 'refine', transform=host_ax.transAxes, fontsize=3.9, color='#667085')

    right_boxes = [
        ('Internal holdout', f'2 sources\nn={holdout_total}', 0.72, 0.625, PAPER_RED),
        ('Spatial validation', '10 Visium\n+ GeoMx', 0.72, 0.435, PAPER_BLUE),
        ('PD genetics', '3 resources', 0.72, 0.245, PAPER_TEAL),
    ]
    for title, subtitle, x0, y0, color in right_boxes:
        add_round_box(host_ax, x0, y0, 0.18, 0.11, facecolor=blend_with_white(color, 0.90), edgecolor=blend_with_white(color, 0.16), linewidth=0.95, radius=0.024)
        host_ax.text(x0 + 0.014, y0 + 0.066, title, transform=host_ax.transAxes, fontsize=4.5, fontweight='semibold', color='#1F2937')
        host_ax.text(x0 + 0.014, y0 + 0.021, subtitle, transform=host_ax.transAxes, fontsize=3.8, color='#667085', linespacing=1.0)

    for start, end, color, rad in [
        ((0.40, 0.64), (0.49, 0.640), PAPER_BLUE, 0.05),
        ((0.40, 0.62), (0.49, 0.450), PAPER_BLUE, -0.12),
        ((0.40, 0.25), (0.72, 0.680), PAPER_RED, 0.18),
        ((0.40, 0.25), (0.72, 0.490), PAPER_RED, 0.04),
        ((0.59, 0.640), (0.72, 0.680), PAPER_BLUE, 0.05),
        ((0.59, 0.450), (0.72, 0.490), PAPER_BLUE, 0.02),
        ((0.59, 0.450), (0.72, 0.300), PAPER_BLUE, -0.08),
    ]:
        figure6_flow_arrow(host_ax, start, end, color=color, rad=rad, alpha=0.84)

    host_ax.text(0.50, 0.76, 'Only atlas cohorts feed discovery', transform=host_ax.transAxes, fontsize=4.3, color='#667085')


def figure1_resource_summary_panel(host_ax: plt.Axes) -> None:
    figure5_panel_label(host_ax, 'b', x=-0.02)
    host_ax.axis('off')
    floating_title(host_ax, 'Study backbone counts', y=1.03)

    source_summary = figure1_source_summary()
    visium = package_primary_visium_manifest()

    role_colors = {
        'Discovery atlas': PAPER_BLUE,
        'Outside atlas': PAPER_RED,
        'Primary spatial': PAPER_BLUE_LIGHT,
        'Genetics': PAPER_TEAL,
    }
    role_totals = pd.Series(
        {
            'Discovery atlas': float(source_summary.loc[source_summary['group'] == 'Discovery atlas', 'n_units'].sum()),
            'Outside atlas': float(source_summary.loc[source_summary['group'] == 'Outside atlas', 'n_units'].sum()),
            'Primary spatial': float(visium['section_id'].nunique()),
            'Genetics': 3.0,
        }
    )

    disease_frame = pd.DataFrame(
        {
            'Control': [
                float(source_summary.loc[source_summary['group'] == 'Discovery atlas', 'control'].sum()),
                float(source_summary.loc[source_summary['group'] == 'Outside atlas', 'control'].sum()),
                float((visium['disease_group'] == 'control').sum()),
            ],
            'LBD': [
                float(source_summary.loc[source_summary['group'] == 'Discovery atlas', 'lbd'].sum()),
                float(source_summary.loc[source_summary['group'] == 'Outside atlas', 'lbd'].sum()),
                0.0,
            ],
            'PD': [
                float(source_summary.loc[source_summary['group'] == 'Discovery atlas', 'pd'].sum()),
                float(source_summary.loc[source_summary['group'] == 'Outside atlas', 'pd'].sum()),
                float((visium['disease_group'] == 'pd').sum()),
            ],
        },
        index=['Discovery atlas', 'Outside atlas', 'Primary spatial'],
    )

    tissue_frame = pd.DataFrame(
        {
            'SNpc': [
                float(source_summary.loc[(source_summary['group'] == 'Discovery atlas') & (source_summary['tissue_label'] == 'SNpc'), 'n_units'].sum()),
                float(source_summary.loc[(source_summary['group'] == 'Outside atlas') & (source_summary['tissue_label'] == 'SNpc'), 'n_units'].sum()),
                float(visium['section_id'].nunique()),
            ],
            'Sub. nigra': [
                float(source_summary.loc[(source_summary['group'] == 'Discovery atlas') & (source_summary['tissue_label'] == 'Sub. nigra'), 'n_units'].sum()),
                float(source_summary.loc[(source_summary['group'] == 'Outside atlas') & (source_summary['tissue_label'] == 'Sub. nigra'), 'n_units'].sum()),
                0.0,
            ],
            'Midbrain': [
                float(source_summary.loc[(source_summary['group'] == 'Discovery atlas') & (source_summary['tissue_label'] == 'Midbrain'), 'n_units'].sum()),
                float(source_summary.loc[(source_summary['group'] == 'Outside atlas') & (source_summary['tissue_label'] == 'Midbrain'), 'n_units'].sum()),
                0.0,
            ],
        },
        index=['Discovery atlas', 'Outside atlas', 'Primary spatial'],
    )

    modality_frame = pd.DataFrame(
        {
            'snRNA': [
                float(source_summary.loc[(source_summary['group'] == 'Discovery atlas') & (source_summary['modality_label'] == 'snRNA'), 'n_units'].sum()),
                float(source_summary.loc[(source_summary['group'] == 'Outside atlas') & (source_summary['modality_label'] == 'snRNA'), 'n_units'].sum()),
                0.0,
                0.0,
            ],
            'Multiome': [
                float(source_summary.loc[(source_summary['group'] == 'Discovery atlas') & (source_summary['modality_label'] == 'Multiome'), 'n_units'].sum()),
                float(source_summary.loc[(source_summary['group'] == 'Outside atlas') & (source_summary['modality_label'] == 'Multiome'), 'n_units'].sum()),
                0.0,
                0.0,
            ],
            'Triplets': [
                float(source_summary.loc[(source_summary['group'] == 'Discovery atlas') & (source_summary['modality_label'] == 'Triplets'), 'n_units'].sum()),
                float(source_summary.loc[(source_summary['group'] == 'Outside atlas') & (source_summary['modality_label'] == 'Triplets'), 'n_units'].sum()),
                0.0,
                0.0,
            ],
            'Spatial': [0.0, 0.0, float(visium['section_id'].nunique()), 0.0],
            'Genetics': [0.0, 0.0, 0.0, 3.0],
        },
        index=['Discovery atlas', 'Outside atlas', 'Primary spatial', 'Genetics'],
    )

    total_ax = host_ax.inset_axes([0.02, 0.56, 0.44, 0.30])
    disease_ax = host_ax.inset_axes([0.52, 0.56, 0.44, 0.30])
    tissue_ax = host_ax.inset_axes([0.02, 0.10, 0.44, 0.30])
    modality_ax = host_ax.inset_axes([0.52, 0.10, 0.44, 0.30])

    ypos = np.arange(role_totals.shape[0], dtype=float)
    total_ax.barh(ypos, role_totals.to_numpy(dtype=float), color=[role_colors[item] for item in role_totals.index], height=0.58)
    for idx, value in enumerate(role_totals.to_numpy(dtype=float)):
        total_ax.text(value + 1.2, ypos[idx], f'{int(value)}', va='center', fontsize=4.4, color='#475467', fontweight='semibold')
    total_ax.set_yticks(ypos, labels=role_totals.index.tolist())
    total_ax.invert_yaxis()
    total_ax.set_title('Independent units by role', loc='left', fontsize=5.6, pad=3.0)
    total_ax.tick_params(axis='x', labelsize=4.2)
    total_ax.tick_params(axis='y', labelsize=4.2, length=0)
    total_ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')
    total_ax.set_axisbelow(True)
    total_ax.set_xlim(0.0, float(role_totals.max()) * 1.22)
    for spine in total_ax.spines.values():
        spine.set_visible(False)

    draw_compact_stacked_hbars(disease_ax, disease_frame, colors={'Control': PAPER_BLUE, 'LBD': PAPER_TEAL_LIGHT, 'PD': PAPER_RED}, title='Disease mix', legend_ncol=3)
    draw_compact_stacked_hbars(tissue_ax, tissue_frame, colors={'SNpc': PAPER_BLUE, 'Sub. nigra': PAPER_RED, 'Midbrain': PAPER_TEAL}, title='Tissue origin', legend_ncol=3)
    draw_compact_stacked_hbars(modality_ax, modality_frame, colors={'snRNA': PAPER_BLUE, 'Multiome': PAPER_TEAL, 'Triplets': PAPER_BLUE_LIGHT, 'Spatial': PAPER_RED, 'Genetics': PAPER_TEAL_LIGHT}, title='Modality / layer', legend_ncol=3)
    host_ax.text(0.98, 0.02, 'Sequencing roles use donor units; spatial uses sections; genetics uses resources.', transform=host_ax.transAxes, ha='right', va='bottom', fontsize=4.0, color='#667085')


def figure1_task_coverage_panel(ax: plt.Axes) -> None:
    frame = (
        step2_graph_manifest()
        .loc[step2_graph_manifest()['task_name'] == 'microglia_to_dopaminergic_neuron']
        .groupby(['source_id', 'cohort_name'], as_index=False)
        .agg(n_donors=('split_group_id', 'nunique'), n_graphs=('graph_unit_id', 'nunique'))
    )
    row_order = DISCOVERY_ATLAS_SOURCES + AUXILIARY_SOURCES
    col_order = ['control_known_age', 'lbd_bridge', 'pd_known_age', 'unknown_age_auxiliary']
    donor_mat = frame.pivot(index='source_id', columns='cohort_name', values='n_donors').reindex(index=row_order, columns=col_order).fillna(0.0)
    graph_totals = frame.groupby('source_id')['n_graphs'].sum().reindex(row_order).fillna(0.0)

    plot_df = donor_mat.rename(index={item: pretty_source(item) for item in row_order}, columns={item: pretty_cohort(item) for item in col_order})
    graph_totals.index = [pretty_source(item) for item in row_order]

    ax.axhspan(-0.5, 3.5, color='#F8FAFC', zorder=0)
    ax.axhspan(3.5, 4.5, color=blend_with_white(PAPER_RED, 0.95), zorder=0)
    draw_compact_stacked_hbars(
        ax,
        plot_df,
        colors={'Control': PAPER_BLUE, 'LBD bridge': PAPER_TEAL, 'PD': PAPER_RED, 'Auxiliary': PAPER_BLUE_LIGHT},
        title='Microglia -> DA donor coverage by source',
        overlay=graph_totals,
        overlay_label='Graph units total',
        x_label='Donors',
        legend_ncol=3,
        value_label_min=3.0,
    )
    figure5_panel_label(ax, 'c', x=-0.10, y=1.05)
    ax.set_ylabel('')
    ax.text(0.99, -0.18, 'Open circle marks graph-unit total; in this task the graph total closely tracks donor-grouped coverage.', transform=ax.transAxes, ha='right', va='top', fontsize=4.0, color='#667085')


def figure1_funnel_panel(host_ax: plt.Axes) -> None:
    figure5_panel_label(host_ax, 'd', x=-0.02)
    host_ax.axis('off')
    floating_title(host_ax, 'Candidate funnel and final Step 4 outcomes', y=1.03)

    step2_n = int(step2_step3_candidates().shape[0])
    step3_n = int(package_step3_top10().shape[0])
    step4_n = int(package_step4_results_table().shape[0])
    results = package_step4_results_table().copy().set_index('program_id')

    funnel_ax = host_ax.inset_axes([0.03, 0.12, 0.43, 0.74])
    score_ax = host_ax.inset_axes([0.51, 0.15, 0.45, 0.68])
    funnel_ax.axis('off')
    funnel_ax.set_xlim(0.0, 1.0)
    funnel_ax.set_ylim(0.0, 1.0)

    stages = [
        ('Step 2', 'ranked candidates', step2_n, 0.78, 0.84, PAPER_BLUE),
        ('Step 3', 'refined programs', step3_n, 0.50, 0.52, PAPER_TEAL),
        ('Step 4', 'validation shortlist', step4_n, 0.22, 0.28, PAPER_BLUE_LIGHT),
    ]
    height = 0.14
    for title, subtitle, count, center_y, width, color in stages:
        x0 = 0.50 - width / 2.0
        add_round_box(funnel_ax, x0, center_y - height / 2.0, width, height, facecolor=blend_with_white(color, 0.87), edgecolor=blend_with_white(color, 0.18), linewidth=0.9, radius=0.025)
        funnel_ax.text(x0 + 0.03, center_y + 0.020, title, transform=funnel_ax.transAxes, fontsize=4.9, fontweight='semibold', color='#1F2937')
        funnel_ax.text(x0 + 0.03, center_y - 0.025, subtitle, transform=funnel_ax.transAxes, fontsize=3.9, color='#667085')
        funnel_ax.text(x0 + width - 0.04, center_y, f'{count}', transform=funnel_ax.transAxes, ha='right', va='center', fontsize=10.0, fontweight='bold', color=color)

    figure6_flow_arrow(funnel_ax, (0.50, 0.69), (0.50, 0.58), color=PAPER_BLUE, rad=0.0, alpha=0.90)
    figure6_flow_arrow(funnel_ax, (0.50, 0.41), (0.50, 0.30), color=PAPER_TEAL, rad=0.0, alpha=0.90)
    funnel_ax.text(0.68, 0.62, f'{step2_n - step3_n} filtered out', transform=funnel_ax.transAxes, fontsize=4.0, color='#667085')
    funnel_ax.text(0.62, 0.34, f'{step3_n - step4_n} filtered out', transform=funnel_ax.transAxes, fontsize=4.0, color='#667085')

    score_order = [
        ('apoe_lipoprotein_receptor_signaling', 'APOE'),
        ('slit_robo_guidance_canonical', 'Canonical SLIT-ROBO'),
        ('slit_robo_guidance_context', 'Guidance context'),
    ]
    plot_df = pd.DataFrame(
        [
            {
                'Program': label,
                'Score': float(results.loc[program_id, 'final_step4_score']),
                'Decision': str(results.loc[program_id, 'promotion_decision']).replace('_', ' '),
                'Color': PAPER_BLUE if program_id == 'apoe_lipoprotein_receptor_signaling' else (PAPER_TEAL if program_id == 'slit_robo_guidance_canonical' else PAPER_RED),
            }
            for program_id, label in score_order
        ]
    )
    ypos = np.arange(plot_df.shape[0], dtype=float)
    score_ax.barh(ypos, plot_df['Score'], color=plot_df['Color'], height=0.56)
    for idx, row in plot_df.iterrows():
        score_ax.text(float(row['Score']) + 0.014, ypos[idx], f"{row['Score']:.2f}", va='center', fontsize=4.4, color='#1F2937', fontweight='bold')
        score_ax.text(0.82, ypos[idx], str(row['Decision']).replace(' candidate', '').replace('spatial validation', 'validation'), va='center', ha='right', fontsize=3.7, color='#667085')
    score_ax.set_yticks(ypos, labels=plot_df['Program'].tolist())
    score_ax.invert_yaxis()
    score_ax.set_title('Step 4 programs', loc='left', fontsize=5.6, pad=3.0)
    score_ax.set_xlabel('Final Step 4 score')
    score_ax.tick_params(axis='x', labelsize=4.2)
    score_ax.tick_params(axis='y', labelsize=3.8, pad=1.0, length=0)
    score_ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')
    score_ax.set_axisbelow(True)
    score_ax.set_xlim(0.0, 0.84)
    for spine in score_ax.spines.values():
        spine.set_visible(False)


def figure2_roc_panel(ax: plt.Axes) -> None:
    donor = figure2_donor_level_predictions()
    figure5_panel_label(ax, 'a', x=-0.08)
    set_left_title(ax, 'Donor-grouped ROC across folds')
    for idx, (fold, sub) in enumerate(donor.groupby('fold', sort=True)):
        fpr, tpr, _ = roc_curve(sub['disease_target'], sub['pd_probability'])
        fold_auc = roc_auc_score(sub['disease_target'], sub['pd_probability'])
        ax.plot(fpr, tpr, color=FIGURE2_FOLD_PALETTE[idx % len(FIGURE2_FOLD_PALETTE)], linewidth=1.4, label=f'Fold {fold} | AUROC {fold_auc:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='#98A2B3', linewidth=0.8)
    ax.set_xlabel('False-positive rate')
    ax.set_ylabel('True-positive rate')
    ax.legend(frameon=False, fontsize=4.3, loc='lower right')
    ax.tick_params(labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def figure2_pr_panel(ax: plt.Axes) -> None:
    donor = figure2_donor_level_predictions()
    figure5_panel_label(ax, 'b', x=-0.08)
    set_left_title(ax, 'Precision-recall across folds')
    for idx, (fold, sub) in enumerate(donor.groupby('fold', sort=True)):
        precision, recall, _ = precision_recall_curve(sub['disease_target'], sub['pd_probability'])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=FIGURE2_FOLD_PALETTE[idx % len(FIGURE2_FOLD_PALETTE)], linewidth=1.4, label=f'Fold {fold} | AUPR {pr_auc:.2f}')
    prevalence = donor['disease_target'].mean()
    ax.axhline(prevalence, linestyle='--', color='#98A2B3', linewidth=0.8)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(frameon=False, fontsize=4.3, loc='lower right')
    ax.tick_params(labelsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def figure2_calibration_panel(host_ax: plt.Axes) -> None:
    figure5_panel_label(host_ax, 'c', x=-0.02)
    host_ax.axis('off')
    floating_title(host_ax, 'Calibration and decision parameters', y=1.03)
    donor = figure2_donor_level_predictions()

    calib_ax = host_ax.inset_axes([0.00, 0.14, 0.47, 0.76])
    frac_raw, mean_raw = calibration_curve(donor['disease_target'], donor['pd_probability_raw'], n_bins=4, strategy='quantile')
    frac_cal, mean_cal = calibration_curve(donor['disease_target'], donor['pd_probability'], n_bins=4, strategy='quantile')
    calib_ax.plot([0, 1], [0, 1], linestyle='--', color='#98A2B3', linewidth=0.8)
    calib_ax.plot(mean_raw, frac_raw, marker='o', linewidth=1.3, color=FIGURE2_CALIBRATION_COLORS['raw'], label='Raw')
    calib_ax.plot(mean_cal, frac_cal, marker='o', linewidth=1.3, color=FIGURE2_CALIBRATION_COLORS['calibrated'], label='Calibrated')
    calib_ax.set_xlabel('Predicted PD probability')
    calib_ax.set_ylabel('Observed PD fraction')
    calib_ax.legend(
        frameon=False,
        fontsize=3.8,
        loc='lower left',
        ncol=2,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.4,
        borderaxespad=0.3,
    )
    calib_ax.tick_params(labelsize=4.8)
    calib_ax.spines['top'].set_visible(False)
    calib_ax.spines['right'].set_visible(False)

    param_ax = host_ax.inset_axes([0.53, 0.14, 0.45, 0.76])
    fold_meta = donor.groupby('fold', as_index=False)[['decision_threshold', 'calibration_intercept', 'calibration_slope']].mean()
    param_positions = np.arange(fold_meta.shape[0], dtype=float)
    for column, style in FIGURE2_PARAMETER_STYLES.items():
        param_ax.scatter(param_positions, fold_meta[column], color=style['color'], marker=style['marker'], s=28, zorder=3)
        param_ax.plot(param_positions, fold_meta[column], color=style['color'], linewidth=1.0, alpha=0.85)
        param_ax.text(param_positions[-1] + 0.16, float(fold_meta[column].iloc[-1]), style['label'], va='center', fontsize=3.6, color=style['color'])
    param_ax.set_xticks(param_positions, labels=[f'Fold {int(item)}' for item in fold_meta['fold']])
    param_ax.set_ylabel('Parameter value')
    param_ax.tick_params(axis='x', labelsize=4.3, rotation=0)
    param_ax.tick_params(axis='y', labelsize=4.3)
    param_ax.spines['top'].set_visible(False)
    param_ax.spines['right'].set_visible(False)
    param_ax.grid(axis='y', linestyle='--', linewidth=0.45, color='#E4E7EC')
    param_ax.set_xlim(-0.1, param_positions[-1] + 0.80)


def figure2_metric_tables_panel(host_ax: plt.Axes) -> None:
    figure5_panel_label(host_ax, 'd', x=-0.02)
    host_ax.axis('off')
    floating_title(host_ax, 'Fold-wise and source-wise validation metrics', y=1.03)

    fold_df = package_pd_model_fold_metrics()[['fold', 'auroc', 'balanced_accuracy', 'brier_score', 'log_loss']].copy()
    fold_df = fold_df.rename(columns={'balanced_accuracy': 'bal_acc', 'brier_score': 'brier', 'log_loss': 'logloss'})
    fold_df['row_name'] = [f'Fold {int(item)}' for item in fold_df['fold']]

    donor = figure2_donor_level_predictions()
    source_rows = []
    for source_id, sub in donor.groupby('source_id'):
        if sub['disease_target'].nunique() < 2:
            continue
        source_rows.append(
            {
                'row_name': pretty_source(source_id),
                'auroc': roc_auc_score(sub['disease_target'], sub['pd_probability']),
                'bal_acc': balanced_accuracy_score(sub['disease_target'], (sub['pd_probability'] >= sub['decision_threshold']).astype(int)),
                'brier': brier_score_loss(sub['disease_target'], sub['pd_probability']),
                'logloss': log_loss(sub['disease_target'], sub['pd_probability'], labels=[0, 1]),
            }
        )
    source_df = pd.DataFrame(source_rows)

    def draw_table(ax: plt.Axes, frame: pd.DataFrame, title: str, cmap_colors: list[str], *, y_fontsize: float = 4.4) -> None:
        value_cols = ['auroc', 'bal_acc', 'brier', 'logloss']
        scaled = frame[value_cols].copy()
        scaled['brier'] = 1.0 - scaled['brier']
        scaled['logloss'] = 1.0 / (1.0 + scaled['logloss'])
        sns.heatmap(
            scaled.set_index(frame['row_name']),
            cmap=LinearSegmentedColormap.from_list(f'metric_table_{title.lower().replace(" ", "_")}', cmap_colors),
            annot=frame[value_cols].round(2).to_numpy(),
            fmt='',
            cbar=False,
            linewidths=0.8,
            linecolor='white',
            ax=ax,
            annot_kws={'fontsize': 4.3, 'color': '#243B53'},
        )
        ax.set_title(title, fontsize=5.2, loc='left', pad=3)
        ax.tick_params(axis='x', labelsize=4.4, rotation=0, length=0)
        ax.tick_params(axis='y', labelsize=y_fontsize, rotation=0, length=0, pad=1.0)
        ax.set_xlabel('')
        ax.set_ylabel('')

    draw_table(host_ax.inset_axes([0.00, 0.14, 0.40, 0.76]), fold_df, 'Folds', [blend_with_white(PAPER_COOL_PALE, 0.25), PAPER_COOL_LIGHT, PAPER_BLUE], y_fontsize=4.2)
    draw_table(host_ax.inset_axes([0.60, 0.14, 0.36, 0.76]), source_df, 'Validation sources', [blend_with_white(PAPER_WARM_PALE, 0.20), PAPER_WARM_SOFT, PAPER_RED], y_fontsize=3.6)


def figure2_baseline_panel(ax: plt.Axes) -> None:
    figure5_panel_label(ax, 'e', x=-0.08)
    baseline = step2_baseline_metrics()
    pd_row = baseline.loc[
        baseline['task_name'].eq('microglia_to_dopaminergic_neuron') & baseline['analysis'].eq('pd_deviation')
    ].iloc[0]
    gnn = package_pd_model_fold_metrics()[['auroc', 'balanced_accuracy']].mean()
    plot_df = pd.DataFrame(
        [
            {'model': 'Logistic', 'metric': 'AUROC', 'value': float(pd_row['auroc_logistic'])},
            {'model': 'Random forest', 'metric': 'AUROC', 'value': float(pd_row['auroc_random_forest'])},
            {'model': 'Graph model', 'metric': 'AUROC', 'value': float(gnn['auroc'])},
            {'model': 'Logistic', 'metric': 'Bal. acc.', 'value': float(pd_row['balanced_accuracy_logistic'])},
            {'model': 'Random forest', 'metric': 'Bal. acc.', 'value': float(pd_row['balanced_accuracy_random_forest'])},
            {'model': 'Graph model', 'metric': 'Bal. acc.', 'value': float(gnn['balanced_accuracy'])},
        ]
    )
    sns.barplot(
        data=plot_df,
        x='metric',
        y='value',
        hue='model',
        palette=FIGURE2_BASELINE_COLORS,
        ax=ax,
    )
    set_left_title(ax, 'Comparator baseline vs graph model')
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('Validation score')
    ax.set_xlabel('')
    ax.legend(frameon=False, fontsize=4.2, loc='upper right')
    ax.tick_params(axis='x', labelsize=4.8)
    ax.tick_params(axis='y', labelsize=4.8)
    ax.grid(axis='y', linestyle='--', linewidth=0.45, color='#E4E7EC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for patch in ax.patches:
        height = float(patch.get_height())
        if np.isfinite(height) and height > 0:
            x = float(patch.get_x() + patch.get_width() / 2.0)
            ax.text(x, min(height + 0.025, 1.02), f'{height:.2f}', ha='center', va='bottom', fontsize=4.0, color='#1F2937', fontweight='semibold')


def figure2_distribution_panel(host_ax: plt.Axes) -> None:
    figure5_panel_label(host_ax, 'f', x=-0.02)
    host_ax.axis('off')
    floating_title(host_ax, 'LBD bridge and holdout probability distributions', y=1.03)

    donor = figure2_donor_level_predictions()
    bridge = figure2_bridge_predictions()
    discovery = donor.assign(group=donor['disease_group'].map({'control': 'Discovery control', 'pd': 'Discovery PD'}))
    bridge = bridge.assign(group='LBD bridge')
    dist = pd.concat(
        [
            discovery[['group', 'pd_probability']],
            bridge[['group', 'pd_probability']],
        ],
        ignore_index=True,
    )
    order = ['Discovery control', 'LBD bridge', 'Discovery PD']
    left_ax = host_ax.inset_axes([0.00, 0.14, 0.50, 0.76])
    sns.boxplot(
        data=dist,
        x='group',
        y='pd_probability',
        hue='group',
        order=order,
        palette=FIGURE2_DISTRIBUTION_COLORS,
        dodge=False,
        ax=left_ax,
        showfliers=False,
        width=0.55,
        legend=False,
    )
    sns.stripplot(data=dist, x='group', y='pd_probability', order=order, color='white', edgecolor='#344054', linewidth=0.3, size=3.0, ax=left_ax)
    left_ax.set_ylabel('Calibrated PD probability')
    left_ax.set_xlabel('')
    left_ax.tick_params(axis='x', labelsize=4.2, rotation=12)
    left_ax.tick_params(axis='y', labelsize=4.6)
    left_ax.spines['top'].set_visible(False)
    left_ax.spines['right'].set_visible(False)

    right_ax = host_ax.inset_axes([0.58, 0.14, 0.39, 0.76])
    holdout = package_step4_decision_log()  # noop to keep package dependency warm
    baseline = step2_holdout_baseline_scores()[['split_group_id', 'disease_group', 'baseline_pd_probability']].copy()
    gnn = pd.read_csv(STEP2_DIR / '07_robustness' / 'step2_holdout_gnn_summary.csv').rename(columns={'graph_unit_id': 'split_group_id'})
    merged = baseline.merge(gnn[['split_group_id', 'disease_group', 'pd_probability']], on=['split_group_id', 'disease_group'], how='inner')
    y_pos = np.arange(merged.shape[0], dtype=float)
    colors = [PAPER_COOL_MID if value == 'control' else PAPER_RED for value in merged['disease_group']]
    for idx, row in enumerate(merged.itertuples(index=False)):
        right_ax.plot([row.baseline_pd_probability, row.pd_probability], [idx, idx], color=colors[idx], linewidth=1.2, alpha=0.85)
        right_ax.scatter([row.baseline_pd_probability], [idx], color=blend_with_white(colors[idx], 0.55), edgecolors=colors[idx], s=28, zorder=3)
        right_ax.scatter([row.pd_probability], [idx], color=colors[idx], edgecolors='white', linewidths=0.5, s=28, zorder=3)
    right_ax.set_yticks(y_pos, labels=merged['split_group_id'].str.replace('GSE157783::', '', regex=False))
    right_ax.set_xlabel('Baseline -> GNN\nholdout PD', labelpad=1.5)
    right_ax.tick_params(axis='x', labelsize=4.0)
    right_ax.tick_params(axis='y', labelsize=4.4)
    right_ax.set_xlim(0.0, 1.05)
    right_ax.spines['top'].set_visible(False)
    right_ax.spines['right'].set_visible(False)
    right_ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')
    right_ax.text(0.02, 1.02, 'GSE157783 donors', transform=right_ax.transAxes, ha='left', va='bottom', fontsize=3.8, color='#667085', clip_on=False)


def figure3_pathway_landscape_panel(ax: plt.Axes) -> None:
    frame = step2_ranked_pathways().nlargest(10, 'ranking_score').sort_values('ranking_score')
    ypos = np.arange(frame.shape[0], dtype=float)
    colors = FIGURE3_PATHWAY_PALETTE[: frame.shape[0]]
    ax.barh(ypos, frame['ranking_score'], color=colors, edgecolor='white', linewidth=0.6)
    for idx, row in enumerate(frame.itertuples(index=False)):
        x = float(row.ranking_score)
        x_text = x + 0.04 if x >= 0 else x - 0.04
        ha = 'left' if x >= 0 else 'right'
        ax.text(x_text, ypos[idx], f'{x:.2f}', va='center', ha=ha, fontsize=4.2, color='#344054', fontweight='semibold')
    ax.set_yticks(ypos, labels=[fill(str(item), 16) for item in frame['pathway_family']])
    ax.set_xlabel('Ranking score')
    set_left_title(ax, 'Ranked Step 2 pathway families')
    figure5_panel_label(ax, 'a', x=-0.10)
    ax.tick_params(axis='x', labelsize=4.6)
    ax.tick_params(axis='y', labelsize=4.5)
    ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')
    xmin = min(-0.12, float(frame['ranking_score'].min()) - 0.10)
    xmax = float(frame['ranking_score'].max()) + 0.24
    ax.set_xlim(xmin, xmax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def figure3_pair_landscape_panel(ax: plt.Axes) -> None:
    frame = step2_ranked_lr_pairs().nlargest(30, 'ranking_score').copy()
    frame['plot_color'] = [FIGURE3_PAIR_HIGHLIGHT_COLORS.get(str(key), blend_with_white(PAPER_COOL_LIGHT, 0.10)) for key in frame['pair_key']]
    frame['plot_alpha'] = np.where(frame['pair_key'].isin(FIGURE3_PAIR_HIGHLIGHT_COLORS.keys()), 0.92, 0.80)
    ax.scatter(frame['delta_score'], frame['pd_score'], s=18 + 8 * frame['ranking_score'], color=frame['plot_color'], alpha=frame['plot_alpha'], edgecolors='white', linewidths=0.4)
    for row in frame.loc[frame['pair_key'].isin(FIGURE3_PAIR_HIGHLIGHT_COLORS.keys())].itertuples(index=False):
        ax.text(float(row.delta_score) + 0.04, float(row.pd_score) + 0.04, str(row.pair_key), fontsize=4.2, color='#344054')
    ax.axvline(0.0, linestyle='--', linewidth=0.7, color='#D0D5DD')
    ax.axhline(0.0, linestyle='--', linewidth=0.7, color='#D0D5DD')
    set_left_title(ax, 'Top ligand-receptor landscape')
    figure5_panel_label(ax, 'b', x=-0.10)
    ax.set_xlabel('Delta score')
    ax.set_ylabel('PD score')
    ax.tick_params(labelsize=4.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def figure3_candidate_heatmap_panel(ax: plt.Axes) -> None:
    frame = package_step3_candidate_evidence().set_index('refined_program_name')
    frame.index = [short_program_name(str(item), width=14) for item in frame.index]
    metrics = [
        'step3_program_score',
        'anchor_pair_score_mean',
        'anchor_pair_score_max',
        'program_coherence',
        'donor_trajectory_coherence',
        'validation_readiness',
    ]
    cmap = LinearSegmentedColormap.from_list('figure3_evidence', [PAPER_COOL_PALE, PAPER_COOL_LIGHT, PAPER_BLUE, PAPER_WARM_SOFT])
    sns.heatmap(frame[metrics], cmap=cmap, linewidths=0.7, linecolor='white', annot=True, fmt='.2f', cbar=False, ax=ax, annot_kws={'fontsize': 4.0})
    figure5_panel_label(ax, 'c', x=-0.10)
    set_left_title(ax, 'Step 3 candidate evidence')
    ax.tick_params(axis='x', labelsize=4.0, rotation=25)
    ax.tick_params(axis='y', labelsize=3.6, rotation=0, pad=1.2)
    ax.set_xlabel('')
    ax.set_ylabel('')


def figure3_state_bubble_panel(ax: plt.Axes) -> None:
    top = package_step3_top10().nlargest(8, 'step3_program_score').copy()
    program_order = top['candidate_id'].tolist()
    program_labels = {row.candidate_id: short_program_name(str(row.refined_program_name), width=16) for row in top.itertuples(index=False)}
    state = step3_candidate_state_support().copy()
    state = state.loc[state['candidate_id'].isin(program_order) & state['state_pair_rank'].le(3)].copy()
    state['program_label'] = state['candidate_id'].map(program_labels)
    state['state_pair'] = [compact_state_pair(sender, receiver) for sender, receiver in zip(state['sender_state_id'], state['receiver_state_id'], strict=True)]
    pair_order = (
        state.groupby('state_pair')['high_contribution_edge_enrichment']
        .sum()
        .sort_values(ascending=False)
        .head(6)
        .index
        .tolist()
    )
    state = state.loc[state['state_pair'].isin(pair_order)].copy()
    x_map = {label: idx for idx, label in enumerate(pair_order)}
    y_map = {candidate_id: idx for idx, candidate_id in enumerate(program_order)}

    ax.set_xlim(-0.5, len(pair_order) - 0.5)
    ax.set_ylim(-0.5, len(program_order) - 0.5)
    for x in range(len(pair_order)):
        ax.axvline(x - 0.5, color='#E4E7EC', linewidth=0.5, zorder=0)
    for y in range(len(program_order)):
        ax.axhline(y - 0.5, color='#EEF2F6', linewidth=0.5, zorder=0)

    sizes = 24.0 + 18.0 * np.sqrt(state['high_contribution_edge_enrichment'].to_numpy(dtype=float))
    colors = [FIGURE3_BUBBLE_RANK_COLORS.get(int(rank), PAPER_COOL_MID) for rank in state['state_pair_rank']]
    ax.scatter(
        [x_map[item] for item in state['state_pair']],
        [y_map[item] for item in state['candidate_id']],
        s=sizes,
        c=colors,
        alpha=0.86,
        edgecolors='white',
        linewidths=0.5,
        zorder=3,
    )
    top_hits = state.nlargest(6, 'high_contribution_edge_enrichment')
    for row in top_hits.itertuples(index=False):
        ax.text(x_map[row.state_pair], y_map[row.candidate_id], f'{row.high_contribution_edge_enrichment:.0f}', ha='center', va='center', fontsize=3.5, color='white', fontweight='bold', zorder=4)

    legend_handles = [
        mpl.lines.Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5, markersize=5.5, label=f'Rank {rank}')
        for rank, color in FIGURE3_BUBBLE_RANK_COLORS.items()
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=3.6, loc='upper right', ncol=3, handletextpad=0.4, columnspacing=0.8)
    figure5_panel_label(ax, 'd', x=-0.10)
    set_left_title(ax, 'State-pair enrichment bubble map')
    ax.set_xticks(np.arange(len(pair_order)), labels=pair_order)
    ax.set_yticks(np.arange(len(program_order)), labels=[program_labels[item] for item in program_order])
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=3.8, rotation=28)
    ax.tick_params(axis='y', labelsize=4.0, rotation=0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure3_sankey_panel(ax: plt.Axes) -> None:
    top = package_step3_top10().nlargest(6, 'step3_program_score').copy()
    top['sender_label'] = top['dominant_sender_state'].map(sankey_state_label)
    top['receiver_label'] = top['dominant_receiver_state'].map(sankey_state_label)
    top['program_label'] = top['refined_program_name'].map(sankey_program_label)

    sender_nodes = list(dict.fromkeys(top['sender_label'].tolist()))
    program_nodes = top['program_label'].tolist()
    receiver_nodes = list(dict.fromkeys(top['receiver_label'].tolist()))
    node_names = sender_nodes + program_nodes + receiver_nodes
    node_index = {name: idx for idx, name in enumerate(node_names)}

    source: list[int] = []
    target: list[int] = []
    value: list[float] = []
    for row in top.itertuples(index=False):
        score = max(float(row.step3_program_score), 0.04)
        source.extend([node_index[row.sender_label], node_index[row.program_label]])
        target.extend([node_index[row.program_label], node_index[row.receiver_label]])
        value.extend([score, score])

    sender_palette = [PAPER_BLUE, PAPER_COOL_MID, PAPER_COOL_LIGHT][: len(sender_nodes)]
    program_palette = [PAPER_WARM_DARK, PAPER_BLUE, PAPER_WARM_SOFT, PAPER_COOL_MID, PAPER_WARM_LIGHT, PAPER_COOL_LIGHT][: len(program_nodes)]
    receiver_palette = [PAPER_RED, PAPER_WARM_SOFT, PAPER_WARM_LIGHT][: len(receiver_nodes)]
    pf.sankey(
        source,
        target,
        value,
        node_names=node_names,
        color_palette=sender_palette + program_palette + receiver_palette,
        link_alpha=0.28,
        x_margin=0.02,
        node_width=0.052,
        node_pad=0.024,
        curve_strength=0.38,
        node_label_fontsize=3.8,
        ax=ax,
    )
    figure5_panel_label(ax, 'e', x=-0.02)
    floating_title(ax, 'Mechanistic sender-program-receiver flow', y=1.03)


def figure4_trajectory_panel(ax: plt.Axes, *, label: str, program_id: str, title: str) -> None:
    donor = step3_program_scores_by_donor().loc[
        step3_program_scores_by_donor()['refined_program_id'].eq(program_id)
        & step3_program_scores_by_donor()['age_years'].notna()
    ].copy()
    figure5_panel_label(ax, label, x=-0.10)
    set_left_title(ax, title)
    for disease_group in ['control', 'lbd', 'pd']:
        sub = donor.loc[donor['disease_group'].eq(disease_group)].copy()
        if sub.empty:
            continue
        ax.scatter(sub['age_years'], sub['score_normalized'], color=cohort_color(disease_group), s=20, alpha=0.88, edgecolors='white', linewidths=0.4, label=pretty_cohort(disease_group))
        if sub.shape[0] >= 2:
            coeff = np.polyfit(sub['age_years'], sub['score_normalized'], deg=1)
            x_grid = np.linspace(float(sub['age_years'].min()), float(sub['age_years'].max()), 50)
            ax.plot(x_grid, np.polyval(coeff, x_grid), color=cohort_color(disease_group), linewidth=1.0, alpha=0.80)
    ax.axhline(0.0, linestyle='--', linewidth=0.7, color='#D0D5DD')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Program score (z)')
    ax.tick_params(labelsize=4.8)
    ax.grid(axis='y', linestyle='--', linewidth=0.45, color='#E4E7EC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=4.0, loc='upper left')


def figure4_state_support_panel(ax: plt.Axes, *, label: str, candidate_id: str, title: str, legend_loc: str = 'lower right') -> None:
    frame = step3_candidate_state_support().loc[step3_candidate_state_support()['candidate_id'].eq(candidate_id)].copy()
    frame = frame.sort_values('state_pair_rank').head(3)
    pairs = [f"{short_state_label(row.sender_state_id)}\n-> {short_state_label(row.receiver_state_id)}" for row in frame.itertuples(index=False)]
    ypos = np.arange(frame.shape[0], dtype=float)[::-1]
    figure5_panel_label(ax, label, x=-0.10)
    set_left_title(ax, title)
    if 'apoe' in candidate_id.lower():
        bar_colors = [PAPER_BLUE, PAPER_RED, PAPER_COOL_MID]
        ligand_color = PAPER_RED
        receptor_color = PAPER_WARM_SOFT
    else:
        bar_colors = [PAPER_RED, PAPER_BLUE, PAPER_WARM_SOFT]
        ligand_color = PAPER_BLUE
        receptor_color = PAPER_COOL_MID
    ax.barh(
        ypos,
        frame['high_contribution_edge_enrichment'],
        color=bar_colors[: frame.shape[0]],
        height=0.42,
        label='Edge enrichment',
        alpha=0.94,
    )
    ax.scatter(frame['ligand_expression_support'], ypos + 0.14, s=28, color=ligand_color, edgecolors='white', linewidths=0.5, label='Ligand expr', zorder=3)
    ax.scatter(
        frame['receptor_expression_support'],
        ypos - 0.14,
        s=30,
        color=receptor_color,
        edgecolors=blend_with_black(receptor_color, 0.18),
        linewidths=0.5,
        marker='D',
        label='Receptor expr',
        zorder=3,
    )
    for y0, row in zip(ypos, frame.itertuples(index=False), strict=True):
        ax.text(float(row.high_contribution_edge_enrichment) + 0.4, y0, f"{row.high_contribution_edge_enrichment:.1f}", va='center', fontsize=4.2, color='#475467')
    ax.set_yticks(ypos, labels=pairs)
    ax.set_xlabel('State support score')
    ax.tick_params(axis='x', labelsize=4.6)
    ax.tick_params(axis='y', labelsize=4.0)
    if legend_loc == 'upper left':
        ax.legend(frameon=False, fontsize=4.0, loc='upper left', bbox_to_anchor=(0.01, 0.99), handletextpad=0.4, borderaxespad=0.2)
    elif legend_loc == 'bottom center':
        ax.legend(
            frameon=False,
            fontsize=4.0,
            loc='lower center',
            bbox_to_anchor=(0.50, 0.01),
            ncol=3,
            columnspacing=0.8,
            handletextpad=0.35,
            borderaxespad=0.2,
        )
    else:
        ax.legend(frameon=False, fontsize=4.0, loc='lower right', bbox_to_anchor=(0.99, 0.02), handletextpad=0.4, borderaxespad=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')


def figure4_exemplar_panel(host_ax: plt.Axes, *, label: str, program_id: str, title: str, anchor_pairs: list[str]) -> None:
    host_ax.axis('off')
    figure5_panel_label(host_ax, label, x=-0.02)
    floating_title(host_ax, title, y=1.03)
    pair_scores = step3_pair_scores_by_donor().loc[
        step3_pair_scores_by_donor()['refined_program_id'].eq(program_id)
        & step3_pair_scores_by_donor()['task_name'].eq('microglia_to_dopaminergic_neuron')
        & step3_pair_scores_by_donor()['pair_key'].isin(anchor_pairs)
        & step3_pair_scores_by_donor()['disease_group'].isin(['control', 'pd'])
    ].copy()
    exemplar_rows = (
        pair_scores.sort_values('contribution_per_sampled_subgraph', ascending=False)
        .groupby('disease_group', as_index=False)
        .head(1)
        .sort_values('disease_group')
    )

    bounds = {'control': [0.03, 0.12, 0.43, 0.78], 'pd': [0.54, 0.12, 0.43, 0.78]}
    for row in exemplar_rows.itertuples(index=False):
        x0, y0, width, height = bounds[str(row.disease_group)]
        color = cohort_color(str(row.disease_group))
        add_round_box(host_ax, x0, y0, width, height, facecolor=blend_with_white(color, 0.92), edgecolor=color, linewidth=0.95, radius=0.03)
        host_ax.text(x0 + 0.03, y0 + height - 0.08, pretty_cohort(str(row.disease_group)), transform=host_ax.transAxes, fontsize=5.0, fontweight='semibold', color=color)
        host_ax.text(x0 + 0.03, y0 + height - 0.14, str(row.pair_key).replace('->', '  ->  '), transform=host_ax.transAxes, fontsize=4.6, color='#1F2937')
        host_ax.text(x0 + 0.03, y0 + height - 0.20, f"donor {row.donor_id} | {pretty_source(row.source_id)}", transform=host_ax.transAxes, fontsize=3.7, color='#667085')
        center_y = y0 + 0.38
        left_node_x = x0 + width * 0.20
        right_node_x = x0 + width * 0.80
        node_radius = 0.050
        host_ax.add_patch(
            mpl.patches.Circle(
                (left_node_x, center_y),
                node_radius,
                transform=host_ax.transAxes,
                facecolor=blend_with_white(PAPER_RED, 0.40),
                edgecolor=PAPER_RED,
                linewidth=1.0,
            )
        )
        host_ax.add_patch(
            mpl.patches.Circle(
                (right_node_x, center_y),
                node_radius,
                transform=host_ax.transAxes,
                facecolor=blend_with_white(PAPER_TEAL, 0.35),
                edgecolor=PAPER_TEAL,
                linewidth=1.0,
            )
        )
        host_ax.text(left_node_x, center_y, 'Lig', transform=host_ax.transAxes, ha='center', va='center', fontsize=4.8, fontweight='bold', color='#1F2937')
        host_ax.text(right_node_x, center_y, 'Rec', transform=host_ax.transAxes, ha='center', va='center', fontsize=4.8, fontweight='bold', color='#1F2937')
        arrow = mpl.patches.FancyArrowPatch(
            (left_node_x + node_radius + 0.01, center_y),
            (right_node_x - node_radius - 0.01, center_y),
            transform=host_ax.transAxes,
            arrowstyle='Simple,tail_width=0.8,head_width=6,head_length=8',
            linewidth=0.0,
            facecolor=color,
            edgecolor=color,
            alpha=0.85,
        )
        host_ax.add_patch(arrow)
        host_ax.text(x0 + width / 2.0, center_y + 0.08, f"contribution {float(row.contribution_per_sampled_subgraph):.1f}", transform=host_ax.transAxes, ha='center', va='center', fontsize=4.3, color='#344054')
        bars = [
            ('Lig. pct', float(row.sender_pct_expr), PAPER_RED),
            ('Rec. pct', float(row.receiver_pct_expr), PAPER_TEAL),
            ('Lig. norm', float(row.sender_mean_norm) / 100.0, PAPER_BLUE),
            ('Rec. norm', float(row.receiver_mean_norm) / 100.0, PAPER_BLUE_SOFT),
        ]
        bar_x = x0 + 0.04
        bar_width = width * 0.60
        label_x = bar_x + bar_width + 0.02
        for idx, (name, value, bar_color) in enumerate(bars):
            base_y = y0 + 0.21 - idx * 0.047
            add_round_box(host_ax, bar_x, base_y, bar_width, 0.022, facecolor='#E9EEF3', edgecolor='#E9EEF3', linewidth=0.0, radius=0.01)
            add_round_box(host_ax, bar_x, base_y, bar_width * np.clip(value, 0.0, 1.0), 0.022, facecolor=bar_color, edgecolor=bar_color, linewidth=0.0, radius=0.01)
            host_ax.text(label_x, base_y + 0.011, name, transform=host_ax.transAxes, ha='left', va='center', fontsize=4.1, color='#667085')


def figure7_gene_support_panel(ax: plt.Axes) -> None:
    genes = package_gene_support_summary().copy()
    support_code = {
        'non_instrumentable': 1,
        'weak_or_inconclusive': 2,
        'moderate_genetic_support': 3,
        'strong_genetic_support': 4,
    }
    support_text = {1: 'NI', 2: 'W', 3: 'M', 4: 'S'}
    genes['support_code'] = genes['gene_support_bin'].map(support_code).fillna(0).astype(int)
    genes['membership_n'] = genes['program_ids_json'].map(lambda value: len(parse_json_list(value)))
    genes = genes.sort_values(['support_code', 'membership_n', 'gene_symbol'], ascending=[False, False, True]).reset_index(drop=True)

    matrix = np.zeros((genes.shape[0], len(FIGURE7_PROGRAM_ORDER)), dtype=int)
    for row_idx, row in enumerate(genes.itertuples(index=False)):
        membership = set(parse_json_list(row.program_ids_json))
        for col_idx, program_id in enumerate(FIGURE7_PROGRAM_ORDER):
            if program_id in membership:
                matrix[row_idx, col_idx] = int(row.support_code)

    sns.heatmap(
        matrix,
        cmap=ListedColormap(FIGURE7_SUPPORT_COLOR_SCALE),
        cbar=False,
        linewidths=0.7,
        linecolor='white',
        ax=ax,
        vmin=0,
        vmax=4,
    )
    figure5_panel_label(ax, 'a', x=-0.10)
    set_left_title(ax, 'Gene-level genetic support map')
    ax.text(
        0.99,
        1.05,
        'S strong | M moderate | W weak | NI non-instrumentable | blank not in program',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=3.8,
        color='#667085',
        clip_on=False,
    )
    ax.set_xticks(np.arange(len(FIGURE7_PROGRAM_ORDER)) + 0.5, labels=['APOE', 'Canon.', 'Context'])
    ax.set_yticks(np.arange(genes.shape[0]) + 0.5, labels=genes['gene_symbol'].tolist())
    ax.set_box_aspect(0.88)
    ax.tick_params(axis='x', labelsize=4.8, length=0, rotation=0)
    ax.tick_params(axis='y', labelsize=3.8, length=0, pad=1)
    ax.set_xlabel('')
    ax.set_ylabel('')
    for label, program_id in zip(ax.get_xticklabels(), FIGURE7_PROGRAM_ORDER, strict=True):
        label.set_color(FIGURE7_PROGRAM_COLORS[program_id])
        label.set_fontweight('semibold')
    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_ha('right')
        label.set_va('center')
    for y_idx, gene in enumerate(genes['gene_symbol'].tolist()):
        for x_idx in range(len(FIGURE7_PROGRAM_ORDER)):
            code = int(matrix[y_idx, x_idx])
            if code == 0:
                continue
            fill_color = FIGURE7_SUPPORT_COLOR_SCALE[code]
            ax.text(
                x_idx + 0.5,
                y_idx + 0.5,
                support_text[code],
                ha='center',
                va='center',
                fontsize=3.9,
                color=text_color_for_fill(fill_color),
                fontweight='bold',
            )
        if gene == 'GPNMB':
            ax.add_patch(
                mpl.patches.Rectangle(
                    (0, y_idx),
                    len(FIGURE7_PROGRAM_ORDER),
                    1,
                    fill=False,
                    linewidth=1.2,
                    edgecolor=PAPER_WARM_DARK,
                )
            )
    for tick in ax.get_yticklabels():
        if tick.get_text() == 'GPNMB':
            tick.set_color(PAPER_WARM_DARK)
            tick.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure7_gpnmb_summary_panel(host_ax: plt.Axes) -> None:
    row = figure5_gpnmb_support()
    figure5_panel_label(host_ax, 'b', x=-0.02)
    host_ax.axis('off')
    floating_title(host_ax, 'GPNMB MR and coloc convergence', 'single strong instrumented gene', y=1.03, subtitle_y=1.00)
    host_ax.text(
        0.0,
        0.95,
        'Bars show direct effect size and coloc support instead of repeated donuts.',
        transform=host_ax.transAxes,
        ha='left',
        va='top',
        fontsize=4.3,
        color='#667085',
    )

    labels = ['Primary', 'Secondary']
    y = np.array([1, 0], dtype=float)
    beta_values = [float(row['best_primary_mr_beta']), float(row['best_secondary_mr_beta'])]
    coloc_values = [float(row['best_primary_coloc_h4_proxy']), float(row['best_secondary_coloc_h4_proxy'])]
    beta_fdr = [float(row['best_primary_mr_p_adj_fdr']), float(row['best_secondary_mr_p_adj_fdr'])]

    mr_ax = host_ax.inset_axes([0.03, 0.34, 0.39, 0.43])
    mr_colors = [PAPER_WARM_DARK, PAPER_BLUE]
    beta_limit = max(0.14, max(abs(value) for value in beta_values) * 1.35)
    mr_ax.barh(y, beta_values, color=mr_colors, height=0.38)
    mr_ax.axvline(0.0, linewidth=0.75, color='#98A2B3')
    for y_pos, value, fdr in zip(y, beta_values, beta_fdr, strict=True):
        ha = 'left' if value >= 0 else 'right'
        x_text = value + 0.005 if value >= 0 else value - 0.005
        mr_ax.text(x_text, y_pos, f'{value:+.2f} | FDR {format_compact_stat(fdr)}', ha=ha, va='center', fontsize=4.0, color='#475467')
    mr_ax.set_yticks(y, labels=labels)
    mr_ax.set_xlim(min(-0.02, min(beta_values) * 1.25), beta_limit)
    mr_ax.set_xlabel('MR beta')
    mr_ax.set_title('MR effect', loc='left', fontsize=5.2, pad=3.5)
    mr_ax.tick_params(axis='x', labelsize=4.2, length=0)
    mr_ax.tick_params(axis='y', labelsize=4.2, length=0)
    mr_ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')
    mr_ax.spines['top'].set_visible(False)
    mr_ax.spines['right'].set_visible(False)

    coloc_ax = host_ax.inset_axes([0.64, 0.34, 0.24, 0.43])
    coloc_colors = [PAPER_WARM_SOFT, PAPER_COOL_MID]
    coloc_ax.barh(y, coloc_values, color=coloc_colors, height=0.38)
    coloc_ax.axvline(0.80, linestyle='--', linewidth=0.75, color='#98A2B3')
    for y_pos, value in zip(y, coloc_values, strict=True):
        coloc_ax.text(value + 0.03, y_pos, f'{value:.2f}', ha='left', va='center', fontsize=4.0, color='#475467')
    coloc_ax.set_yticks(y, labels=labels)
    coloc_ax.set_xlim(0.0, 1.0)
    coloc_ax.set_xlabel('Coloc H4')
    coloc_ax.set_title('Coloc support', loc='left', fontsize=5.2, pad=3.5)
    coloc_ax.tick_params(axis='x', labelsize=4.2, length=0)
    coloc_ax.tick_params(axis='y', labelsize=4.2, length=0)
    coloc_ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')
    coloc_ax.spines['top'].set_visible(False)
    coloc_ax.spines['right'].set_visible(False)

    tissue_label = str(row['best_tissue']).replace('Brain_', '').replace('_basal_ganglia', '').replace('_', ' ')
    summary_items = [
        ('Top tissue', tissue_label),
        ('Top SNP', str(row['best_snp_id'])),
        ('Mean F', f"{float(row['best_primary_mean_F']):.1f}"),
        ('Direction', 'Consistent' if bool(row['cross_outcome_direction_consistent']) else 'Mixed'),
    ]
    chip_colors = [PAPER_BLUE_LIGHT, PAPER_WARM_LIGHT, PAPER_COOL_MID, PAPER_WARM_SOFT]
    chip_x = [0.03, 0.27, 0.51, 0.75]
    chip_w = [0.18, 0.18, 0.17, 0.17]
    for (title, value), color, x0, width in zip(summary_items, chip_colors, chip_x, chip_w, strict=True):
        add_round_box(
            host_ax,
            x0,
            0.02,
            width,
            0.085,
            facecolor=blend_with_white(color, 0.84),
            edgecolor=color,
            linewidth=0.8,
            radius=0.025,
        )
        host_ax.text(x0 + 0.015, 0.072, title, transform=host_ax.transAxes, ha='left', va='center', fontsize=3.7, color='#667085')
        host_ax.text(x0 + 0.015, 0.045, value, transform=host_ax.transAxes, ha='left', va='center', fontsize=4.3, color='#344054', fontweight='semibold')


def figure7_program_genetics_panel(ax: plt.Axes) -> None:
    frame = package_program_genetics().copy()
    frame['empirical_logp'] = -np.log10(frame['empirical_p_value'])
    frame['magma_logp'] = -np.log10(frame['magma_like_p_value'])
    frame = frame.sort_values('empirical_logp', ascending=False).reset_index(drop=True)
    y = np.arange(frame.shape[0], dtype=float)[::-1]

    figure5_panel_label(ax, 'c', x=-0.10)
    set_left_title(ax, 'Program-level matched-null enrichment')
    ax.text(
        0.99,
        1.04,
        'Empirical matched-null stays stronger than MAGMA-like in all 3 programs',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=3.9,
        color='#667085',
        clip_on=False,
    )

    ax.hlines(y, frame['magma_logp'], frame['empirical_logp'], color=blend_with_white(PAPER_BLUE, 0.48), linewidth=1.4, zorder=1)
    ax.scatter(frame['magma_logp'], y, s=34, color=PAPER_BLUE, edgecolors='white', linewidths=0.55, label='MAGMA-like P', zorder=3)
    ax.scatter(frame['empirical_logp'], y, s=38, marker='D', color=PAPER_WARM_DARK, edgecolors='white', linewidths=0.55, label='Empirical matched-null P', zorder=4)
    ax.axvline(-np.log10(0.05), linestyle='--', linewidth=0.8, color='#98A2B3', zorder=0)
    ax.text(-np.log10(0.05) + 0.02, y.max() + 0.35, 'P = 0.05', fontsize=3.8, color='#667085')

    for y_pos, row in zip(y, frame.itertuples(index=False), strict=True):
        note = f"{int(row.n_strong_gene_support)} strong gene" if int(row.n_strong_gene_support) == 1 else 'no strong gene'
        ax.text(max(float(row.empirical_logp), float(row.magma_logp)) + 0.06, y_pos, note, va='center', ha='left', fontsize=4.0, color='#475467')

    ax.set_yticks(y, labels=[short_program_name(item, width=18) for item in frame['program_name']])
    ax.set_xlabel('-log10 P')
    ax.set_xlim(0.0, float(frame['empirical_logp'].max()) + 0.42)
    ax.tick_params(axis='x', labelsize=4.8)
    ax.tick_params(axis='y', labelsize=4.1)
    ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')
    ax.legend(frameon=False, fontsize=4.0, loc='lower right', handletextpad=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def figure7_integrated_evidence_panel(host_ax: plt.Axes) -> None:
    evidence = pd.read_csv(PACKAGE_TABLE_DIR / 'step4_final_evidence_matrix.csv').copy()
    evidence['program_id'] = pd.Categorical(evidence['program_id'], categories=FIGURE7_PROGRAM_ORDER, ordered=True)
    evidence = evidence.sort_values('program_id').reset_index(drop=True)

    figure5_panel_label(host_ax, 'd', x=-0.02)
    host_ax.axis('off')
    floating_title(host_ax, 'Integrated evidence ladder', 'component bars, penalties, final score, and claim tier', y=1.03, subtitle_y=1.00)

    components = [
        ('Primary', 'primary_spatial_component', PAPER_BLUE),
        ('Repl.', 'secondary_spatial_component', PAPER_COOL_MID),
        ('Gene', 'gene_level_component', PAPER_WARM_LIGHT),
        ('Program', 'program_level_genetic_component', PAPER_WARM_DARK),
        ('Holdout', 'cross_dataset_consistency_component', PAPER_WARM_SOFT),
    ]
    header_x = [0.27 + idx * 0.098 for idx in range(len(components))]
    for (label, _, color), x0 in zip(components, header_x, strict=True):
        add_round_box(
            host_ax,
            x0,
            0.91,
            0.074,
            0.045,
            facecolor=blend_with_white(color, 0.84),
            edgecolor=color,
            linewidth=0.7,
            radius=0.02,
        )
        host_ax.text(x0 + 0.037, 0.933, label, transform=host_ax.transAxes, ha='center', va='center', fontsize=3.8, color='#344054', fontweight='semibold')
    add_round_box(host_ax, 0.78, 0.91, 0.07, 0.045, facecolor=blend_with_white(PAPER_RED, 0.86), edgecolor=PAPER_RED, linewidth=0.7, radius=0.02)
    host_ax.text(0.815, 0.933, 'Penalty', transform=host_ax.transAxes, ha='center', va='center', fontsize=3.8, color='#344054', fontweight='semibold')
    add_round_box(host_ax, 0.88, 0.91, 0.07, 0.045, facecolor='#F8FAFC', edgecolor='#D6DEE6', linewidth=0.7, radius=0.02)
    host_ax.text(0.915, 0.933, 'Score', transform=host_ax.transAxes, ha='center', va='center', fontsize=3.8, color='#344054', fontweight='semibold')

    row_y = [0.67, 0.40, 0.13]
    for y0, row in zip(row_y, evidence.itertuples(index=False), strict=True):
        program_color = FIGURE7_PROGRAM_COLORS[str(row.program_id)]
        add_round_box(
            host_ax,
            0.015,
            y0,
            0.95,
            0.19,
            facecolor=blend_with_white(program_color, 0.92),
            edgecolor=blend_with_white(program_color, 0.20),
            linewidth=0.9,
            radius=0.03,
        )
        host_ax.text(
            0.025,
            y0 + 0.137,
            short_program_name(str(row.program_name), width=22).replace('\n', ' '),
            transform=host_ax.transAxes,
            ha='left',
            va='center',
            fontsize=4.8,
            color=blend_with_black(program_color, 0.14),
            fontweight='semibold',
        )
        host_ax.text(
            0.025,
            y0 + 0.092,
            pretty_claim_tier(str(row.claim_tier)),
            transform=host_ax.transAxes,
            ha='left',
            va='center',
            fontsize=4.1,
            color='#344054',
        )
        host_ax.text(
            0.025,
            y0 + 0.048,
            pretty_promotion_decision(str(row.promotion_decision)),
            transform=host_ax.transAxes,
            ha='left',
            va='center',
            fontsize=4.0,
            color='#667085',
        )

        for (label, key, color), x0 in zip(components, header_x, strict=True):
            value = float(getattr(row, key))
            add_round_box(host_ax, x0, y0 + 0.090, 0.074, 0.022, facecolor='#EEF2F6', edgecolor='#EEF2F6', linewidth=0.0, radius=0.01)
            add_round_box(host_ax, x0, y0 + 0.090, 0.074 * np.clip(value, 0.0, 1.0), 0.022, facecolor=color, edgecolor=color, linewidth=0.0, radius=0.01)
            host_ax.text(x0 + 0.037, y0 + 0.056, f'{value:.2f}', transform=host_ax.transAxes, ha='center', va='center', fontsize=4.0, color='#475467')

        penalty = float(row.total_penalty)
        add_round_box(host_ax, 0.78, y0 + 0.090, 0.07, 0.022, facecolor=blend_with_white(PAPER_RED, 0.86), edgecolor=blend_with_white(PAPER_RED, 0.86), linewidth=0.0, radius=0.01)
        add_round_box(host_ax, 0.78, y0 + 0.090, 0.07 * np.clip(penalty / 0.10, 0.0, 1.0), 0.022, facecolor=PAPER_RED, edgecolor=PAPER_RED, linewidth=0.0, radius=0.01)
        host_ax.text(0.815, y0 + 0.056, f'-{penalty:.2f}', transform=host_ax.transAxes, ha='center', va='center', fontsize=4.0, color='#475467')

        score = float(row.final_step4_score)
        add_round_box(host_ax, 0.88, y0 + 0.078, 0.07, 0.046, facecolor=program_color, edgecolor=program_color, linewidth=0.0, radius=0.02)
        host_ax.text(
            0.915,
            y0 + 0.101,
            f'{score:.2f}',
            transform=host_ax.transAxes,
            ha='center',
            va='center',
            fontsize=4.8,
            color=text_color_for_fill(program_color),
            fontweight='bold',
        )


def figure8_label_confidence_panel(ax: plt.Axes) -> None:
    frame = step2_label_confidence_sensitivity().copy()
    summary = frame.groupby(['confidence_threshold', 'task_name'], as_index=False)['n_graph_units'].sum()
    task_palette = [PAPER_BLUE, PAPER_WARM_DARK, PAPER_COOL_MID, PAPER_WARM_SOFT]
    for (task_name, sub), color in zip(summary.groupby('task_name'), task_palette, strict=False):
        ax.plot(
            sub['confidence_threshold'],
            sub['n_graph_units'],
            marker='o',
            linewidth=1.2,
            markersize=4.0,
            color=color,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=0.5,
            label=pretty_task(task_name),
        )
    figure5_panel_label(ax, 'a', x=-0.10)
    set_left_title(ax, 'Label-confidence sensitivity')
    ax.set_xlabel('Confidence threshold')
    ax.set_ylabel('Graph units retained')
    ax.legend(frameon=False, fontsize=3.8, loc='upper right')
    ax.tick_params(labelsize=4.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.45, color='#E4E7EC')


def figure8_cell_count_panel(ax: plt.Axes) -> None:
    frame = step2_cell_count_robustness().copy()
    color_map = {
        'pd prob vs n edges': PAPER_WARM_SOFT,
        'pd prob vs n nodes': PAPER_WARM_DARK,
        'pred age vs n edges': PAPER_COOL_MID,
        'pred age vs n nodes': PAPER_BLUE,
    }
    y = np.arange(frame.shape[0], dtype=float)
    colors = [color_map.get(str(metric), PAPER_BLUE) for metric in frame['metric']]
    ax.barh(y, frame['spearman_rho'], color=colors, height=0.62)
    ax.axvline(0.0, linestyle='--', linewidth=0.7, color='#D0D5DD')
    for y_pos, value in zip(y, frame['spearman_rho'], strict=True):
        ha = 'left' if float(value) >= 0 else 'right'
        x_text = float(value) + 0.01 if float(value) >= 0 else float(value) - 0.01
        ax.text(x_text, y_pos, f'{float(value):+.2f}', ha=ha, va='center', fontsize=4.0, color='#475467')
    ax.set_yticks(y, labels=[item.replace('_', ' ') for item in frame['metric']])
    ax.set_xlabel("Spearman's rho")
    figure5_panel_label(ax, 'b', x=-0.10)
    set_left_title(ax, 'Cell-count robustness')
    ax.tick_params(axis='x', labelsize=4.8)
    ax.tick_params(axis='y', labelsize=4.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')


def figure8_state_mapping_panel(ax: plt.Axes) -> None:
    frame = package_state_mapping_sensitivity().copy()
    frame = frame.sort_values('primary_sensitivity_correlation', ascending=False).reset_index(drop=True)
    y = np.arange(frame.shape[0], dtype=float)
    palette = [PAPER_WARM_DARK, PAPER_BLUE, PAPER_WARM_SOFT, PAPER_COOL_MID]
    colors = [palette[idx % len(palette)] for idx in range(frame.shape[0])]
    ax.barh(y, frame['primary_sensitivity_correlation'], color=colors, height=0.76)
    for y_pos, value in zip(y, frame['primary_sensitivity_correlation'], strict=True):
        ax.text(float(value) + 0.015, y_pos, f'{float(value):.2f}', ha='left', va='center', fontsize=4.0, color='#475467')
    ax.set_yticks(y, labels=[short_state_label(item) for item in frame['state_id']])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('Primary vs sensitivity correlation')
    figure5_panel_label(ax, 'c', x=-0.10)
    set_left_title(ax, 'State-mapping sensitivity')
    ax.tick_params(axis='x', labelsize=4.8)
    ax.tick_params(axis='y', labelsize=4.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')


def figure8_caveat_matrix_panel(ax: plt.Axes) -> None:
    top10 = package_step3_top10().set_index('refined_program_id')
    evidence = pd.read_csv(PACKAGE_TABLE_DIR / 'step4_final_evidence_matrix.csv').set_index('program_id')
    rows = []
    for program_id, display in [
        ('app_centered_stress_signaling', 'APP stress'),
        ('slit_robo_guidance', 'Unsplit guidance'),
        ('apoe_lipoprotein_receptor_signaling', 'APOE'),
        ('slit_robo_guidance_canonical', 'Canonical'),
        ('slit_robo_guidance_context', 'Context'),
    ]:
        flags = set()
        if program_id in top10.index:
            flags = parse_pipe_flags(top10.loc[program_id, 'caveat_summary'])
        elif program_id in evidence.index:
            row = evidence.loc[program_id]
            for flag_name in ['source_dominant', 'source_flip', 'region_flip', 'negative_holdout', 'cell_count_link']:
                if bool(row.get(f'{flag_name}_flag', False)):
                    flags.add(flag_name)
        rows.append(
            {
                'Program': display,
                'Source': 'Flag' if 'source_dominant' in flags else 'Clear',
                'Src flip': 'Flag' if 'source_flip' in flags else 'Clear',
                'Region': 'Flag' if 'region_flip' in flags else 'Clear',
                'Holdout': 'Flag' if 'negative_holdout' in flags else 'Clear',
                'Counts': 'Flag' if 'cell_count_link' in flags else 'Clear',
            }
        )
    frame = pd.DataFrame(rows).set_index('Program')
    numeric = frame.eq('Flag').astype(float)
    sns.heatmap(
        numeric,
        cmap=ListedColormap([blend_with_white(PAPER_COOL_PALE, 0.30), PAPER_WARM_DARK]),
        cbar=False,
        linewidths=0.7,
        linecolor='white',
        annot=frame,
        fmt='',
        ax=ax,
        annot_kws={'fontsize': 4.0},
    )
    figure5_panel_label(ax, 'd', x=-0.10)
    set_left_title(ax, 'Caveat matrix across promoted and failed programs')
    ax.tick_params(axis='x', labelsize=4.2, rotation=20, length=0)
    ax.tick_params(axis='y', labelsize=4.2, rotation=0, length=0)
    ax.set_xlabel('')
    ax.set_ylabel('')


def figure8_leaderboard_panel(host_ax: plt.Axes) -> None:
    frame = package_full_refresh_leaderboard().copy()
    frame = frame.loc[frame['mean_pd_auroc'].notna()].copy()
    frame = frame.sort_values('mean_pd_auroc', ascending=False).head(10).reset_index(drop=True)
    frame['archive_label'] = frame['archive'].map(lambda value: compact_archive_label(value, width=18))
    claim_colors = {
        'external_mechanistic_candidate': PAPER_WARM_DARK,
        'externally_supported_program': PAPER_WARM_SOFT,
        'needs_internal_resolution': PAPER_BLUE,
    }

    figure5_panel_label(host_ax, 'e', x=-0.02)
    host_ax.axis('off')
    floating_title(host_ax, 'Cross-run / seed / weighting leaderboard', 'Top 10 non-NaN reruns by AUROC', y=1.03, subtitle_y=1.00)

    bar_ax = host_ax.inset_axes([0.00, 0.08, 0.72, 0.84])
    score_ax = host_ax.inset_axes([0.77, 0.08, 0.21, 0.84], sharey=bar_ax)
    y = np.arange(frame.shape[0], dtype=float)
    bar_colors = [claim_colors.get(str(value), PAPER_COOL_MID) for value in frame['top_claim_tier']]
    bar_ax.barh(y, frame['mean_pd_auroc'], color=bar_colors, height=0.62)
    for y_pos, value in zip(y, frame['mean_pd_auroc'], strict=True):
        bar_ax.text(float(value) + 0.006, y_pos, f'{float(value):.2f}', ha='left', va='center', fontsize=4.0, color='#475467')
    bar_ax.set_yticks(y, labels=frame['archive_label'].tolist())
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel('Mean PD AUROC')
    bar_ax.tick_params(axis='x', labelsize=4.2, length=0)
    bar_ax.tick_params(axis='y', labelsize=3.9, length=0)
    bar_ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)

    valid_score = frame['top_final_step4_score'].notna()
    score_colors = [claim_colors.get(str(value), blend_with_white(PAPER_BLUE, 0.28)) for value in frame['top_claim_tier']]
    score_ax.axvline(0.50, linestyle='--', linewidth=0.75, color='#D0D5DD')
    score_ax.scatter(
        frame.loc[valid_score, 'top_final_step4_score'],
        y[valid_score.to_numpy()],
        s=18 + 3 * frame.loc[valid_score, 'n_programs'].to_numpy(dtype=float),
        color=[score_colors[idx] for idx, keep in enumerate(valid_score.tolist()) if keep],
        edgecolors='white',
        linewidths=0.5,
        zorder=3,
    )
    for idx, row in enumerate(frame.itertuples(index=False)):
        if pd.isna(row.top_final_step4_score):
            score_ax.text(0.47, y[idx], 'none', ha='center', va='center', fontsize=3.8, color='#98A2B3')
        else:
            score_ax.text(float(row.top_final_step4_score) + 0.008, y[idx], f'{float(row.top_final_step4_score):.2f}', ha='left', va='center', fontsize=4.0, color='#475467')
    score_ax.set_xlim(0.30, 0.56)
    score_ax.set_xlabel('Top score')
    score_ax.tick_params(axis='x', labelsize=4.2, length=0)
    score_ax.tick_params(axis='y', left=False, labelleft=False)
    score_ax.grid(axis='x', linestyle='--', linewidth=0.45, color='#E4E7EC')
    score_ax.spines['top'].set_visible(False)
    score_ax.spines['right'].set_visible(False)


def figure8_failure_panel(host_ax: plt.Axes) -> None:
    host_ax.axis('off')
    figure5_panel_label(host_ax, 'f', x=-0.02)
    floating_title(host_ax, 'Failure and non-promotion cases', y=1.03)
    step3 = package_step3_top10().set_index('refined_program_id')
    step4 = package_step4_results_table().set_index('program_id')
    cards = [
        (
            'APP stress',
            float(step3.loc['app_centered_stress_signaling', 'step3_program_score']),
            'Hold back',
            'source dominant, source flip, region flip',
            PAPER_WARM_DARK,
        ),
        (
            'Unsplit guidance',
            float(step3.loc['slit_robo_guidance', 'step3_program_score']),
            'Split before Step 4',
            'source dominant, negative holdout',
            PAPER_WARM_SOFT,
        ),
        (
            'Guidance context',
            float(step4.loc['slit_robo_guidance_context', 'final_step4_score']),
            'Rerun spatial validation',
            'Genetic layer supportive, but spatial validation remains incomplete.',
            PAPER_BLUE,
        ),
    ]
    card_y = [0.71, 0.39, 0.07]
    for (title, score, status, note, color), y0 in zip(cards, card_y, strict=True):
        add_round_box(
            host_ax,
            0.02,
            y0,
            0.96,
            0.24,
            facecolor=blend_with_white(color, 0.92),
            edgecolor=color,
            linewidth=0.95,
            radius=0.03,
        )
        host_ax.text(0.05, y0 + 0.18, title, transform=host_ax.transAxes, fontsize=5.6, fontweight='semibold', color='#1F2937')
        host_ax.text(0.05, y0 + 0.105, f'score {score:.2f}', transform=host_ax.transAxes, fontsize=6.0, fontweight='bold', color=color)
        add_round_box(
            host_ax,
            0.52,
            y0 + 0.135,
            0.30,
            0.060,
            facecolor='white',
            edgecolor=blend_with_white(color, 0.15),
            linewidth=0.7,
            radius=0.02,
        )
        host_ax.text(0.67, y0 + 0.165, status, transform=host_ax.transAxes, ha='center', va='center', fontsize=4.6, color=blend_with_black(color, 0.18), fontweight='semibold')
        host_ax.text(
            0.05,
            y0 + 0.035,
            fill(note, 42),
            transform=host_ax.transAxes,
            fontsize=4.8,
            color='#475467',
            linespacing=1.15,
        )


def build_figure1(out_dir: Path) -> list[Path]:
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[0.98, 0.92], hspace=0.28)
    top_gs = gs[0].subgridspec(1, 2, width_ratios=[0.4, 0.6], wspace=0.16)
    bottom_gs = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.08)
    figure1_architecture_panel(fig.add_subplot(top_gs[0, 0]))
    figure1_resource_summary_panel(fig.add_subplot(top_gs[0, 1]))
    figure1_task_coverage_panel(fig.add_subplot(bottom_gs[0, 0]))
    figure1_funnel_panel(fig.add_subplot(bottom_gs[0, 1]))
    fig.subplots_adjust(left=0.055, right=0.994, top=0.955, bottom=0.060)
    return export_figure(fig, 'figure1_cohort_architecture', out_dir, 330, 186)


def build_figure2(out_dir: Path) -> list[Path]:
    fig = plt.figure()
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[0.96, 1.04], hspace=0.34)
    top_gs = outer_gs[0].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.0], wspace=0.26)
    bottom_gs = outer_gs[1].subgridspec(1, 3, wspace=0.32)
    figure2_roc_panel(fig.add_subplot(top_gs[0, 0]))
    figure2_pr_panel(fig.add_subplot(top_gs[0, 1]))
    figure2_calibration_panel(fig.add_subplot(top_gs[0, 2:4]))
    figure2_metric_tables_panel(fig.add_subplot(bottom_gs[0, 0]))
    figure2_baseline_panel(fig.add_subplot(bottom_gs[0, 1]))
    figure2_distribution_panel(fig.add_subplot(bottom_gs[0, 2]))
    fig.subplots_adjust(left=0.050, right=0.992, top=0.955, bottom=0.065)
    return export_figure(fig, 'figure2_model_performance', out_dir, 330, 198)


def build_figure3(out_dir: Path) -> list[Path]:
    fig = plt.figure()
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[0.92, 1.08], hspace=0.34)
    top_gs = outer_gs[0].subgridspec(1, 3, width_ratios=[1.00, 1.00, 1.16], wspace=0.34)
    bottom_gs = outer_gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.20)
    figure3_pathway_landscape_panel(fig.add_subplot(top_gs[0, 0]))
    figure3_pair_landscape_panel(fig.add_subplot(top_gs[0, 1]))
    figure3_candidate_heatmap_panel(fig.add_subplot(top_gs[0, 2]))
    figure3_state_bubble_panel(fig.add_subplot(bottom_gs[0, 0]))
    figure3_sankey_panel(fig.add_subplot(bottom_gs[0, 1]))
    fig.subplots_adjust(left=0.050, right=0.992, top=0.955, bottom=0.060)
    return export_figure(fig, 'figure3_discovery_landscape', out_dir, 330, 205)


def build_figure4(out_dir: Path) -> list[Path]:
    fig = plt.figure()
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[0.98, 1.02], hspace=0.34)
    top_gs = outer_gs[0].subgridspec(1, 3, wspace=0.30)
    bottom_gs = outer_gs[1].subgridspec(1, 3, width_ratios=[0.88, 1.06, 1.06], wspace=0.22)
    figure4_trajectory_panel(
        fig.add_subplot(top_gs[0, 0]),
        label='a',
        program_id='apoe_lipoprotein_receptor_signaling',
        title='APOE donor trajectory',
    )
    figure4_trajectory_panel(
        fig.add_subplot(top_gs[0, 1]),
        label='b',
        program_id='slit_robo_guidance',
        title='SLIT-ROBO donor trajectory',
    )
    figure4_state_support_panel(
        fig.add_subplot(top_gs[0, 2]),
        label='c',
        candidate_id='program::apoe_lipoprotein_receptor_signaling',
        title='APOE dominant state support',
    )
    figure4_state_support_panel(
        fig.add_subplot(bottom_gs[0, 0]),
        label='d',
        candidate_id='program::slit_robo_guidance',
        title='SLIT-ROBO dominant state support',
        legend_loc='bottom center',
    )
    figure4_exemplar_panel(
        fig.add_subplot(bottom_gs[0, 1]),
        label='e',
        program_id='apoe_lipoprotein_receptor_signaling',
        title='APOE representative donor exemplars',
        anchor_pairs=['APOE->LRP1', 'APOE->LRP2', 'APOE->LRP8', 'APP->LRP1'],
    )
    figure4_exemplar_panel(
        fig.add_subplot(bottom_gs[0, 2]),
        label='f',
        program_id='slit_robo_guidance',
        title='SLIT-ROBO representative donor exemplars',
        anchor_pairs=['SLIT2->ROBO1', 'SLIT2->ROBO2', 'OLFM2->ROBO2', 'NCAM1->ROBO1'],
    )
    fig.subplots_adjust(left=0.050, right=0.992, top=0.955, bottom=0.060)
    return export_figure(fig, 'figure4_state_localization_and_exemplars', out_dir, 330, 198)


def build_figure5(out_dir: Path) -> list[Path]:
    fig = plt.figure()
    gs = fig.add_gridspec(
        2,
        12,
        height_ratios=[0.90, 1.10],
        wspace=0.52,
        hspace=0.24,
    )

    figure5_metric_panel(
        fig.add_subplot(gs[0, 0:3]),
        label='a',
        metric='adjacency_excess_score',
        title='SNpc adjacency',
        y_label='Adjacency excess',
        beta_key='snpc_adjacency_beta',
        directional_p_key='directional_p_snpc',
        y_limits=(-0.15, 0.45),
    )
    figure5_metric_panel(
        fig.add_subplot(gs[0, 3:6]),
        label='b',
        metric='snpc_minus_surround',
        title='SNpc-surround contrast',
        y_label='Difference',
        beta_key='contrast_beta',
        directional_p_key='directional_p_contrast',
        y_limits=(-0.10, 0.45),
    )
    figure5_metric_panel(
        fig.add_subplot(gs[0, 6:9]),
        label='c',
        metric='same_spot_program_score',
        title='Same-spot score',
        y_label='Same-spot score',
        beta_key='same_spot_beta',
        directional_p_key='directional_p_same_spot',
        y_limits=(-0.30, 0.60),
    )
    gpnmb_genetics_panel(fig.add_subplot(gs[0, 9:12]))
    spatial_sections_panel(fig.add_subplot(gs[1, 0:6]))
    geomx_dumbbell_panel(fig.add_subplot(gs[1, 6:9]))
    holdout_donut_panel(fig.add_subplot(gs[1, 9:12]))

    fig.subplots_adjust(left=0.040, right=0.994, top=0.95, bottom=0.055)
    return export_figure(fig, 'figure5_apoe_validation_story', out_dir, 293, 186)


def build_figure6(out_dir: Path) -> list[Path]:
    fig = plt.figure()
    outer = fig.add_gridspec(2, 1, height_ratios=[0.80, 1.00], hspace=0.34)
    top = outer[0].subgridspec(1, 3, width_ratios=[2.95, 1.28, 1.16], wspace=0.28)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[1.12, 1.12, 1.78], wspace=0.30)

    figure6_split_flow_panel(fig.add_subplot(top[0, 0]))
    figure6_branch_score_panel(fig.add_subplot(top[0, 1]))
    figure6_decision_panel(fig.add_subplot(top[0, 2]))
    figure6_adjacency_panel(
        fig.add_subplot(bottom[0, 0]),
        label='d',
        program_id=FIGURE6_PROGRAM_IDS['canonical'],
        title='Canonical SNpc adjacency',
    )
    figure6_adjacency_panel(
        fig.add_subplot(bottom[0, 1]),
        label='e',
        program_id=FIGURE6_PROGRAM_IDS['context'],
        title='Context SNpc adjacency',
    )
    figure6_pair_support_panel(fig.add_subplot(bottom[0, 2]))

    fig.subplots_adjust(left=0.040, right=0.994, top=0.955, bottom=0.055)
    return export_figure(fig, 'figure6_guidance_branch_split', out_dir, 330, 186)


def build_figure7(out_dir: Path) -> list[Path]:
    fig = plt.figure()
    outer = fig.add_gridspec(2, 1, height_ratios=[1.02, 0.98], hspace=0.34)
    top = outer[0].subgridspec(1, 2, width_ratios=[0.96, 1.04], wspace=0.26)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.22)
    figure7_gene_support_panel(fig.add_subplot(top[0, 0]))
    figure7_gpnmb_summary_panel(fig.add_subplot(top[0, 1]))
    figure7_program_genetics_panel(fig.add_subplot(bottom[0, 0]))
    figure7_integrated_evidence_panel(fig.add_subplot(bottom[0, 1]))
    fig.subplots_adjust(left=0.045, right=0.992, top=0.955, bottom=0.060)
    return export_figure(fig, 'figure7_genetic_triangulation', out_dir, 330, 198)


def build_figure8(out_dir: Path) -> list[Path]:
    fig = plt.figure()
    outer = fig.add_gridspec(2, 1, height_ratios=[0.92, 1.08], hspace=0.34)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.02, 1.00, 1.02], wspace=0.32)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[0.92, 1.34, 0.94], wspace=0.28)
    figure8_label_confidence_panel(fig.add_subplot(top[0, 0]))
    figure8_cell_count_panel(fig.add_subplot(top[0, 1]))
    figure8_state_mapping_panel(fig.add_subplot(top[0, 2]))
    figure8_caveat_matrix_panel(fig.add_subplot(bottom[0, 0]))
    figure8_leaderboard_panel(fig.add_subplot(bottom[0, 1]))
    figure8_failure_panel(fig.add_subplot(bottom[0, 2]))
    fig.subplots_adjust(left=0.050, right=0.992, top=0.955, bottom=0.060)
    return export_figure(fig, 'figure8_robustness_and_failure_cases', out_dir, 330, 194)


def write_figure_manifests(paths: list[Path]) -> None:
    ensure_dir(FIG_DIR / 'tables')
    manifest = pd.DataFrame(
        [{'figure_stem': path.stem, 'format': path.suffix.lstrip('.'), 'path': str(path)} for path in paths]
    )
    manifest.to_csv(FIG_DIR / 'tables' / 'generated_figure_manifest.csv', index=False)
    summary = (
        manifest.groupby('figure_stem', as_index=False)
        .agg(n_formats=('format', 'size'), formats=('format', lambda values: ','.join(sorted(values))))
        .sort_values('figure_stem')
    )
    summary.to_csv(FIG_DIR / 'tables' / 'generated_figure_summary.csv', index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    configure_style()
    outputs = []
    outputs.extend(build_figure1(out_dir))
    outputs.extend(build_figure2(out_dir))
    outputs.extend(build_figure3(out_dir))
    outputs.extend(build_figure4(out_dir))
    outputs.extend(build_figure5(out_dir))
    outputs.extend(build_figure6(out_dir))
    outputs.extend(build_figure7(out_dir))
    outputs.extend(build_figure8(out_dir))
    write_figure_manifests(outputs)
    print('\n'.join(str(path) for path in outputs))


if __name__ == '__main__':
    main()
