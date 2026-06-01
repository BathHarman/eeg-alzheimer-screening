"""
EEG-Based Biomarker Analysis for Early Alzheimer's Screening
Dataset: OpenNeuro ds004504
88 subjects: 36 AD, 29 HC, 23 FTD — ages 49-78
Resting-state eyes-closed EEG recordings

Author: Harman Bath
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
from scipy import stats
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIGURATION — update DATA_ROOT to your path
# ─────────────────────────────────────────────
DATA_ROOT = Path("./ds004504")          # root of the OpenNeuro dataset
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# Frequency bands (Hz)
BANDS = {
    "delta":  (1,   4),
    "theta":  (4,   8),
    "alpha":  (8,  13),
    "beta":   (13, 30),
    "gamma":  (30, 45),
}

# Groups present in ds004504
GROUPS = {
    "AD":  "Alzheimer's Disease",
    "HC":  "Healthy Control",
    "FTD": "Frontotemporal Dementia",
}


# ─────────────────────────────────────────────
# STEP 1: DISCOVER SUBJECT FILES
# ─────────────────────────────────────────────
def discover_subjects(data_root: Path) -> dict[str, list[Path]]:
    """
    Walk ds004504 BIDS structure and return a dict mapping
    group label -> list of .set file paths.

    Expects participants.tsv at dataset root with a 'Group' column.
    """
    participants_file = data_root / "participants.tsv"
    if not participants_file.exists():
        raise FileNotFoundError(
            f"Could not find participants.tsv at {participants_file}.\n"
            "Make sure DATA_ROOT points to the ds004504 dataset root."
        )

    participants = pd.read_csv(participants_file, sep="\t")
    # normalize column names
    participants.columns = [c.strip().lower() for c in participants.columns]

    group_col = next(
        (c for c in participants.columns if "group" in c), None
    )
    if group_col is None:
        raise ValueError("Could not find a 'group' column in participants.tsv")

    subject_files: dict[str, list[Path]] = {g: [] for g in GROUPS}

    for _, row in participants.iterrows():
        sub_id = str(row["participant_id"]).strip()
        group  = str(row[group_col]).strip().upper()

        if group not in GROUPS:
            continue

        # BIDS path: sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set
        eeg_dir  = data_root / sub_id / "eeg"
        set_files = list(eeg_dir.glob("*eyesclosed*eeg.set"))

        if set_files:
            subject_files[group].append(set_files[0])
        else:
            print(f"  [warn] No .set file found for {sub_id} ({group})")

    for g, files in subject_files.items():
        print(f"  {g}: {len(files)} subjects found")

    return subject_files


# ─────────────────────────────────────────────
# STEP 2: COMPUTE PSD + BAND POWER PER SUBJECT
# ─────────────────────────────────────────────
def compute_band_powers(set_path: Path) -> dict[str, float] | None:
    """
    Load a single .set file, compute PSD via Welch's method,
    and return mean power in each canonical frequency band.
    Returns None if the file cannot be loaded.
    """
    try:
        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)

        # Basic preprocessing: bandpass 1-45 Hz, re-reference to average
        raw.filter(1.0, 45.0, fir_design="firwin", verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)

        # Welch PSD — 4-second windows, 50% overlap
        psd = raw.compute_psd(
            method="welch",
            fmin=1.0,
            fmax=45.0,
            n_fft=int(raw.info["sfreq"] * 4),
            n_overlap=int(raw.info["sfreq"] * 2),
            verbose=False,
        )
        freqs        = psd.freqs
        power_matrix = psd.get_data()          # shape: (channels, freqs)
        mean_power   = power_matrix.mean(axis=0)  # average across channels

        band_powers: dict[str, float] = {}
        for band, (flo, fhi) in BANDS.items():
            idx = (freqs >= flo) & (freqs <= fhi)
            band_powers[band] = float(np.mean(mean_power[idx]))

        # Primary biomarker
        band_powers["theta_alpha_ratio"] = (
            band_powers["theta"] / band_powers["alpha"]
        )

        return band_powers

    except Exception as exc:
        print(f"  [error] {set_path.name}: {exc}")
        return None


# ─────────────────────────────────────────────
# STEP 3: BUILD RESULTS DATAFRAME
# ─────────────────────────────────────────────
def build_dataframe(subject_files: dict[str, list[Path]]) -> pd.DataFrame:
    """Process all subjects and return a tidy DataFrame."""
    records = []

    for group, paths in subject_files.items():
        print(f"\nProcessing {GROUPS[group]} ({len(paths)} subjects)...")
        for path in paths:
            powers = compute_band_powers(path)
            if powers is not None:
                records.append({"subject": path.stem, "group": group, **powers})

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "band_powers.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'band_powers.csv'}")
    return df


# ─────────────────────────────────────────────
# STEP 4: STATISTICAL ANALYSIS
# ─────────────────────────────────────────────
def run_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run independent-samples t-tests comparing AD vs HC
    on each frequency band and the theta/alpha ratio.
    """
    metrics = list(BANDS.keys()) + ["theta_alpha_ratio"]
    ad_data = df[df["group"] == "AD"]
    hc_data = df[df["group"] == "HC"]

    rows = []
    for metric in metrics:
        ad_vals = ad_data[metric].dropna().values
        hc_vals = hc_data[metric].dropna().values

        t_stat, p_val = stats.ttest_ind(ad_vals, hc_vals, equal_var=False)
        cohens_d = (ad_vals.mean() - hc_vals.mean()) / np.sqrt(
            (ad_vals.std() ** 2 + hc_vals.std() ** 2) / 2
        )

        rows.append({
            "metric":       metric,
            "AD_mean":      ad_vals.mean(),
            "AD_std":       ad_vals.std(),
            "HC_mean":      hc_vals.mean(),
            "HC_std":       hc_vals.std(),
            "t_statistic":  t_stat,
            "p_value":      p_val,
            "cohens_d":     cohens_d,
            "significant":  p_val < 0.05,
        })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(RESULTS_DIR / "statistics.csv", index=False)

    print("\n── Statistical Results (AD vs HC) ──────────────────────────")
    print(stats_df[["metric", "AD_mean", "HC_mean", "p_value", "cohens_d", "significant"]].to_string(index=False))
    return stats_df


# ─────────────────────────────────────────────
# STEP 5: VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_results(df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    """Generate and save all figures."""

    GROUP_COLORS = {"AD": "#E63946", "HC": "#457B9D", "FTD": "#F4A261"}
    GROUP_LABELS = GROUPS

    # ── Figure 1: Theta/Alpha Ratio by Group ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    groups_present = [g for g in GROUPS if g in df["group"].unique()]
    means  = [df[df["group"] == g]["theta_alpha_ratio"].mean() for g in groups_present]
    sems   = [df[df["group"] == g]["theta_alpha_ratio"].sem()  for g in groups_present]
    colors = [GROUP_COLORS[g] for g in groups_present]
    labels = [GROUP_LABELS[g] for g in groups_present]

    bars = ax.bar(labels, means, yerr=sems, color=colors,
                  capsize=6, edgecolor="white", linewidth=1.2, width=0.5)

    # Significance annotation (AD vs HC)
    row = stats_df[stats_df["metric"] == "theta_alpha_ratio"].iloc[0]
    if row["significant"]:
        y_top = max(means) + max(sems) + 0.05
        ax.annotate(
            f"p = {row['p_value']:.4f} *",
            xy=(0.5, y_top), xycoords=("axes fraction", "data"),
            ha="center", fontsize=10, color="#333333"
        )

    ax.set_ylabel("Theta / Alpha Power Ratio", fontsize=12)
    ax.set_title("Theta/Alpha Ratio by Diagnostic Group\n(Mean ± SEM)", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "theta_alpha_ratio.png", dpi=150)
    plt.close(fig)

    # ── Figure 2: Band Power Heatmap ────────────────────────────────
    band_names = list(BANDS.keys())
    group_means = np.array([
        [df[df["group"] == g][b].mean() for b in band_names]
        for g in groups_present
    ])
    # Normalize each band to [0,1] for visual comparison
    norm = (group_means - group_means.min(axis=0)) / (
        group_means.max(axis=0) - group_means.min(axis=0) + 1e-12
    )

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(norm, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(band_names)))
    ax.set_xticklabels([b.capitalize() for b in band_names], fontsize=11)
    ax.set_yticks(range(len(groups_present)))
    ax.set_yticklabels(labels, fontsize=11)
    plt.colorbar(im, ax=ax, label="Normalized Power")
    ax.set_title("Relative Band Power by Group", fontsize=13)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "band_power_heatmap.png", dpi=150)
    plt.close(fig)

    # ── Figure 3: Boxplot — Theta/Alpha Ratio ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    data_to_plot = [df[df["group"] == g]["theta_alpha_ratio"].dropna().values
                    for g in groups_present]
    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Theta / Alpha Power Ratio", fontsize=12)
    ax.set_title("Distribution of Theta/Alpha Ratio by Group", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "theta_alpha_boxplot.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures saved to {RESULTS_DIR}/")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("EEG Biomarker Analysis — Alzheimer's Screening")
    print("Dataset: OpenNeuro ds004504")
    print("=" * 55)

    print("\n[1/4] Discovering subjects...")
    subject_files = discover_subjects(DATA_ROOT)

    print("\n[2/4] Computing band powers...")
    df = build_dataframe(subject_files)

    if df.empty:
        print("\n[ERROR] No data was loaded. Check DATA_ROOT path and file structure.")
        return

    print(f"\nTotal subjects processed: {len(df)}")
    print(df.groupby("group").size().rename("n").to_string())

    print("\n[3/4] Running statistical analysis...")
    stats_df = run_statistics(df)

    print("\n[4/4] Generating figures...")
    plot_results(df, stats_df)

    print("\n✓ Analysis complete.")
    print(f"  Results: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
