import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "aggregate" / "injection"
ADDITIONAL_ANALYSIS = ROOT / "results" / "aggregate" / "additional_context"
OUT = ROOT / "submission_figures"
OUT.mkdir(parents=True, exist_ok=True)

MM_TO_INCH = 1.0 / 25.4
FIGURE_WIDTH_IN = 174.0 * MM_TO_INCH

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "savefig.dpi": 600,
        "savefig.bbox": None,
    }
)

COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "teal": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#E69F00",
    "gray": "#6B7280",
    "light": "#F3F4F6",
    "dark": "#1F2937",
}

MODEL_LABEL = {
    "attentive_fp": "AttentiveFP",
    "d_mpnn": "D-MPNN",
    "gcn": "GCN",
    "gine": "GINE",
}
CONDITION_LABEL = {
    "structure_only": "Structure only",
    "onehot_late": "One-hot, late",
    "onehot_early": "One-hot, early",
    "physiology_late": "Physiology-based, late",
    "physiology_early": "Physiology-based, early",
    "additive_tissue_intercept": "Molecule + tissue intercept",
}


def save(fig, stem):
    png_path = OUT / f"{stem}.png"
    tiff_path = OUT / f"{stem}.tiff"
    fig.patch.set_facecolor("white")
    for axis in fig.axes:
        axis.set_facecolor("white")
    fig.savefig(png_path, dpi=600, facecolor="white", transparent=False)
    fig.savefig(
        tiff_path,
        dpi=600,
        facecolor="white",
        transparent=False,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    with Image.open(png_path) as image:
        if image.mode != "RGB":
            image.convert("RGB").save(png_path, dpi=(600, 600))
    with Image.open(tiff_path) as image:
        if image.mode != "RGB":
            image.convert("RGB").save(
                tiff_path,
                compression="tiff_lzw",
                dpi=(600, 600),
            )


def box(ax, xy, width, height, text, face, edge=None, fontsize=9, weight="normal"):
    edge = edge or face
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color="white" if face != COLORS["light"] else COLORS["dark"],
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, color=COLORS["gray"], style="-|>", connection="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=1.1,
            color=color,
            connectionstyle=connection,
        )
    )


def figure_1():
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIGURE_WIDTH_IN, 3.0),
        gridspec_kw={"wspace": 0.28},
    )
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.text(-0.05, 0.98, "a", va="top", fontweight="bold", fontsize=12)
    ax.text(0.53, 0.96, "Inputs and five conditions", ha="center", va="top", fontweight="bold")
    box(ax, (0.02, 0.70), 0.42, 0.14, "SMILES\nmolecular graph", COLORS["blue"], fontsize=8, weight="bold")
    box(ax, (0.56, 0.70), 0.42, 0.14, "Tissue context", COLORS["orange"], fontsize=8, weight="bold")
    box(ax, (0.46, 0.46), 0.23, 0.13, "11-class\none-hot", COLORS["purple"], fontsize=8)
    box(ax, (0.74, 0.46), 0.24, 0.13, "5 tissue\ndescriptors", COLORS["teal"], fontsize=8)
    arrow(ax, (0.77, 0.70), (0.66, 0.59))
    arrow(ax, (0.77, 0.70), (0.86, 0.59))
    box(ax, (0.02, 0.12), 0.96, 0.21, "Structure only\nOne-hot: early or late\nPhysiology-based: early or late", COLORS["light"], edge=COLORS["gray"], fontsize=8)
    arrow(ax, (0.23, 0.70), (0.30, 0.33))
    arrow(ax, (0.58, 0.46), (0.58, 0.33))
    arrow(ax, (0.86, 0.46), (0.76, 0.33))

    ax = axes[1]
    ax.text(-0.05, 0.98, "b", va="top", fontweight="bold", fontsize=12)
    ax.text(0.53, 0.96, "Where context enters", ha="center", va="top", fontweight="bold")
    box(ax, (0.05, 0.68), 0.26, 0.13, "Atom/bond\nfeatures", COLORS["blue"], fontsize=8)
    box(ax, (0.38, 0.68), 0.26, 0.13, "Message\npassing", COLORS["blue"], fontsize=8)
    box(ax, (0.71, 0.68), 0.24, 0.13, "Pooling", COLORS["blue"], fontsize=8)
    arrow(ax, (0.31, 0.745), (0.38, 0.745))
    arrow(ax, (0.64, 0.745), (0.71, 0.745))
    box(ax, (0.05, 0.39), 0.26, 0.13, "Context", COLORS["orange"], fontsize=8)
    arrow(ax, (0.18, 0.52), (0.18, 0.68), color=COLORS["orange"])
    ax.text(0.29, 0.59, "early", ha="center", color=COLORS["orange"], fontweight="bold", fontsize=8)
    box(ax, (0.70, 0.39), 0.25, 0.13, "Context", COLORS["orange"], fontsize=8)
    box(ax, (0.36, 0.16), 0.30, 0.13, "Prediction\nhead", COLORS["dark"], fontsize=8)
    arrow(ax, (0.83, 0.68), (0.57, 0.29))
    arrow(ax, (0.82, 0.39), (0.60, 0.29), color=COLORS["orange"])
    ax.text(0.77, 0.33, "late", ha="center", color=COLORS["orange"], fontweight="bold", fontsize=8)
    box(ax, (0.36, 0.01), 0.30, 0.10, r"$\log_{10}(K_p)$", COLORS["gray"], fontsize=8)
    arrow(ax, (0.51, 0.16), (0.51, 0.11))

    ax = axes[2]
    ax.text(-0.05, 0.98, "c", va="top", fontweight="bold", fontsize=12)
    ax.text(0.53, 0.96, "Evaluation hierarchy", ha="center", va="top", fontweight="bold")
    box(ax, (0.03, 0.68), 0.94, 0.17, "4 GNN architectures\nGCN · GINE\nD-MPNN · AttentiveFP", COLORS["dark"], fontsize=8)
    box(ax, (0.02, 0.37), 0.47, 0.25, "Repeated splits\nParent group\nand scaffold\n10 seeds", COLORS["blue"], fontsize=8)
    box(ax, (0.51, 0.37), 0.47, 0.25, "LOTO\n11 held-out tissues\n5 training seeds", COLORS["teal"], fontsize=8)
    arrow(ax, (0.37, 0.70), (0.28, 0.61))
    arrow(ax, (0.63, 0.70), (0.72, 0.61))
    box(ax, (0.03, 0.13), 0.94, 0.16, "RMSE, MAE, fold accuracy\nPaired inference and robustness", COLORS["light"], edge=COLORS["gray"], fontsize=8)
    arrow(ax, (0.28, 0.47), (0.38, 0.35))
    arrow(ax, (0.72, 0.47), (0.62, 0.35))
    ax.text(0.5, 0.035, "Separate PT/RR paper benchmarks\nand matched OOF direct panels", ha="center", va="center", color=COLORS["orange"], fontweight="bold", fontsize=8)
    arrow(ax, (0.5, 0.17), (0.5, 0.10), color=COLORS["orange"])

    fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.06)
    save(fig, "Fig1")


def heatmap(ax, data, split_label, vmin, vmax):
    models = ["attentive_fp", "d_mpnn", "gcn", "gine"]
    conditions = [
        "structure_only",
        "additive_tissue_intercept",
        "onehot_late",
        "onehot_early",
        "physiology_late",
        "physiology_early",
    ]
    piv = data[data["split_type"] == split_label].pivot(index="model", columns="condition", values="mean_rmse")
    arr = piv.reindex(index=models, columns=conditions).to_numpy()
    im = ax.imshow(arr, cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="auto")
    tick_labels = {
        "structure_only": "Structure\nonly",
        "additive_tissue_intercept": "Additive\nintercept",
        "onehot_late": "One-hot\nlate",
        "onehot_early": "One-hot\nearly",
        "physiology_late": "Physiology-based\nlate",
        "physiology_early": "Physiology-based\nearly",
    }
    ax.set_xticks(
        range(len(conditions)),
        [tick_labels[c] for c in conditions],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(models)), [MODEL_LABEL[m] for m in models])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if arr[i, j] > (vmin + vmax) / 2 else "black")
    ax.set_title("Parent-group split" if split_label == "parent_group" else "Scaffold split", fontweight="bold")
    return im


def forest(ax, data, title, xlab, color_map, label_fn):
    plot = data.copy().sort_values(["split_type", "model", "comparison"])
    y = np.arange(len(plot))[::-1]
    for idx, (_, row) in enumerate(plot.iterrows()):
        yy = y[idx]
        key = row["comparison"]
        c = color_map.get(key, COLORS["gray"])
        marker = "o" if row["split_type"] == "parent_group" else "s"
        ax.errorbar(
            row["mean_delta_rmse"],
            yy,
            xerr=[[row["mean_delta_rmse"] - row["ci95_low"]], [row["ci95_high"] - row["mean_delta_rmse"]]],
            fmt=marker,
            color=c,
            ecolor=c,
            elinewidth=1,
            capsize=2,
            markersize=4,
        )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, [label_fn(r) for _, r in plot.iterrows()])
    ax.set_xlabel(xlab)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.5)
    return plot


def figure_2():
    perf = pd.read_csv(ANALYSIS / "deep_dive" / "primary_performance_summary.csv")
    additive_jobs = pd.read_csv(ADDITIONAL_ANALYSIS / "all_additional_job_metrics.csv")
    additive_perf = (
        additive_jobs.query("analysis == 'primary_context_decomposition'")
        .groupby(["model", "condition", "split_type"], as_index=False)
        .agg(mean_rmse=("rmse_log10", "mean"))
    )
    perf = pd.concat([perf, additive_perf], ignore_index=True, sort=False)
    ctx = pd.read_csv(ANALYSIS / "part1_context_effects.csv").query("analysis_set == 'primary'")
    pos = pd.read_csv(ANALYSIS / "part1_position_effects.csv").query("analysis_set == 'primary'")

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 9.20))
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.90, 0.10, 2.80, 1.55],
        left=0.36,
        right=0.98,
        top=0.97,
        bottom=0.06,
        hspace=0.74,
        wspace=0.46,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, :])
    ax_d = fig.add_subplot(gs[3, :])
    vmin, vmax = perf.mean_rmse.min(), perf.mean_rmse.max()
    im = heatmap(ax_a, perf, "parent_group", vmin, vmax)
    heatmap(ax_b, perf, "scaffold", vmin, vmax)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"Mean test RMSE in $\log_{10}(K_p)$")

    forest(
        ax_c,
        ctx,
        "Context versus structure only",
        "Paired ΔRMSE (context − structure)",
        {
            "onehot_early": COLORS["blue"],
            "onehot_late": COLORS["teal"],
            "physiology_early": COLORS["orange"],
            "physiology_late": COLORS["purple"],
        },
        lambda r: f"{MODEL_LABEL[r['model']]} · {'P' if r['split_type']=='parent_group' else 'S'} · {CONDITION_LABEL[r['comparison']]}",
    )
    ax_c.set_xlim(-0.13, 0.015)
    ax_c.text(0.01, -0.09, "Favors context", transform=ax_c.transAxes, ha="left", va="top", color=COLORS["teal"], fontsize=8)

    forest(
        ax_d,
        pos,
        "Early versus late fusion",
        "Paired ΔRMSE (early − late)",
        {"onehot_early": COLORS["blue"], "physiology_early": COLORS["orange"]},
        lambda r: f"{MODEL_LABEL[r['model']]} · {'P' if r['split_type']=='parent_group' else 'S'} · {'One-hot' if r['comparison']=='onehot_early' else 'Physiology-based'}",
    )
    lim = max(abs(pos.ci95_low.min()), abs(pos.ci95_high.max())) * 1.15
    ax_d.set_xlim(-lim, lim)
    ax_d.text(0.01, -0.13, "Favors early", transform=ax_d.transAxes, ha="left", va="top", color=COLORS["blue"], fontsize=8)
    ax_d.text(0.99, -0.13, "Favors late", transform=ax_d.transAxes, ha="right", va="top", color=COLORS["orange"], fontsize=8)

    for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        ax.text(-0.22 if label in "ab" else -0.38, 1.08 if label in "ab" else 1.03, label,
                transform=ax.transAxes, fontweight="bold", fontsize=12, va="top")
    save(fig, "Fig2")


def figure_3():
    inf = pd.read_csv(ANALYSIS / "loto_inference.csv")
    extra = pd.read_csv(ANALYSIS / "deep_dive" / "loto_context_extrapolation.csv")
    raw_inf = pd.read_csv(ADDITIONAL_ANALYSIS / "loto_raw_inference.csv")
    comparison = pd.read_csv(ADDITIONAL_ANALYSIS / "loto_preprocessing_comparison.csv")

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 4.15))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[0.86, 1.20, 1.05],
        left=0.13,
        right=0.98,
        top=0.82,
        bottom=0.22,
        wspace=0.55,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    standardized = inf[inf["contrast"] == "physiology_early_vs_late"].copy()
    standardized["label"] = standardized.model.map(MODEL_LABEL)
    standardized = standardized.sort_values("mean_tissue_equal_delta_rmse")
    raw_inf = raw_inf.set_index("model").loc[standardized.model].reset_index()
    y = np.arange(len(standardized))
    ax_a.errorbar(
        standardized.mean_tissue_equal_delta_rmse,
        y + 0.08,
        xerr=[
            standardized.mean_tissue_equal_delta_rmse - standardized.ci95_low,
            standardized.ci95_high - standardized.mean_tissue_equal_delta_rmse,
        ],
        fmt="o",
        color=COLORS["orange"],
        ecolor=COLORS["orange"],
        capsize=3,
        markersize=4.5,
        label="Training-tissue z score",
    )
    ax_a.errorbar(
        raw_inf.mean_tissue_equal_early_minus_late_rmse,
        y - 0.08,
        xerr=[
            raw_inf.mean_tissue_equal_early_minus_late_rmse - raw_inf.ci95_low,
            raw_inf.ci95_high - raw_inf.mean_tissue_equal_early_minus_late_rmse,
        ],
        fmt="s",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        capsize=3,
        markersize=4,
        label="Unit-harmonized, unstandardized",
    )
    ax_a.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax_a.set_yticks(y, standardized.label)
    ax_a.set_xlabel("Tissue-equal ΔRMSE\n(early − late)")
    ax_a.set_title("LOTO model-level effects", fontweight="bold")
    ax_a.grid(axis="x", color="#D1D5DB", linewidth=0.5)

    non_adipose = comparison.query("Tissue != 'adipose'")
    ax_b.scatter(
        non_adipose.standardized_early_minus_late_rmse,
        non_adipose.early_minus_late_rmse,
        s=29,
        color=COLORS["blue"],
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
    )
    adipose = comparison.query("Tissue == 'adipose'")
    for _, row in adipose.iterrows():
        ax_b.scatter(
            row.standardized_early_minus_late_rmse,
            row.early_minus_late_rmse,
            s=45,
            marker="D",
            color=COLORS["orange"],
            edgecolor="black",
            linewidth=0.5,
        )
    limits = [
        min(comparison.standardized_early_minus_late_rmse.min(), comparison.early_minus_late_rmse.min()) - 0.08,
        max(comparison.standardized_early_minus_late_rmse.max(), comparison.early_minus_late_rmse.max()) + 0.12,
    ]
    ax_b.plot(limits, limits, color=COLORS["gray"], linestyle=":", linewidth=0.9)
    ax_b.axhline(0, color="black", linestyle="--", linewidth=0.6)
    ax_b.axvline(0, color="black", linestyle="--", linewidth=0.6)
    ax_b.set_xlim(limits)
    ax_b.set_ylim(limits)
    ax_b.set_xlabel("Standardized early − late RMSE")
    ax_b.set_ylabel("Unstandardized early − late RMSE")
    ax_b.set_title("Preprocessing sensitivity", fontweight="bold")
    ax_b.grid(color="#E5E7EB", linewidth=0.45)

    mean_gap = (
        comparison.groupby("Tissue", as_index=False)
        .agg(
            standardized_gap=("standardized_early_minus_late_rmse", "mean"),
            raw_gap=("early_minus_late_rmse", "mean"),
        )
    )
    joined = extra.merge(mean_gap, on="Tissue", how="inner")
    ax_c.scatter(
        np.log10(joined.max_abs_train_z),
        joined.standardized_gap,
        s=38,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.6,
        label="Training-tissue z score",
    )
    ax_c.scatter(
        np.log10(joined.max_abs_train_z),
        joined.raw_gap,
        s=32,
        marker="s",
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.6,
        label="Unit-harmonized, unstandardized",
    )
    for _, row in joined.iterrows():
        if row.Tissue in {"adipose", "skin"}:
            ax_c.annotate(
                row.Tissue.capitalize(),
                (np.log10(row.max_abs_train_z), max(row.standardized_gap, row.raw_gap)),
                xytext=(-5, -14) if row.Tissue == "adipose" else (4, 4),
                textcoords="offset points",
                fontsize=8,
                ha="right" if row.Tissue == "adipose" else "left",
            )
    ax_c.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_c.set_xlabel(r"$\log_{10}[d(t)]$")
    ax_c.set_ylabel("Mean early − late RMSE")
    ax_c.set_title("Descriptor extrapolation", fontweight="bold")
    ax_c.grid(color="#D1D5DB", linewidth=0.5)
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.98),
        fontsize=8,
    )

    for label, axis in zip("abc", [ax_a, ax_b, ax_c]):
        axis.text(
            {"a": -0.34, "b": -0.22, "c": -0.22}[label],
            1.05,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=12,
            va="top",
        )
    save(fig, "Fig3")


def figure_4():
    benchmark_dir = ANALYSIS / "separate_literature_benchmarks"
    benchmark = pd.read_csv(benchmark_dir / "direct_panel_metrics.csv")
    chemistry = pd.read_csv(
        benchmark_dir / "chemical_space" / "chemical_space_compounds.csv",
        encoding="utf-8-sig",
    )
    summary = json.loads(
        (benchmark_dir / "chemical_space" / "chemical_space_summary.json").read_text(
            encoding="utf-8"
        )
    )

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 9.0))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.20, 1.00],
        left=0.12,
        right=0.98,
        top=0.96,
        bottom=0.07,
        hspace=0.58,
        wspace=0.42,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])

    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    ax_a.set_title("Separated evaluation domains and direct panels", fontweight="bold", pad=5)
    box(ax_a, (0.02, 0.73), 0.30, 0.18, "GNN internal OOF\nKp_Data target", COLORS["blue"], fontsize=8)
    box(
        ax_a,
        (0.35, 0.73),
        0.30,
        0.18,
        "PT source-paper evaluation\nPrediction vs experimental $K_p$",
        COLORS["teal"],
        fontsize=8,
    )
    box(
        ax_a,
        (0.68, 0.73),
        0.30,
        0.18,
        "RR source-paper evaluation\nPrediction vs experimental $K_p$",
        COLORS["orange"],
        fontsize=8,
    )
    box(
        ax_a,
        (0.10, 0.46),
        0.80,
        0.15,
        "Separate evaluations retain their source-specific targets\n(no substitution with Kp_Data)",
        COLORS["light"],
        edge=COLORS["gray"],
        fontsize=8,
    )
    arrow(ax_a, (0.17, 0.73), (0.28, 0.61), color=COLORS["blue"])
    arrow(ax_a, (0.50, 0.73), (0.50, 0.61), color=COLORS["teal"])
    arrow(ax_a, (0.83, 0.73), (0.72, 0.61), color=COLORS["orange"])
    box(
        ax_a,
        (0.10, 0.24),
        0.80,
        0.12,
        "Exact canonical SMILES–tissue links to held-out GNN predictions\n17 molecular inputs available for OOF linkage",
        COLORS["dark"],
        fontsize=8,
    )
    arrow(ax_a, (0.50, 0.46), (0.50, 0.36))
    panel_counts = {
        (method, split): int(
            benchmark[
                (benchmark["benchmark_method"] == method)
                & (benchmark["split_type"] == split)
            ]["n_rows"].iloc[0]
        )
        for method in ["PT", "RR"]
        for split in ["parent_group", "scaffold"]
    }
    box(
        ax_a,
        (0.15, 0.03),
        0.70,
        0.12,
        "Matched direct panels\n"
        f"PT n = {panel_counts[('PT', 'parent_group')]}/{panel_counts[('PT', 'scaffold')]}; "
        f"RR n = {panel_counts[('RR', 'parent_group')]}/{panel_counts[('RR', 'scaffold')]} "
        "(parent-group/scaffold)",
        COLORS["purple"],
        fontsize=8,
    )
    arrow(ax_a, (0.50, 0.24), (0.50, 0.15))

    outside = chemistry[~chemistry["in_evaluated_union"]]
    inside = chemistry[chemistry["in_evaluated_union"]]
    ax_b.scatter(
        outside.pca1,
        outside.pca2,
        s=20,
        color="#CBD5E1",
        edgecolor="white",
        linewidth=0.35,
        label=f"Other Kp_Data SMILES (n = {len(outside)})",
    )
    ax_b.scatter(
        inside.pca1,
        inside.pca2,
        s=42,
        marker="D",
        color=COLORS["orange"],
        edgecolor="black",
        linewidth=0.45,
        label=f"Direct-panel union (n = {len(inside)})",
    )
    explained = 100 * np.asarray(summary["pca_explained_variance_ratio"])
    ax_b.set_xlabel(f"Fingerprint PC1 ({explained[0]:.1f}% variance)")
    ax_b.set_ylabel(f"Fingerprint PC2 ({explained[1]:.1f}% variance)")
    ax_b.set_title("Morgan-fingerprint chemical space", fontweight="bold")
    ax_b.legend(frameon=False, loc="best")
    ax_b.grid(color="#E5E7EB", linewidth=0.5)

    similarities = np.sort(outside["max_tanimoto_to_evaluated_union"].to_numpy(float))
    cumulative = np.arange(1, len(similarities) + 1) / len(similarities)
    ax_c.step(similarities, cumulative, where="post", color=COLORS["blue"], linewidth=1.8)
    for threshold, color in [(0.4, COLORS["orange"]), (0.6, COLORS["purple"])]:
        ax_c.axvline(threshold, color=color, linestyle="--", linewidth=0.9)
    median = summary["coverage"]["nonbenchmark_max_tanimoto"]["median"]
    ax_c.axvline(median, color=COLORS["dark"], linestyle=":", linewidth=1.1)
    ax_c.text(
        median + 0.012,
        0.08,
        f"median = {median:.3f}",
        rotation=90,
        va="bottom",
        fontsize=8,
    )
    ax_c.text(
        0.98,
        0.08,
        f"{summary['evaluated_union']['scaffold_groups']}/91 scaffold groups\n"
        f"{100 * summary['coverage']['full_smiles_sharing_evaluated_scaffold_fraction']:.1f}% of full SMILES share\n"
        "a direct-panel scaffold",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#9CA3AF"},
    )
    ax_c.set_xlim(0, 0.8)
    ax_c.set_ylim(0, 1.02)
    ax_c.set_xlabel("Maximum Tanimoto similarity to direct-panel union")
    ax_c.set_ylabel("Cumulative fraction of nonbenchmark SMILES")
    ax_c.set_title("Structural coverage", fontweight="bold")
    ax_c.grid(color="#E5E7EB", linewidth=0.5)

    panels = [("PT", "parent_group"), ("PT", "scaffold"), ("RR", "parent_group"), ("RR", "scaffold")]
    x = np.arange(len(panels))
    width = 0.34
    for offset, evaluated_method, color, hatch, label in [
        (-width / 2, "GNN", COLORS["blue"], "", "Reduced-input GNN"),
        (width / 2, "baseline", COLORS["orange"], "//", "PT or RR equation"),
    ]:
        rows = []
        for conventional, split in panels:
            method = "GNN" if evaluated_method == "GNN" else conventional
            rows.append(
                benchmark[
                    (benchmark.benchmark_method == conventional)
                    & (benchmark.split_type == split)
                    & (benchmark.method == method)
                ].iloc[0]
            )
        values = [row.rmse_log10 for row in rows]
        lower = [row.rmse_log10 - row.ci95_rmse_log10_low for row in rows]
        upper = [row.ci95_rmse_log10_high - row.rmse_log10 for row in rows]
        ax_d.bar(
            x + offset,
            values,
            width,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            label=label,
            yerr=[lower, upper],
            capsize=2.5,
            error_kw={"linewidth": 0.7},
        )
        for xx, value in zip(x + offset, values):
            ax_d.text(xx, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax_d.set_xticks(x, ["PT\nParent", "PT\nScaffold", "RR\nParent", "RR\nScaffold"])
    ax_d.set_ylabel(r"RMSE in $\log_{10}(K_p)$")
    ax_d.set_ylim(0, 0.82)
    ax_d.set_title("Matched literature-data direct comparisons", fontweight="bold")
    ax_d.legend(frameon=False, ncol=2, loc="upper left")
    ax_d.grid(axis="y", color="#D1D5DB", linewidth=0.5)

    for label, axis in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        axis.text(
            -0.06 if label == "a" else (-0.08 if label == "d" else -0.18),
            1.04,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=12,
            va="top",
        )
    save(fig, "Fig4")


if __name__ == "__main__":
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    print(f"Wrote figures to {OUT}")
