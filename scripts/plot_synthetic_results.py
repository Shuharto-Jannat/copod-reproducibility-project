"""Plot five-seed synthetic results without rerunning the models."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from summarise_synthetic_results import load_results


def draw_plot(data, output_dir):
    scenarios = ["marginal_shift", "dependence_violation"]
    methods = ["pyod_batch", "fixed_training_reference"]
    labels = [
        "Standard PyOD (batch)",
        "Modified training reference",
    ]
    colors = ["#0072B2", "#D55E00"]
    markers = ["o", "s"]
    x = np.arange(2)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.23,
        top=0.74, wspace=0.28,
    )

    fig.suptitle(
        "COPOD on controlled synthetic data",
        fontsize=21, y=0.96,
    )
    fig.text(
        0.5, 0.90,
        "Five data seeds (42–46) | 20 features | 5% anomalies",
        ha="center", fontsize=13,
    )

    panels = [
        ("roc_auc", "ROC AUC", 0.5, 1.0, "Random ranking: 0.50"),
        (
            "average_precision", "Average precision",
            0.05, 0.16, "Prevalence reference: 0.05",
        ),
    ]

    for ax, panel in zip(axes, panels):
        metric, title, baseline, upper, baseline_label = panel

        for j, method in enumerate(methods):
            for i, scenario in enumerate(scenarios):
                values = data.loc[
                    data["scenario"].eq(scenario)
                    & data["method"].eq(method)
                ].sort_values("data_seed")[metric].to_numpy()

                mean = values.mean()
                sd = values.std(ddof=1)
                position = x[i] + [-0.14, 0.14][j]

                # Small dots show all five individual seed results.
                ax.scatter(
                    position + np.linspace(-0.045, 0.045, len(values)),
                    values,
                    color=colors[j],
                    s=24,
                    alpha=0.45,
                    zorder=3,
                )

                # Large symbol = mean; error bars = one sample SD.
                ax.errorbar(
                    position,
                    mean,
                    yerr=sd,
                    fmt=markers[j],
                    color=colors[j],
                    markersize=8,
                    capsize=6,
                    elinewidth=2,
                    zorder=4,
                )

                ax.annotate(
                    format(mean, ".3f"),
                    (position, mean + sd),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    color=colors[j],
                    fontsize=11,
                )

        ax.axhline(
            baseline,
            color="#666666",
            linestyle="--",
            linewidth=1.3,
        )
        ax.text(
            0.03,
            baseline + upper * 0.025,
            baseline_label,
            transform=ax.get_yaxis_transform(),
            fontsize=10,
            color="#555555",
        )

        ax.set_title(title + " (higher is better)", pad=12)
        ax.set_xticks(x, ["Marginal shift", "Dependence violation"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, upper)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D(
            [0], [0],
            color=color,
            marker=marker,
            linestyle="none",
            markersize=8,
            label=label,
        )
        for color, marker, label in zip(colors, markers, labels)
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.87),
        ncol=2,
        frameon=False,
    )

    fig.text(
        0.5, 0.11,
        "Small dots: individual seeds. Large symbols: means. "
        "Error bars: ±1 sample SD (not confidence intervals).",
        ha="center", fontsize=10,
    )
    fig.text(
        0.5, 0.065,
        "Each run: 6,000 training / 4,000 test rows. "
        "AP axis is zoomed to 0–0.16; its full range is 0–1.",
        ha="center", fontsize=10,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for extension in ["png", "pdf"]:
        path = output_dir / ("five_seed_performance." + extension)
        fig.savefig(path, dpi=220)
        print("Saved:", path)

    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]

    # Reuse the summary script's checks for all five result files.
    data = load_results(root / "results" / "synthetic")
    draw_plot(data, root / "figures" / "synthetic")


if __name__ == "__main__":
    main()