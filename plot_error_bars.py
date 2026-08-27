import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from get_error_bars import PHASE_NAMES
from least_squares import PARAM_NAMES

HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "output" / "error_bars.csv"
STD_PDF = HERE / "output" / "error_bars_std_heatmaps.pdf"
BIAS_PDF = HERE / "output" / "error_bars_bias_heatmaps.pdf"

DECAY_NAMES = [name for name in PARAM_NAMES if name not in PHASE_NAMES]
# The phases and the decay coefficients carry different units and magnitudes, so each
# group gets its own shared color scale (one row and one colorbar per group).
PARAM_GROUPS = [("phases", PHASE_NAMES), ("decay coefficients", DECAY_NAMES)]

# Widest span a std colorbar may cover. The decay group bottoms out at ~2e-9 in a handful
# of degenerate r1 cells; letting those set vmin would flatten every other cell, so clip
# and mark the colorbar with an "extend" arrow instead.
STD_DECADES = 3
# Half-width of the linear region of the symlog bias scale. Median |bias| is ~3e-5 (phases)
# and ~5e-5 (decay), so a purely linear scale would render most cells as blank white.
BIAS_LINTHRESH = 1e-5


def load_grids(path=INPUT_CSV):
    """Std dev, mean, and bias of each fitted param over `count`, on the (shots, repetitions) grid."""
    df = pd.read_csv(path)
    grouped = df.groupby(["repetitions", "shots"])[PARAM_NAMES]
    # ddof=1: these are sample std devs over the finite set of counts per cell
    stds = {
        name: grouped.std(ddof=1)[name].unstack("repetitions") for name in PARAM_NAMES
    }
    means = {name: grouped.mean()[name].unstack("repetitions") for name in PARAM_NAMES}
    true_row = {name: df[f"true_{name}"].iloc[0] for name in PARAM_NAMES}
    biases = {
        name: np.abs(means[name] - true_row[name]) / stds[name] for name in PARAM_NAMES
    }
    return stds, means, biases, true_row, df


def stack(grids, names):
    return np.concatenate([grids[name].to_numpy().ravel() for name in names])


def std_norm(grids, names):
    """Log scale over one group, floored at STD_DECADES below its max."""
    values = stack(grids, names)
    vmax = np.min([np.nanmax(values), 100])
    positive = values[values > 0]
    floor = vmax / 10**STD_DECADES
    vmin = max(np.nanmin(positive), floor)
    print(vmin, vmax)
    clipped = (positive < vmin).any() or (values <= 0).any()
    return LogNorm(vmin=vmin, vmax=vmax), "min" if clipped else "neither"


def bias_norm(grids, names):
    """Logarithmic normalization for the bias grid of a parameter group."""
    values = np.abs(stack(grids, names))
    vmin = max(np.nanmin(values[values > 0]), BIAS_LINTHRESH)
    vmax = np.nanmax(values)
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    return norm, "neither"


def plot_grids(grids, norm_fn, cmap, title, label, path):
    """One heatmap per param, x=repetitions, y=shots, one shared scale per param group."""
    ncols = max(len(names) for _, names in PARAM_GROUPS)
    fig, axes = plt.subplots(
        len(PARAM_GROUPS), ncols, figsize=(5 * ncols, 9), constrained_layout=True
    )
    cmap = plt.get_cmap(cmap).with_extremes(bad="lightgray")
    for row, (group, names) in zip(axes, PARAM_GROUPS):
        norm, extend = norm_fn(grids, names)
        for ax, name in zip(row, names):
            grid = grids[name]
            values = np.ma.masked_invalid(grid.to_numpy())
            if isinstance(norm, LogNorm):
                # exact zeros (all counts identical) are not representable on a log scale
                values = np.ma.masked_less_equal(values, 0)
            # origin="lower" keeps repetitions and shots ascending away from the corner
            im = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm)
            ax.set_xticks(range(len(grid.columns)), grid.columns)
            ax.set_yticks(range(len(grid.index)), grid.index)
            ax.set_xlabel("repetitions")
            ax.set_ylabel("shots")
            ax.set_title(name)
        # groups need not fill the row (three phases against four decay coefficients)
        for ax in row[len(names) :]:
            ax.axis("off")
        fig.colorbar(
            im, ax=row, label=f"{label} ({group})", extend=extend, fraction=0.04
        )
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return fig


def plot_std_heatmaps(means, stds, path=STD_PDF):
    return plot_grids(
        stds,
        norm_fn=std_norm,
        cmap="viridis",
        title="Std dev of fitted parameters across counts (eta ZZ, eps ZI, kap IZ)",
        label="std",
        path=path,
    )


def plot_bias_heatmaps(biases, path=BIAS_PDF):
    return plot_grids(
        biases,
        norm_fn=bias_norm,
        cmap="RdBu_r",
        title="Bias of fitted parameters (mean across counts - true) (eta ZZ, eps ZI, kap IZ)",
        label="(mean - true) / std",
        path=path,
    )


def print_means(means, true_row, df):
    """Per-cell means, then the overall mean of each param next to its true value."""
    for name in PARAM_NAMES:
        print(f"\n=== mean({name}) — rows: shots, columns: repetitions ===")
        print(means[name].to_string(float_format=lambda v: f"{v:.6g}"))

    print("\n=== overall mean across all (repetitions, shots, count) ===")
    summary = pd.DataFrame(
        {
            "mean": [df[name].mean() for name in PARAM_NAMES],
            "std": [df[name].std(ddof=1) for name in PARAM_NAMES],
            "true": [true_row[name] for name in PARAM_NAMES],
        },
        index=PARAM_NAMES,
    )
    summary["bias"] = summary["mean"] - summary["true"]
    print(summary.to_string(float_format=lambda v: f"{v:.6g}"))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Plot std dev and bias heatmaps of fitted parameters from an error bar CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV,
        help="Input CSV of fit results (default: ./output/error_bars.csv)",
    )
    parser.add_argument(
        "--std-output",
        type=Path,
        default=STD_PDF,
        help="Std dev heatmap path (default: ./output/error_bars_std_heatmaps.pdf)",
    )
    parser.add_argument(
        "--bias-output",
        type=Path,
        default=BIAS_PDF,
        help="Bias heatmap path (default: ./output/error_bars_bias_heatmaps.pdf)",
    )
    args = parser.parse_args()

    stds, means, biases, true_row, df = load_grids(args.input)
    print_means(means, true_row, df)
    plot_std_heatmaps(means, stds, path=args.std_output)
    plot_bias_heatmaps(biases, path=args.bias_output)
    print(f"\nWrote {args.std_output}\nWrote {args.bias_output}")
    plt.show()


if __name__ == "__main__":
    main()
