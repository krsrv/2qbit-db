"""
error_bars.py: Tools for generating, processing, and analyzing error bars in parameter estimation
of noisy quantum system fits.

This module provides functions to:
- Generate and fit noisy data to obtain parameter estimates.
- Canonicalize fit parameter signs to a standard form.
- Write collections of fitted results to CSV files.
- Define parameter groupings used in error bar analysis.

Intended to be run in scripts that sweep over repetitions/shots, produce CSVs of fit results,
and enable downstream plotting of errors, biases, and variances.

Exports:
    `STATES`: the four outcome labels, in column order.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_fit import fit_family
from model import DB_SETS, PHASE_NAMES, probabilities

HERE = Path(__file__).resolve().parent
OUTPUT_CSV = HERE / "output" / "error_bars_weighted.csv"
STATES = ["++", "+-", "-+", "--"]
SYNTHETIC = DB_SETS["synthetic"]


def construct_noisy_data(data, sigma=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Additive correlated noise, one draw per (state, n) entry
    noisy_data = np.stack(
        [rng.multivariate_normal(data[i], sigma[i]) for i in range(sigma.shape[0])],
        axis=0,
    )
    return noisy_data


def canonicalize_signs(params):
    """Pick the eps >= 0 branch of the phases -> -phases degeneracy."""
    if params.get("eps", 0.0) < 0:
        return {
            name: -value if name in PHASE_NAMES else value
            for name, value in params.items()
        }
    return dict(params)


def write_rows(rows: list, path: Path):
    """Dump the accumulated fit records to `path` as a tidy CSV, one row per fit."""
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate error bars for fitted parameters and save to CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(HERE / "output" / "error_bars.csv"),
        help="Output file path (default: ./output/error_bars.csv)",
    )
    args = parser.parse_args()

    true_params = {
        "eta": 0.4 * np.pi / 180,
        "eps": np.pi / 180,
        "kap": 0.2 * np.pi / 180,
        "d1": 0.0001,
        "d2": 0.003,
        "r1": 0.002,
        "r2": 0.001,
        "ep1": 0.991,
        "em1": 0.992,
        "ep2": 0.997,
        "em2": 0.995,
        "z1": 0.0,
        "z2": 0.0,
        "z12": 0.0,
    }
    # store truth on the same branch as the fits. Otherwise, there can be a "synthetic"
    # bias in the fits.
    true_row = canonicalize_signs(true_params)
    rows = []

    seed = 1
    rng = np.random.default_rng(seed=seed)
    max_reps = 50
    n_range = np.arange(max_reps)
    shot_range = range(1000, 11000, 1000)
    true_data = probabilities(SYNTHETIC, true_params, n_range, model="mix")

    # Run sampling such that a noisy sample is created for 1,...,max_reps for a given number of
    # shots and prefixes are used for each repetition run.
    for shots in shot_range:
        # Generate covariance matrix:
        # C_ij = sum_ij delta_ij p(AX=e_i) - p(AX=e_i) . p(AX=e_j)
        # Where A is the confusion matrix, X is the true matrix. Note that
        # even when A is Identity, the output vector will have correlated
        # distance.
        sigma = (
            true_data[:, :, None] * np.eye(4)
            - true_data[:, :, None] * true_data[:, None, :]
        ) / shots
        for count in range(20):
            noisy_data_max_rep = construct_noisy_data(true_data, sigma, rng=rng)
            prev_fit_params = None
            for repetitions in range(10, max_reps, 5):
                data = noisy_data_max_rep[:repetitions]
                fit_params = fit_family(
                    SYNTHETIC,
                    np.arange(repetitions),
                    data,
                    shots,
                    rng,
                    n_restarts=10,
                    x0=(
                        prev_fit_params["result"].x
                        if prev_fit_params is not None
                        else None
                    ),
                )
                row = {"repetitions": repetitions, "shots": shots, "count": count}
                row.update(canonicalize_signs(fit_params))
                row.update({f"true_{name}": v for name, v in true_row.items()})
                rows.append(row)
                prev_fit_params = fit_params  # Warm-chaining solutions
                print(
                    f"Finished repetitions={repetitions}, shots={shots}, count={count}."
                )
            # checkpoint after each (repetitions, shots) block
            write_rows(rows, path=args.output)
    return write_rows(rows, path=args.output)


if __name__ == "__main__":
    main()
