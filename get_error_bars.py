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
    `PHASE_NAMES`: list of phase parameter names, used to group parameters by type.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from least_squares import PARAM_NAMES, fit_joint, get_fidelities

HERE = Path(__file__).resolve().parent
OUTPUT_CSV = HERE / "output" / "error_bars_weighted.csv"


def construct_noisy_data(data, sigma=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Additive white noise, one draw per (state, n) entry
    noise = rng.normal(0, sigma, np.shape(data))
    noisy_data = data + noise
    return noisy_data


PHASE_NAMES = ["eta", "eps", "kap"]


def canonicalize_signs(params):
    """Pick the eps >= 0 branch of the (eta, eps, kap) -> -(eta, eps, kap) degeneracy."""
    if params["eps"] < 0:
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

    true_params = (
        0.4 * np.pi / 180,
        np.pi / 180,
        0.2 * np.pi / 180,
        0.9998,
        0.997,
        0.998,
        0.9996,
    )
    # store truth on the same branch as the fits. Otherwise, there can be a "synthetic"
    # bias in the fits.
    true_row = canonicalize_signs(dict(zip(PARAM_NAMES, true_params)))
    rows = []

    seed = 1
    rng = np.random.default_rng(seed=seed)
    max_reps = 50
    n_range = np.arange(max_reps)
    shot_range = range(1000, 11000, 1000)
    true_data = get_fidelities(n_range, *true_params).real

    # Run sampling such that a noisy sample is created for 1,...,max_reps for a given number of
    # shots and prefixes are used for each repetition run.
    for shots in shot_range:
        sigma = np.sqrt(true_data * (1 - true_data) / shots)
        for count in range(20):
            noisy_data_max_rep = construct_noisy_data(true_data, sigma, rng=rng)
            prev_fit_params = None
            for repetitions in range(10, max_reps, 5):
                data = noisy_data_max_rep[:, :repetitions]
                fit_params = fit_joint(
                    np.arange(repetitions),
                    data,
                    shots,
                    n_restarts=10,
                    rng=rng,
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
