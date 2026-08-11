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

from pathlib import Path

import numpy as np
import pandas as pd

from least_squares import PARAM_NAMES, get_fidelities, fit_joint

HERE = Path(__file__).resolve().parent
OUTPUT_CSV = HERE / "output" / "error_bars_weighted.csv"


def get_fit_data(data_gen_fn, fit_fn):
    """
    data_gen_fn: callable fn that produces a (k, n) noisy dataset corresponding to n datapoints
      for k states in each call, with the same underlying noiseless data
    fit_fn: callable fn that takes in data_gen_fn and n and returns a list of params
    """
    data = data_gen_fn()
    fit_params = fit_fn(data)
    return fit_params


def construct_noisy_data(data, sigma=None):
    # Additive white noise, one draw per (state, n) entry
    noise = np.random.normal(0, sigma, np.shape(data))
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


def write_rows(rows, path=OUTPUT_CSV):
    """Dump the accumulated fit records to `path` as a tidy CSV, one row per fit."""
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main():
    true_params = (
        0.4 * np.pi / 180,
        np.pi / 180,
        0.2 * np.pi / 180,
        0.999,
        0.999,
        0.999,
        0.999,
    )
    # store truth on the same branch as the fits. Otherwise, there can be a "synthetic"
    # bias in the fits.
    true_row = canonicalize_signs(dict(zip(PARAM_NAMES, true_params)))
    rows = []
    for repetitions in range(10, 50, 5):
        n_range = np.arange(repetitions)
        true_data = get_fidelities(n_range, *true_params).real
        for shots in range(1000, 11000, 1000):
            sigma = np.sqrt(true_data * (1 - true_data) / shots)
            data_gen_fn = lambda data=true_data, sigma=sigma: construct_noisy_data(
                data, sigma
            )
            fit_fn = lambda data, n=n_range: fit_joint(n, data, n_restarts=5)
            for count in range(20):
                fit_params = get_fit_data(data_gen_fn, fit_fn)
                row = {"repetitions": repetitions, "shots": shots, "count": count}
                row.update(canonicalize_signs(fit_params))
                row.update({f"true_{name}": v for name, v in true_row.items()})
                rows.append(row)
            # checkpoint after each (repetitions, shots) block
            write_rows(rows)
            print(f"Finished repetitions={repetitions}, shots={shots}.")
    return write_rows(rows)


if __name__ == "__main__":
    main()
