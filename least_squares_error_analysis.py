from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt

from least_squares import fit_joint, get_fidelities

HERE = Path(__file__).resolve().parent
OUTPUT_CSV = HERE / "output" / "error_bars.csv"


# ec9fae9a-3fea-46e7-bd7f-227ba238c5c8
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
    n_range = np.arange(20)
    true_data = get_fidelities(n_range, *true_params).real
    sigma = np.sqrt(true_data * (1 - true_data) / 4000)
    noisy_data = construct_noisy_data(true_data, sigma)
    result = fit_joint(n_range, noisy_data, n_restarts=5)
    detailed_result: scipy.OptimizeResult = result["result"]
    J = detailed_result.jac
    I = J.T @ J
    svals = np.linalg.svdvals(J)
    plt.semilogy(svals, label=f"Singular values", marker="o")
    plt.semilogy(np.diag(I) ** (-1), label=f"Std dev", marker="o")
    # plt.ylim(0.1, np.max(svals))
    plt.grid(True)
    plt.legend()
    plt.savefig(
        HERE / "output/least_squares_jacobian_svd_unweighted.pdf", bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    main()
