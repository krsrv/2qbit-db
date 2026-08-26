"""Joint least-squares fit of the two-qubit dephasing fidelities and a polar-chart
visualization of the fitted oscillation modes.

Fits eta, eps, kap, d1, d2, r1, r2 by an equally-weighted, simultaneous least-squares
fit against all four fidelity time series (++, +-, -+, --) at once, for both
decay_noiseless_data.csv and decay_noisy_data.csv. Each fidelity is a sum of terms of
the form (mode) ** n; this module plots the 13 underlying complex "modes" (12
oscillatory conjugate pairs + 1 constant/DC mode) on a single polar chart, one series
per dataset.

eta ZZ, eps ZI, kap IZ
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent

STATES = ["++", "+-", "-+", "--"]
CSV_COLUMN_FOR_STATE = {"++": "pp", "+-": "pm", "-+": "mp", "--": "mm"}
PARAM_NAMES = ["eta", "eps", "kap", "d1", "d2", "r1", "r2"]

# Every fidelity is (1/16) * sum_k c_k * mode_k ** n over the same 13 modes; only the
# coefficients c_k depend on the state. The modes are 6 conjugate pairs
# (amplitude * exp(-i*phi), amplitude * exp(+i*phi)) followed by the constant/DC mode,
# in the same order as MODE_GROUPS in least_squares_fitting.py.
MODE_KEYS = [
    "eps_eta",  # d1^4,       phi = 8 (eps + eta)
    "eta_kap",  # d2^4,       phi = 8 (eta + kap)
    "eps_m_kap",  # d1^4 d2^4,  phi = 8 (eps - kap)
    "eps_p_kap",  # d1^4 d2^4,  phi = 8 (eps + kap)
    "eta_m_kap",  # r1^4 d2^4,  phi = 8 (eta - kap)
    "eps_m_eta",  # d1^4 r2^4,  phi = 8 (eps - eta)
]


def _state_coefficients(state: str) -> np.ndarray:
    """Row of 13 coefficients for `state`, one per mode (pairs share a sign)."""
    s1 = 1.0 if state[0] == "+" else -1.0
    s2 = 1.0 if state[1] == "+" else -1.0
    # per-group sign, in MODE_KEYS order
    group_signs = [s1, s2, s1 * s2, s1 * s2, s2, s1]
    return np.array([s for s in group_signs for _ in (0, 1)] + [4.0])


# (4, 13): one row per state, rows ordered as STATES
COEFF_MATRIX = np.array([_state_coefficients(state) for state in STATES])
STATE_COEFFS = dict(zip(STATES, COEFF_MATRIX))


def mode_values(eta, eps, kap, d1, d2, r1, r2) -> np.ndarray:
    """The 13 complex per-step modes, ordered to match the columns of COEFF_MATRIX."""
    amplitudes = np.array(
        [
            d1**4,
            d2**4,
            (d1**4) * (d2**4),
            (d1**4) * (d2**4),
            (r1**4) * (d2**4),
            (d1**4) * (r2**4),
        ]
    )
    phases = 8.0 * np.array(
        [eps + eta, eta + kap, eps - kap, eps + kap, eta - kap, eps - eta]
    )
    # (6, 2) -> conjugate pair per group, then flattened and given the DC mode
    pairs = amplitudes[:, None] * np.exp(1j * np.array([-1.0, 1.0]) * phases[:, None])
    return np.concatenate([pairs.reshape(-1), [1.0 + 0j]])


def get_fidelities(n, eta, eps, kap, d1, d2, r1, r2) -> np.ndarray:
    """All four fidelities at once, shape (4, len(n)) with rows ordered as `STATES`.
    n may be a scalar or an array."""
    modes = mode_values(eta, eps, kap, d1, d2, r1, r2)
    powers = modes ** np.asarray(n)[..., None]
    return (powers @ COEFF_MATRIX.T / 16).T


# ---------------------------------------------------------------------------
# Simultaneous, equally-weighted least-squares fit
# ---------------------------------------------------------------------------

LOWER_BOUNDS = np.array([-np.pi, -np.pi, -np.pi, 0.0, 0.0, 0.0, 0.0])
UPPER_BOUNDS = np.array([np.pi, np.pi, np.pi, 1.0, 1.0, 1.0, 1.0])


def residuals(
    x: np.ndarray, n: np.ndarray, data: np.ndarray, shots: int, weighted: bool = True
) -> np.ndarray:
    """
    data: shape (4, len(n)) with rows ordered as `STATES`.
    """
    eta, eps, kap, d1, d2, r1, r2 = x
    diffs = []
    model = get_fidelities(n, eta, eps, kap, d1, d2, r1, r2)
    sigma = np.ones_like(data)
    if weighted:
        # Check whether sigma is less than some threshold value.
        sigma = np.real(np.sqrt(model * (1 - model) / (shots + 1e-5))) + 1e-5
    diffs = (model.real - data) / sigma
    # equally weighted: every (state, n) residual counts once
    return diffs.reshape(-1)


def _run_least_squares(x0: np.ndarray, n: np.ndarray, data: np.ndarray, shots: int):
    return least_squares(
        residuals,
        x0,
        args=(n, data, shots),
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
    )


def fit_joint(
    n: np.ndarray,
    data: np.ndarray,
    shots: int,
    n_restarts: int = 40,
    rng: np.random.Generator = None,
    x0: np.ndarray | None = None,
) -> dict[str, float]:
    """Simultaneous, equally-weighted least-squares fit of all four fidelity curves.

    Two modes:
    - x0 is None: global search via random multi-start (n_restarts draws), keeping
      the lowest-cost result. Use this for an unbiased fit.
    - x0 given: a single warm-started local refinement from that point. Use this
      to deterministically "seed" the fits.
    """
    cost_scale = (np.prod(data.shape) - 7) / 2  # len(x0) = 7

    rng = np.random.default_rng() if rng is None else rng
    best_result = None
    idx = 0
    curr_data = data.copy()
    curr_n = n.copy()
    k = 1
    while True:
        if x0 is not None:
            x0_trial = np.asarray(x0, dtype=float) + np.concatenate(
                [
                    rng.uniform(0, 0.01, size=3),
                    rng.uniform(0, 0.001, size=4),
                ]
            )
            x0_trial = np.clip(x0_trial, LOWER_BOUNDS, UPPER_BOUNDS)
        else:
            x0_trial = np.concatenate(
                [
                    rng.uniform(0, 0.02, size=3),
                    rng.uniform(0.97, 1.0, size=4),
                ]
            )
        result = _run_least_squares(x0_trial, curr_n, curr_data, shots)
        if best_result is None or result.cost < best_result.cost:
            best_result = result
        if idx > n_restarts and best_result.cost < 1.2 * cost_scale:
            break
        elif idx > (2**k) * n_restarts:
            drop_idx = rng.integers(0, n.shape[0], k)
            curr_n = np.delete(n, drop_idx)
            curr_data = np.delete(data, drop_idx, axis=1)
        if idx >= (2 ** (k + 1)) * n_restarts:
            k += 1
        idx += 1
        if idx % 10 == 0:
            print(
                f"Running iter {idx}. Current cost scale: {best_result.cost/cost_scale:4e} * {cost_scale}"
            )

    params = dict(zip(PARAM_NAMES, best_result.x))
    params["cost"] = best_result.cost
    params["rmse"] = float(np.sqrt(2 * best_result.cost / best_result.fun.size))
    params["result"] = best_result
    return params
