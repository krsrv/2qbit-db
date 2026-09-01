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
PARAM_NAMES = ["eta", "eps", "kap", "d1", "d2", "r1", "r2", "p1", "p2", "p3", "p4"]

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


# ---------------------------------------------------------------------------
# Vectorized Lindblad generator of one repetition
# ---------------------------------------------------------------------------

I2 = np.eye(2)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]])
H = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]])
SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]])  # lowering operator

IDEAL_MSMT_OPS = np.zeros((4, 4, 4))
for i in range(4):
    IDEAL_MSMT_OPS[i, i, i] = 1
    IDEAL_MSMT_OPS[i] = np.kron(H, H) @ IDEAL_MSMT_OPS[i] @ np.kron(H, H)
IDEAL_MSMT_OPS = IDEAL_MSMT_OPS.reshape(4, -1)


def _on(m: np.ndarray, qubit: int) -> np.ndarray:
    """Single-qubit operator `m` acting on `qubit` (0 = left factor) of the pair."""
    return np.kron(m, I2) if qubit == 0 else np.kron(I2, m)


def _hamiltonian_super(h: np.ndarray) -> np.ndarray:
    """Superoperator of -i [h, .], using vec(A rho B) = kron(A, B.T)."""
    eye = np.eye(4)
    return -1j * (np.kron(h, eye) - np.kron(eye, h.T))


def _dissipator_super(c: np.ndarray) -> np.ndarray:
    """Superoperator of c rho c^dag - {c^dag c, rho} / 2, for jump operator `c`."""
    eye = np.eye(4)
    num = c.conj().T @ c  # c^dag c
    return np.kron(c, c.conj()) - 0.5 * (np.kron(num, eye) + np.kron(eye, num.T))


# The generator is *linear* in (eta, eps, kap, d1**2, d2**2, r1**2, r2**2): each Hamiltonian
# term carries one phase parameter, and each jump operator is a fixed matrix scaled by
# its rate, so its dissipator is quadratic in that rate. Build the seven 16x16 basis
# superoperators once at import; construct_unit_operator is then a single (256, 7) @ (7,)
# product rather than the ~20 np.kron calls it used to cost on every residual evaluation.
_GENERATOR_BASIS = np.array(
    [
        _hamiltonian_super(np.kron(SIGMA_Z, SIGMA_Z)),  # eta
        _hamiltonian_super(_on(SIGMA_Z, 0)),  # eps
        _hamiltonian_super(_on(SIGMA_Z, 1)),  # kap
        _dissipator_super(0.5 * _on(SIGMA_Z, 0)),  # scaled by d1
        _dissipator_super(0.5 * _on(SIGMA_Z, 1)),  # scaled by d2
        _dissipator_super(2.0 * _on(SIGMA_MINUS, 0)),  # scaled by r1
        _dissipator_super(2.0 * _on(SIGMA_MINUS, 1)),  # scaled by r2
    ],
    dtype=complex,
).reshape(7, -1)


def construct_unit_operator(eta, eps, kap, d1, d2, r1, r2) -> np.ndarray:
    """The (16, 16) Lindblad generator of one time step as a superoperator.

    - vec(A rho B) = kron(A, B.T)
    - d rho / dt = -i [H, rho] + sum_k (c_k rho c_k^dag - {c_k^dag c_k, rho} / 2)
    - H = eta ZZ + eps ZI + kap IZ
    - c_k = 1/2 d1 Z_1, 1/2 d2 Z_2, 2 r1 sigma^-_1, 2 r2 sigma^-_2
    """
    coefficients = np.array([eta, eps, kap, d1, d2, r1, r2], dtype=complex)
    return (_GENERATOR_BASIS.T @ coefficients).reshape(16, 16)


def construct_init_state(init1, init2):
    confusion = np.outer([init1, 1 - init1], [init2, 1 - init2]).reshape(-1)
    return confusion @ IDEAL_MSMT_OPS


def construct_msmt_op(out1, out2):
    a = np.array([[out1, 1 - out1], [1 - out1, out1]])
    b = np.array([[out2, 1 - out2], [1 - out2, out2]])
    confusion = (a[:, None, :, None] * b[None, :, None, :]).reshape(4, 4)  # kron(a, b)
    return confusion @ IDEAL_MSMT_OPS


def get_fidelities(
    n, eta, eps, kap, d1, d2, r1, r2, init1, init2, out1, out2
) -> np.ndarray:
    """
    All four fidelities at once, shape (len(n), 4) with columns ordered as `STATES`.
    n may be a scalar or an array.

    Args:
        n: Integer or array of time steps.
        eta, eps, kap: Phase parameters for (ZZ, ZI, IZ).
        d1, d2: Dephasing rates (sigma_z/2).
        r1, r2: Relaxation rates (2 sigma_-).
        init1, init2: Flip probabilities in initial state prep for each qubit
        out1, out2: POVM mixing rates for each qubit

    Returns:
        Array of shape (len(n), 4): fidelities for each computational basis state at each n.
    """
    if isinstance(n, (int, np.integer)):
        n = np.arange(n)
    n = np.asarray(n, dtype=float)

    unit_op = construct_unit_operator(eta, eps, kap, d1, d2, r1, r2)
    state = construct_init_state(init1, init2).astype(complex)
    msmt_ops = construct_msmt_op(out1, out2)

    # exp(G t) s0 = V diag(exp(w t)) V^-1 s0, so one eigendecomposition of the (16, 16)
    # generator gives *every* time step: fold V^-1 s0 and msmt_ops @ V into a single
    # (16, 4) weight matrix, and the whole trajectory is one (len(n), 16) @ (16, 4)
    # product. This replaces a Python loop of len(n) expm_multiply calls (~3 orders of
    # magnitude slower), which matters because `residuals` runs this on every function
    # and finite-difference Jacobian evaluation of every least_squares restart.
    #
    # The generator is a Kronecker sum of single-qubit Lindbladians and stays
    # diagonalizable across the whole parameter box (cond(V) <= 6 everywhere in
    # LOWER_BOUNDS..UPPER_BOUNDS, including the degenerate corners), so this is exact
    # to machine precision rather than an approximation.
    eigenvalues, eigenvectors = np.linalg.eig(unit_op)
    weights = (msmt_ops @ eigenvectors) * np.linalg.solve(eigenvectors, state)
    return np.real(np.exp(np.multiply.outer(4 * n, eigenvalues)) @ weights.T)


# ---------------------------------------------------------------------------
# Simultaneous, inverse-covariance-weighted least-squares fit
# ---------------------------------------------------------------------------

LOWER_BOUNDS = np.array(
    # n, eta, eps, kap, d1, d2, r1, r2, init1, init2, out1, out2
    [-np.pi, -np.pi, -np.pi, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9]
)
UPPER_BOUNDS = np.array(
    # n, eta, eps, kap, d1, d2, r1, r2, init1, init2, out1, out2
    [np.pi, np.pi, np.pi, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)


# Scale of each parameter's natural variation, used as `x_scale` so the trust region
# treats a 1e-2 rad phase step and a 1e-3 decay step as comparably sized.
X_SCALE = np.array([1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3, 1e-3])


# Probabilities below this are treated as this value when they set the weights, so a
# model prediction that runs to zero cannot produce an infinite weight.
MIN_WEIGHT_PROB = 1e-12


def residuals(
    x: np.ndarray,
    n: np.ndarray,
    data: np.ndarray,
    shots: int,
    weight_probs: np.ndarray | None = None,
) -> np.ndarray:
    """Whitened residual vector for `least_squares`.

    Each row of `data` is one estimate of the four outcome probabilities from `shots`
    shots, so its covariance is cov = (diag(p) - p p.T) / shots, which has rank 3 rather
    than 4. Inverse of the 3x3 block gives cov^-1 = shots * (diag(1/q) + 1 1.T / q4)
    with q4 = 1 - sum(q) the dropped outcome. Since the four residuals sum to zero,
    the quadratic form r.T cov^-1 r becomes shots * sum_i r_i^2 / p_i over all four
    outcomes. So the whitened residual is just sqrt(shots) * r / sqrt(p).

    The vector has 4 entries per time step but still only 3 independent
    ones, so the degrees of freedom are 3 * len(n) - len(PARAM_NAMES).

    data: shape (len(n), 4) with columns ordered as `STATES`.
    shots: shots per time step, setting the scale of the covariance.
    weight_probs: probabilities defining the covariance, shape (len(n), 4). None means
        "use the model prediction at `x`", so the weights track the current estimate.
        Pass an array to hold them fixed, as the iterated GLS passes in `fit_joint` do.
    """
    model_data = get_fidelities(n, *x).real
    probs = model_data if weight_probs is None else weight_probs
    diffs = model_data - data
    return (
        np.sqrt(shots) * diffs / np.sqrt(np.clip(probs, MIN_WEIGHT_PROB, None))
    ).reshape(-1)


def _run_least_squares(
    x0: np.ndarray,
    n: np.ndarray,
    data: np.ndarray,
    shots: int,
    weight_probs: np.ndarray | None = None,
):
    return least_squares(
        residuals,
        x0,
        args=(n, data, shots, weight_probs),
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        # x_scale=X_SCALE,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )


def fit_joint(
    n: np.ndarray,
    data: np.ndarray,
    shots: int,
    n_restarts: int = 40,
    rng: np.random.Generator = None,
    x0: np.ndarray | None = None,
    n_gls_passes: int = 2,
    gls_tol: float = 1e-9,
) -> dict[str, float]:
    """Simultaneous, inverse-covariance-weighted least-squares fit of all four curves.

    Iterated GLS:
    - 1st pass: Uses noisy estimates of the covariance. The resulting optimization
     adds a term  whose expectation is -tr(Sigma^-1 dSigma/dx) / 2, which has a
     O(1/shots) bias which does not decrease with number of time steps.
    - 2nd pass: Freeze the covariance weights and rerun GLS. This step removes the
    bias -tr(Sigma^-1 dSigma/dx) / 2.

    Args:
    - x0 : if None, search via random multi-start (n_restarts draws). If not None
        search via perturbations around x0.
    - n_gls_passes: maximum reweighting passes after the noisy search. 0 disables it.
    - gls_tol: stop reweighting once no parameter moves by more than this.
    """
    cost_scale = (np.prod(data[:, :-1].shape) - len(PARAM_NAMES)) / 2  # len(x0) = 7

    rng = np.random.default_rng() if rng is None else rng
    best_result = None
    idx = 0
    curr_data = data.copy()
    curr_n = n.copy()
    num_data = n.shape[0]
    # Track which rows produced `best_result`, since deletions below can change them.
    best_n, best_data = curr_n, curr_data
    k = 1
    while True:
        if x0 is not None:
            x0_trial = np.asarray(x0, dtype=float) + np.concatenate(
                [
                    rng.uniform(0, 0.01, size=3),
                    rng.uniform(0, 0.001, size=4),
                    rng.uniform(0.0, 0.001, size=4),
                ]
            )
            x0_trial = np.clip(x0_trial, LOWER_BOUNDS, UPPER_BOUNDS)
        else:
            x0_trial = np.concatenate(
                [
                    rng.uniform(0.0, 0.02, size=3),
                    rng.uniform(0.0, 0.003, size=4),
                    rng.uniform(0.99, 1.0, size=4),
                ]
            )
        result = _run_least_squares(x0_trial, curr_n, curr_data, shots)
        if best_result is None or result.cost < best_result.cost:
            best_result = result
            best_n, best_data = curr_n, curr_data
        if idx > n_restarts and best_result.cost < 1.2 * cost_scale:
            break
        elif idx > (4**k) * n_restarts:
            drop_idx = rng.integers(0, n.shape[0], k)
            curr_n = np.delete(n, drop_idx)
            curr_data = np.delete(data, drop_idx, axis=0)
            weight_probs = np.clip(curr_data, 1e-6, None)
        if idx >= (4 ** (k + 1)) * n_restarts:
            k += 1
        idx += 1
        if idx % 10 == 0:
            print(
                f"Running iter {idx}. Current cost scale: {best_result.cost/cost_scale:4e} * {cost_scale}"
            )
        if k / num_data > 0.1:
            break

    # Iterated GLS refinement on the rows the search actually converged on.
    # Costs are not comparable across passes -- each uses different weights -- so every pass is
    # accepted rather than being kept only when it lowers the cost.
    weight_probs = get_fidelities(best_n, *best_result.x).real
    for _ in range(n_gls_passes):
        refined = _run_least_squares(
            best_result.x, best_n, best_data, shots, weight_probs
        )
        shift = np.max(np.abs(refined.x - best_result.x))
        best_result = refined
        if shift < gls_tol:
            break

    params = dict(zip(PARAM_NAMES, best_result.x))
    params["true_cost"] = 0.5 * np.sum(residuals(best_result.x, n, data, shots) ** 2)
    # The residual vector carries 4 entries per time step but only 3 independent ones,
    # so divide by the degrees of freedom rather than by `best_result.fun.size`.
    dof = 3 * len(n) - len(PARAM_NAMES)
    params["rmse"] = float(np.sqrt(2 * params["true_cost"] / dof))
    params["result"] = best_result
    return params
