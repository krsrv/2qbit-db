"""Matrix pencil estimation for sums of complex exponentials (decaying modes).

Model: each time series (row) i is assumed to be
    y_i[k] = sum_{n=1}^{N} w_{i,n} * z_n**k,    k = 0 .. M-1
where the poles z_n (eigenvalues of the underlying time-shift matrix) are
SHARED across all rows, while the weights w_{i,n} may differ per row.

Joint estimation: for each row build the pair of Hankel matrices
    G_i^0[k, j] = y_i[k + j]        (shape (M-L) x L)
    G_i^1[k, j] = y_i[k + j + 1]    (shape (M-L) x L)
Both factor as G_i^0 = P D_i Q and G_i^1 = P Z D_i Q with
    P[k, n] = z_n**k   (temporal/row factor, shared by all rows)
    D_i     = diag(w_{i,n})
    Z       = diag(z_n)
    Q[n, j] = z_n**j.
Concatenating along axis=1 keeps the shared factor P on the left:
    Ghat^0 = P [D_1 Q | ... | D_R Q] = P W,      Ghat^1 = P Z W
so the time-shift matrix A with Ghat^1 = A Ghat^0 satisfies A = P Z P^+ and
its (nonzero) eigenvalues are exactly the shared poles z_n.

Guarding against spurious large-phase modes (two independent tools):
  - select_model_order : pick N from the singular value spectrum so the SVD
    filter never keeps noise directions in the first place;
  - filter_modes : post-select an existing estimate by a phase window (and
    optional weight floor / magnitude cap), then refit the weights.
"""

import numpy as np


def _hankel_pair(data, L):
    """Concatenated Hankel pair (Ghat^0, Ghat^1) for rows of `data`."""
    M = data.shape[1]
    idx = np.arange(M - L)[:, None] + np.arange(L)[None, :]
    G0 = np.concatenate([y[idx] for y in data], axis=1)
    G1 = np.concatenate([y[idx + 1] for y in data], axis=1)
    return G0, G1


def _fit_weights(data, eigenvalues):
    """Least-squares weights of each row on the basis z_n**k, plus residuals."""
    M = data.shape[1]
    V = eigenvalues[None, :] ** np.arange(M)[:, None]
    weights, *_ = np.linalg.lstsq(V, data.T, rcond=None)
    weights = weights.T  # (R, N)
    residuals = np.sqrt(np.mean(np.abs(data - (V @ weights.T).T.real) ** 2, axis=1))
    return weights, residuals


def matrix_pencil(data, N, L=None, error_floor=None):
    """Estimate N shared exponential modes from one or more time series.

    Parameters
    ----------
    data : array_like, shape (M,) or (R, M)
        Time series data; row i holds the M samples of experiment i.
    N : int
        Number of modes (eigenvalues) to estimate; rank of the SVD filter.
    L : int, optional
        Pencil parameter (number of Hankel columns). Defaults to M // 2,
        the usual choice. Must satisfy N <= L <= M - N.

    Returns
    -------
    eigenvalues : ndarray, shape (N,), complex
        Eigenvalues z_n of the estimated time-shift matrix, sorted by
        descending mean |weight| across rows.
    weights : ndarray, shape (R, N), complex
        Least-squares amplitudes w_{i,n} of each mode in each row.
    residuals : ndarray, shape (R,), float
        Root-mean-square reconstruction error per row,
        rms_k( y_i[k] - sum_n w_{i,n} z_n**k ).
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    R, M = data.shape
    if L is None:
        L = M // 2
    if not (N <= L <= M - N):
        raise ValueError(f"need N <= L <= M - N, got N={N}, L={L}, M={M}")

    G0, G1 = _hankel_pair(data, L)

    # SVD filter: rank-N truncation of Ghat^0 and of Ghat^1.
    U, s, Vh = np.linalg.svd(G0, full_matrices=False)
    if error_floor is not None and N == 0:
        idx = np.where(s < 0.01)[0]
        N = idx[0]
    U_N, s_N, V_N = U[:, :N], s[:N], Vh[:N].conj().T
    U1, s1, V1h = np.linalg.svd(G1, full_matrices=False)
    G1_f = (U1[:, :N] * s1[:N]) @ V1h[:N]

    # Time-shift matrix projected onto the rank-N signal subspace:
    # A = U_N^H Ghat^1_f V_N S_N^{-1}, the restriction of
    # Ghat^1_f (Ghat^0)^+ to the signal subspace.
    A = U_N.conj().T @ G1_f @ V_N / s_N[None, :]
    _, M = np.linalg.eig(A)
    print("Condition number", np.linalg.cond(M))
    eigenvalues = np.linalg.eigvals(A)

    weights, residuals = _fit_weights(data, eigenvalues)
    order = np.argsort(-np.abs(weights).mean(axis=0))
    return eigenvalues[order], weights[:, order], residuals


def select_model_order(data, L=None, gap_factor=1.5, floor_factor=1.5, rtol=1e-12):
    """Pick the model order N from the singular value spectrum of Ghat^0.

    N is the largest k such that s_k / s_{k+1} >= gap_factor, counting only
    singular values that stand above the noise bulk — s_k above
    floor_factor * median(s) — and above rtol * s_1 (machine-precision
    structure). Rationale: the signal edge is a drop at the TOP of the noise
    bulk, but the bulk itself also contains ratio gaps near its bottom edge
    (especially for a single short series), so a gap only counts if it starts
    from a singular value clearly above the bulk scale; the median of the
    spectrum estimates that scale. Keeping only those directions means the
    SVD filter never admits the noise directions whose eigenvalues land at
    arbitrary (large) phases.

    Assumes fewer than about half the singular values are signal, so the
    median sits in the noise bulk / numerical floor.

    Returns
    -------
    N : int
        Selected model order.
    singular_values : ndarray
        Spectrum of Ghat^0, for inspection.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    M = data.shape[1]
    if L is None:
        L = M // 2
    G0, _ = _hankel_pair(data, L)
    s = np.linalg.svd(G0, compute_uv=False)

    N_max = min(L, M - L, len(s) - 1)
    ratios = s[:-1] / s[1:]
    floor = floor_factor * np.median(s)
    candidates = [
        k + 1
        for k in range(N_max)
        if ratios[k] >= gap_factor and s[k] >= floor and s[k] > rtol * s[0]
    ]
    N = max(candidates) if candidates else int(np.argmax(ratios[:N_max])) + 1
    return N, s


def filter_modes(
    data, eigenvalues, weights, max_phase, min_weight=0.0, max_magnitude=None
):
    """Post-select modes of an existing estimate, then refit the weights.

    Keeps modes with |arg(z)| <= max_phase, mean |weight| >= min_weight and
    (optionally) |z| <= max_magnitude, and re-solves the least squares on the
    surviving Vandermonde basis. Because the basis columns are strongly
    correlated (all poles near z = 1), the surviving weights must be refit —
    simply deleting entries would leave the discarded modes' share of the
    signal misattributed. Compare the returned residuals with the unfiltered
    ones: unchanged residuals mean the dropped modes carried no signal.

    Both tests are symmetric under conjugation, so for real data the ±phase
    partners of a pair are always kept or dropped together.

    Returns
    -------
    eigenvalues : ndarray, complex
        Surviving modes (input order preserved).
    weights : ndarray, shape (R, N_kept), complex
        Refit amplitudes.
    residuals : ndarray, shape (R,), float
        Per-row RMS reconstruction error of the reduced model.
    kept : ndarray of bool
        Mask of surviving modes, aligned with the input eigenvalues.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    eigenvalues = np.asarray(eigenvalues)
    kept = np.abs(np.angle(eigenvalues)) <= max_phase
    kept &= np.abs(weights).mean(axis=0) >= min_weight
    if max_magnitude is not None:
        kept &= np.abs(eigenvalues) <= max_magnitude
    if not kept.any():
        raise ValueError("all modes were filtered out; loosen the thresholds")
    weights, residuals = _fit_weights(data, eigenvalues[kept])
    return eigenvalues[kept], weights, residuals, kept
