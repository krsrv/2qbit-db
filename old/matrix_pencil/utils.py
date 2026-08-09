import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

##################################
# Signal utilities
##################################


def print_params(params):
    titles = ["frequency (rad/s)", "decay (1/s)", "weights", "phases (rad)", "error"]
    title = "".join([f"{x:<20}" for x in titles])
    print(title)
    print("-" * len(title))
    for icomp in range(params.shape[0]):
        row = "".join([f"{x:<20.5E}" for x in params[icomp, :]])
        print(row)
    return


def signal(ts, params, sigma=0):
    sig = 0
    for param in params:
        p1, p2, p3, p4 = param[0], param[1], param[2], param[3]
        sig += p3 * np.exp(p2 * ts) * np.cos(p1 * ts + p4)
    if sigma:
        return sig + np.random.randn(len(ts)) * sigma
    else:
        return sig


def get_params_pencil(output, dt):
    """
    Extracts estimated signal parameters from the matrix pencil decomposition output.

    Parameters
    ----------
    output : tuple
        Output of the matrix pencil method, typically (Z, R, M, ...):
        - Z : ndarray
            Array of system poles (complex exponentials).
        - R : ndarray
            Residues associated with each mode.
        - M : int or other (not used here)
            Model rank or descriptor (ignored here).
        - _ : any
            Placeholder for unused values from output tuple.
    dt : float
        Time step (sampling period) of the signal.

    Returns
    -------
    params_est : ndarray, shape (n_modes, 5)
        Matrix of estimated parameters for each mode. The columns correspond to:
        - 0: frequency (rad/s)
        - 1: decay rate (1/s)
        - 2: magnitude of the residue (weight)
        - 3: phase of the residue (radians)
        - 4: placeholder for error term (filled as np.nan)

        The output is sorted by descending magnitude of the residues.
    """
    Z, R, M, _ = output
    inds = np.argsort(np.abs(R))[::-1]
    Z, R = Z[inds], R[inds]
    params_est = np.zeros((len(Z), 5), dtype=np.float64)
    params_est[:, 0] = Z / np.abs(Z)  # np.imag(np.log(Z)) / dt  # frequency (rad/s)
    params_est[:, 1] = np.real(np.log(Z)) / dt  # decay rate (1/s)
    params_est[:, 2] = np.abs(R)  # weight
    params_est[:, 3] = np.angle(R)  # phase (rad)
    params_est[:, 4] = np.nan  # error (not estimated here)
    return params_est


def plot_params(data, inds=None, xlim=None, ylim=None):
    """
    data: [true_params, est_params_1, est_params_2, ...]
    inds: which indices of true params are actually sense-able
    """
    if not inds:
        inds = np.arange(len(data[0]))

    markers = ["x", "+", "o"]
    colors = ["C0", "red", "blue", "green", "purple", "orange"]
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 6], width_ratios=[6, 1])
    ax = fig.add_subplot(gs[1, 0])
    for ii, params in enumerate(data):
        if ii == 0:
            ax.scatter(
                params[inds, 1], params[inds, 0], marker=markers[ii], color=colors[ii]
            )
        else:
            ax.scatter(params[:, 1], params[:, 0], marker=markers[ii], color=colors[ii])
    ax.grid()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(r"$\omega$")
    ax.set_xlabel(r"$\gamma$")

    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    for ii, params in enumerate(data):
        for jj, param in enumerate(params):
            if (ii == 0 and jj in inds) or (ii != 0):
                axt.axvline(x=param[1], color=colors[ii], alpha=0.25, linewidth=1)
    axt.get_yaxis().set_visible(False)  # Hide y-axis labels
    axt.get_xaxis().set_visible(False)  # Hide x-axis labels

    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    for ii, params in enumerate(data):
        for jj, param in enumerate(params):
            if (ii == 0 and jj in inds) or (ii != 0):
                axr.axhline(y=param[0], color=colors[ii], alpha=0.25, linewidth=1)
    axr.get_yaxis().set_visible(False)  # Hide y-axis labels
    axr.get_xaxis().set_visible(False)  # Hide y-axis labels

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    return fig


def filter_params(
    params, times, freqtol=None, decaytol=None, amptol=None, matchtol=None, errtol=None
):
    """
    Filter out spurious/unimportant modes from `params`.

    params: array shaped (M, 5) with columns [freq, decay, amp, phase, err]
      - freq  [rad/s]  (can be +/-)
      - decay [1/s]    (should be <= 0 for decaying modes)
      - amp
      - phase
      - err

    times: 1D array of sampling times (assumed uniform)
    freqtol:  dimensionless threshold for "low frequency" relative to window T.
    decaytol: dimensionless threshold for "slow decay" relative to window T.
    amptol:   minimum |amp|
    matchtol: frequency pairing tolerance (|f_i + f_j| < matchtol for a +/− pair)
    errtol:   maximum allowed error

    Logic:
      - Require decay <= 0 (decaying)
      - Require |freq| <= f_Nyquist
      - EXCLUDE modes that are simultaneously "low frequency" AND "slow decay"
        (i.e., would look nearly constant over the time window)
      - Require |amp| > amptol (if given)
      - If matchtol is given, keep only modes that have an opposite-sign frequency partner within `matchtol`
      - Require err <= errtol (if given)
      - Return remaining rows, sorted by descending |amp|
    """

    params = np.asarray(params, dtype=float)
    times = np.asarray(times, dtype=float)
    assert params.ndim == 2 and params.shape[1] == 5, "params must be (M,5)"

    freqs = params[:, 0]
    decays = params[:, 1]
    amps = params[:, 2]
    phases = params[:, 3]  # unused in filtering, but kept in output
    errs = params[:, 4]

    # Basic sampling quantities
    dt = times[1] - times[0]
    T = times.max() - times.min() if times.size > 1 else 0.0
    fNyq = 1.0 / (2.0 * dt)

    M = len(params)
    keep = np.ones(M, dtype=bool)

    # 1) Physical sanity: decays must be <= 0 (decaying)
    keep &= decays <= 0.0

    # 2) Nyquist: |freq| <= fNyq
    keep &= np.abs(freqs) <= fNyq

    # 3) Exclude near-constant signals:
    #    "low frequency"  ~ period > T/freqtol  <=> |f| < 2π * freqtol / T
    #    "slow decay"     ~ decay time 1/|d| > T/decaytol  <=> |d| < decaytol / T
    if (freqtol is not None) and (decaytol is not None) and (T > 0):
        low_freq = np.abs(freqs) < (2.0 * np.pi) * (freqtol / T)
        slow_decay = np.abs(decays) < (decaytol / T)
        # Exclude modes that are BOTH low-frequency AND slow-decay
        keep &= ~(low_freq & slow_decay)

    # 4) Amplitude threshold
    if amptol is not None:
        keep &= np.abs(amps) > amptol

    # 5) Error threshold
    if errtol is not None:
        # Treat NaN errs as failing the check
        keep &= np.isfinite(errs) & (errs <= errtol)

    # 6) Frequency pairing (+/- matches within tolerance)
    if matchtol is not None:
        # Keep only those freq entries that have an opposite-sign partner within `matchtol`
        fp_idx = np.where(freqs >= 0)[0]
        fm_idx = np.where(freqs < 0)[0]
        if fm_idx.size == 0:
            # No negative partners -> drop all positives if matching is required
            keep &= freqs < 0
        elif fp_idx.size == 0:
            # No positive partners -> drop all negatives if matching is required
            keep &= freqs >= 0
        else:
            # For each positive freq, check if any negative matches within tolerance
            # Vectorized: compute |f_p + f_m| for all pairs and see if any < matchtol
            fp = freqs[fp_idx][:, None]  # (P,1)
            fm = freqs[fm_idx][None, :]  # (1,N)
            paired_pos = (np.abs(fp + fm) < matchtol).any(axis=1)
            # Similarly, for negatives paired with positives
            paired_neg = (np.abs((-fm.T) + fp.T) < matchtol).any(
                axis=1
            )  # same condition, but symmetric

            mask_pair = np.zeros(M, dtype=bool)
            mask_pair[fp_idx] = paired_pos
            mask_pair[fm_idx] = paired_neg
            keep &= mask_pair

    # Apply mask, sort by |amp| descending
    if not keep.any():
        return np.zeros((0, 5), dtype=float)

    out = params[keep]
    order = np.abs(out[:, 2]).argsort()[::-1]
    return out[order]


##################################
# Matrix pencil utilities
##################################

"""
@author: zbb
@date: 20190811
@updates:2020-09-17 
@ref: Tapan K. Sakar and Odilon Pereira, Using the Matrix Pencil Method to Estimate the Parameters of Sum of Complex Exponetials, 
IEEE Antennas and Propagation Magazine, Vol. 37, No. 1, February 1995.
"""


def _constructY(y, N, L):
    """
    y: complex signal sequence.
    N: len(y)
    L: L<N, pencil parameter, N/3 < L < N/2 recommended.
    return: constructed Y matrix,
    [
        [y[0], y[1], ..., y[L-1]],
        [y[1], y[1, 0], ..., y[L]],
        ...
        [y[N-L-1], y[N-L], ..., y[N-1]]
    ]
    (N-L)*(L+1) matrix.
    """
    Y = np.zeros((N - L, L + 1), dtype=np.complex128)
    for k in range(N - L):
        Y[k, :] = y[k : (k + L + 1)]
    return Y


def _constructZM(Z, N):
    """
    Z: 1-D complex array.
    return N*M complex matrix (M=len(Z)):
    [
        [1,  1,  1, ..., 1 ],
        [z[0], z[1], .., z[M-1]],
        ...
        [z[0]**(N-1), z[1]**(N-1), ..., z[M-1]**(N-1)]
    ]
    """
    M = len(Z)
    ZM = np.zeros((N, M), dtype=np.complex128)
    for k in range(N):
        ZM[k, :] = Z**k
    return ZM


def _SVDFilter(Sp, p=3.0):
    """
    Sp: 1-D normed eigenvalues of Y after SVD, 1-st the biggest
    p: precise ditigits, default 3.0.
    return: M, M is the first integer that S[M]/S_max <= 10**(-p)
    """
    Sm = np.max(Sp)
    pp = 10.0 ** (-p)
    for m in range(len(Sp)):
        if Sp[m] / Sm <= pp:
            return m + 1
    return m + 1


def pencil(y, M=None, p=3.0, Lfactor=0.40):
    """
    Purpose:
      Complex exponential fit of a sampled complex waveform by Matrix Pencil Method.
    Authors:
      Zbb
    Arguments:
      N    - number of data samples. ==len(y)       [INPUT]
      y    - 1-D complex array of sampled signal.   [INPUT]
      dt   - sample interval.                       [INPUT]
      M    - pencil parameter.
             if None: use p to determin M.
             if given in range(0, Lfractor*N), then use it
             if given out of range, then use p to determin M.
      p    - precise digits of signal, default 8.0, corresponding to 10**(-8.0).
    Returns: (Z, R, M, (residuals, rank, s))
      Z    - 1-D Z array.
      R    - 1-D R array.
      M    - M in use.
      (residuals, rank, s)   - np.linalg.lstsq further results.
    Method:
      y[k] = y(k*dt) = sum{i=0--> M} R[i]*( Z[i]**k )
      Z[i] = exp(si*dt)

    Comment:
      To some extent, it is a kind of PCA method.
    """
    N = len(y)
    # better between N/3~N/2, pencil parameter:
    L = int(N * Lfactor)
    # construct Y matrix (Hankel data matrix) from signal y[i], shape=(N-L, L+1):
    Y = _constructY(y, N, L)
    # SVD of Y:
    _, S, V = np.linalg.svd(Y, full_matrices=True)
    # results: U.shape=(N-L, N-L), S.shape=(L+1, ), V.shape=(L+1, L+1)

    # find M:
    if M is None:
        M = _SVDFilter(np.abs(S), p=p)
    elif M not in range(0, L + 1):
        M = _SVDFilter(np.abs(S), p=p)
    else:
        pass

    # matrix primes based on M:
    # Vprime = V[0:M, :] # remove [M:] data set. only 0, 1, 2, ..., M-1 remains
    # Sprime = S[0:M]
    V1prime = V[0:M, 0:-1]  # remove last column
    V2prime = V[0:M, 1:]  # remove first column
    # smat = np.zeros((U.shape[0], M), dtype=np.complex128)
    # smat[:M, :M] = np.diag(Sprime)
    # Y1 = np.dot(U[:-1, :], np.dot(smat, V1prime))

    V1prime_H_MPinv = np.linalg.pinv(
        V1prime.T
    )  # find V1'^+ , Moore-Penrose pseudoinverse
    V1V2 = np.dot(V1prime_H_MPinv, V2prime.T)  # construct V1V2 = np.dot(V1'^+, V2')
    Z = np.linalg.eigvals(V1V2)  # find eigenvalues of V1V2. Zs.shape=(M,)
    # print(V1V2.shape, Z)

    # find R by solving least-square problem: Y = np.dot(ZM, R)
    ZM = np.row_stack([Z**k for k in range(N)])  # N*M
    R, residuals, rank, s = np.linalg.lstsq(ZM, y, rcond=-1)
    return (Z, R, M, (residuals, rank, s))


def simultaneous_pencil(y, M=None, p=3.0, Lfactor=0.40):
    """
    Purpose:
      Complex exponential fit of a sampled complex waveform by Matrix Pencil Method.
    Authors:
      Zbb
    Arguments:
      N    - number of data samples. ==len(y)       [INPUT]
      y    - 1-D complex array of sampled signal.   [INPUT]
      dt   - sample interval.                       [INPUT]
      M    - pencil parameter.
             if None: use p to determin M.
             if given in range(0, Lfractor*N), then use it
             if given out of range, then use p to determin M.
      p    - precise digits of signal, default 8.0, corresponding to 10**(-8.0).
    Returns: (Z, R, M, (residuals, rank, s))
      Z    - 1-D Z array.
      R    - 1-D R array.
      M    - M in use.
      (residuals, rank, s)   - np.linalg.lstsq further results.
    Method:
      y[k] = y(k*dt) = sum{i=0--> M} R[i]*( Z[i]**k )
      Z[i] = exp(si*dt)

    Comment:
      To some extent, it is a kind of PCA method.
    """
    C = y.shape[0]  # channels
    N = y.shape[1]
    # better between N/3~N/2, pencil parameter:
    L = int(N * Lfactor)

    Y = np.zeros((C, N - L, L + 1), dtype=np.complex128)
    for k in range(N - L):
        Y[:, k, :] = y[:, k : (k + L + 1)]

    # SVD of Y:
    _, S, V = np.linalg.svd(Y, full_matrices=True)
    # results: U.shape=(N-L, N-L), S.shape=(L+1, ), V.shape=(L+1, L+1)

    # find M:
    if M is None:
        raise Exception("M should have a non-None value")
    elif M not in range(0, L + 1):
        raise Exception(f"M should lie between 0 and (L+1)={L+1}")
    else:
        pass

    # matrix primes based on M:
    V1prime = np.concat(V[:, 0:M, 0:-1], axis=1)  # remove last column
    V2prime = np.concat(V[:, 0:M, 1:], axis=1)  # remove first column

    V1prime_H_MPinv = np.linalg.pinv(
        V1prime.T
    )  # find V1'^+ , Moore-Penrose pseudoinverse
    V1V2 = np.dot(V1prime_H_MPinv, V2prime.T)  # construct V1V2 = np.dot(V1'^+, V2')
    Z = np.linalg.eigvals(V1V2)  # find eigenvalues of V1V2. Zs.shape=(M,)
    # print(V1V2.shape, Z)

    # find R by solving least-square problem: Y = np.dot(ZM, R)
    ZM = np.row_stack([Z**k for k in range(N)])  # N*M
    # Initialize outputs
    R = np.zeros((C, ZM.shape[1]), dtype=np.complex128)
    residuals = np.zeros(C)
    rank = np.zeros(C)
    s = np.zeros(C)
    # for c in range(C):
    #     R[c], residuals[c], rank[c], s[c] = np.linalg.lstsq(ZM, y[c], rcond=-1)
    return (Z, R, M, (residuals, rank, s))
