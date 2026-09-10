import functools

import numpy as np
from scipy.linalg import expm


# 80db6c19-27e9-40f4-a759-4f4339c1372f
@functools.lru_cache(maxsize=8)
def _tables(n):
    """Index tables that depend only on n, cached because `probs` runs inside
    tight finite-difference loops and these are O(4^n) to rebuild.

    parity[j, z] = (-1)^{j . z}  (symmetric)
    weight[z]    = popcount(z)
    xor[z, z']   = z XOR z'
    """
    N = 1 << n
    z = np.arange(N)
    bits = (z[:, None] >> np.arange(n)[None, :]) & 1
    tables = ((-1.0) ** (bits @ bits.T), bits.sum(1), np.bitwise_xor.outer(z, z))
    for a in tables:
        a.setflags(write=False)
    return tables


def _embed(m, i, n):
    """Single-qubit operator m acting on qubit i of n, with qubit i = bit i of z."""
    return np.kron(np.eye(1 << (n - 1 - i)), np.kron(m, np.eye(1 << i)))


def lindbladian(theta, n, T1=np.inf, T2=np.inf):
    """Dense 4^n x 4^n Lindblad generator, acting on the C-order (row-major)
    flattening of rho.  For that convention vec(A rho B) = kron(A, B.T) vec(rho),
    so

        d rho / dt = -i [H, rho]
                     + g1     sum_i (s_i rho s_i^dag - {s_i^dag s_i, rho} / 2)
                     + g_phi/2 sum_i (Z_i rho Z_i - rho)

    becomes

        L = -i (kron(H, I) - kron(I, H.T))
            + g1     sum_i [ kron(s_i, conj(s_i)) - (kron(ni, I) + kron(I, ni.T))/2 ]
            + g_phi/2 sum_i [ kron(Z_i, Z_i.T) - kron(I, I) ]

    with H|z> = E(z)|z>, E(z) = sum_j theta_j (-1)^{j.z}, s = |0><1| the lowering
    operator, ni = s^dag s the number operator, g1 = 1/T1 the relaxation rate and
    g_phi = 1/T2 - 1/(2 T1) the pure-dephasing rate (so a lone coherence decays
    as exp(-t/T2), the usual convention).  T2 = inf means no pure dephasing,
    leaving T1 to set the coherence time; T2 > 2 T1 is unphysical and raises.
    """
    N = 1 << n
    parity, _, _ = _tables(n)
    g1 = 0.0 if np.isinf(T1) else 1.0 / T1
    g_phi = 0.0 if np.isinf(T2) else 1.0 / T2 - 0.5 * g1
    if g_phi < 0:
        raise ValueError(
            f"T2={T2} exceeds 2*T1={2 * T1}, so the pure dephasing rate "
            "1/T2 - 1/(2 T1) is negative"
        )

    eye = np.eye(N)
    H = np.diag(parity @ theta).astype(complex)
    L = -1j * (np.kron(H, eye) - np.kron(eye, H.T))
    lower = np.array([[0.0, 1.0], [0.0, 0.0]])  # |0><1|
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    for i in range(n):
        if g1:
            s = _embed(lower, i, n)
            num = s.T @ s  # s^dag s, real
            L += g1 * (
                np.kron(s, s.conj()) - 0.5 * (np.kron(num, eye) + np.kron(eye, num.T))
            )
        if g_phi:
            zi = _embed(pauli_z, i, n)
            L += 0.5 * g_phi * (np.kron(zi, zi.T) - np.kron(eye, eye))
    return L


def unit_channel(theta, n, t, T1=np.inf, T2=np.inf):
    """Superoperator of one repetition, exp(L t), as a 4^n x 4^n matrix acting on
    vec(rho).  The channel for r repetitions is its r-th matrix power, because L
    is time independent."""
    return expm(lindbladian(theta, n, T1, T2) * t)


def probs(theta, n, t, reps=1, T1=np.inf, T2=np.inf):
    """X-basis outcome distribution after 1, 2, ..., `reps` evolutions of length t.

    Returns a (2**n, reps) array whose column r-1 is p_k at time r*t, i.e.
    p_k = <k| E_t^r(|+..+><+..+|) |k> with |k> the X-basis state (-1)^{k.z}/sqrt(N).
    With reps=1 and T1=T2=inf this is the old f_k = |<psi_k| e^{-iHt} |+..+>|^2.

    n: system size
    t: length of one repetition (t >= 0 once T1 is finite)
    theta: parameter vector of size 2**n, H|z> = E(z)|z>, E(z) = sum_j theta_j (-1)^{j.z}
    T1, T2: single-qubit relaxation / total coherence times (same units as t).
        Defaults np.inf = closed system, i.e. the original noiseless model.

    Dissipation model: independent single-qubit amplitude damping at rate
    g1 = 1/T1 plus pure dephasing at rate g_phi = 1/T2 - 1/(2 T1), so a lone
    coherence decays as exp(-t/T2) as usual.  T2 = inf means "no pure
    dephasing", leaving T1 to set the coherence time; T2 > 2 T1 is unphysical
    and raises.

    Dephasing alone commutes with the (diagonal) unitary, so both act as one
    elementwise factor on rho_{z,z'} and the amplitudes are enough: the noisy
    distribution is the clean one pushed through an independent bit-flip
    channel, lam^{|z XOR z'|} with lam = exp(-t/T2), which is a diagonal
    rescaling by lam^{|j|} in the Walsh domain -- one extra pair of transforms.

    Relaxation does not commute with either: sigma^- rho sigma^+ moves
    rho_{z+e_i, z'+e_i} into rho_{z,z'}, and those two elements carry different
    Hamiltonian phases E(z)-E(z') vs E(z+e_i)-E(z'+e_i).  So the T1 branch drops
    to the full density matrix: it builds the one-repetition superoperator
    exp(L t) with `unit_channel` and steps it, since L is time independent and
    hence rho(r t) = E_t^r(rho_0) = E_t(rho((r-1) t)).  Stepping is the same
    matrix power the docstring of `unit_channel` advertises, just accumulated
    left to right so every column costs one 4^n x 4^n mat-vec instead of a
    4^n x 4^n mat-mat.

    Cost: the dense expm is 4^n x 4^n, so this branch is practical to about
    n = 5 and out of reach by n = 7 (a 16384^2 complex matrix is 4.3 GB).
    """
    N = 1 << n
    parity, weight, xor = _tables(n)
    E = parity @ theta

    if np.isinf(T1):
        times = t * np.arange(1, reps + 1)
        f = np.abs(np.exp(-1j * times[:, None] * E[None, :]) @ parity) ** 2 / N**2
        if np.isinf(T2):
            return f.T  # closed system, skip the transforms
        lam = np.exp(-np.abs(times) / T2)
        return (((lam[:, None] ** weight[None, :]) * (f @ parity)) @ parity / N).T

    if t < 0:
        raise ValueError(f"relaxation is irreversible, need t >= 0, got {t}")

    step = unit_channel(theta, n, t, T1, T2)
    vec = np.full(N * N, 1.0 / N, dtype=complex)  # vec(|+..+><+..+|)
    zs = np.arange(N)
    out = np.empty((N, reps))
    for r in range(reps):
        vec = step @ vec
        rho = vec.reshape(N, N)
        # p_k = sum_m (-1)^{k.m} g(m) / N with g(m) = sum_z rho_{z, z XOR m}
        out[:, r] = (parity @ rho[zs[:, None], xor].sum(0)).real / N
    return out


def jac_fd(theta, n, reps, t, h=1e-6, T1=np.inf, T2=np.inf):
    """Central-difference Jacobian, shape (reps, 2**n, 2**n):
    J[r, k, j] = d p_k(time (r+1) t) / d theta_j."""
    N = 1 << n
    J = np.empty((reps, N, N))
    e = np.zeros(N)
    for j in range(N):
        e[j] = h
        J[:, :, j] = (
            (
                probs(theta + e, n, t, reps, T1, T2)
                - probs(theta - e, n, t, reps, T1, T2)
            )
            / (2 * h)
        ).T
        e[j] = 0.0
    return J


def fim_terms(theta, n, reps, t, h=None, shots=1, T1=np.inf, T2=np.inf):
    """Per-repetition multinomial FIM contributions, shape (reps, 2**n, 2**n).

    Each repetition is an independent multinomial experiment, so the FIM of a
    protocol using repetitions 1..r is terms[:r].sum(0), and np.cumsum(terms, 0)
    gives the FIM for every prefix of a rep_range from a single evolution.
    """
    N = 1 << n
    p = probs(theta, n, t, reps, T1, T2)
    J = jac_fd(theta, n, reps, t, h if h is not None else 1e-6, T1, T2)
    terms = np.empty((reps, N, N))
    for r in range(reps):
        good = p[:, r] > 1e-300  # exact zeros are 0/0, not 0
        Jr = J[r][good]
        terms[r] = shots * (Jr.T / p[good, r]) @ Jr
    return terms


def fim(theta, n, reps, t, h=None, shots=1, T1=np.inf, T2=np.inf):
    """Multinomial FIM of repetitions 1..reps, shots * sum_k dp_k dp_k^T / p_k.

    Equals J^T Sigma^+ J for the multinomial covariance Sigma = diag(p) - p p^T,
    because the columns of J are tangent to the simplex (sum_k dp_k = 0).  The
    dissipation of `probs` is a stochastic map, so it preserves both properties.
    """
    return fim_terms(theta, n, reps, t, h, shots, T1, T2).sum(0)


if __name__ == "__main__":
    import pathlib

    import matplotlib.pyplot as plt

    outdir = pathlib.Path(__file__).parent / "output"
    outdir.mkdir(exist_ok=True)

    rng = np.random.default_rng(10)
    t = 1
    T1 = 80  # in units of t
    T2 = 80  # in units of t

    print("\n=== FIM as theta -> 0 along a random direction ===")
    # dense expm is 4^n x 4^n: n=5 costs ~32 s per direction, n=7 needs 4.3 GB
    n_range = np.arange(3, 4)
    rep_range = np.arange(1, 71)
    max_rep = int(rep_range[-1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for n in n_range:
        N = 1 << n
        v = rng.normal(size=(4, N))
        v[:, 0] = 0.0  # identity term = global phase, exclude
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        min_svals = np.empty((4, max_rep))
        max_svals = np.empty((4, max_rep))
        for idx, eps in enumerate([1e-4, 1e-4, 1e-4, 1e-4]):
            # one evolution out to max_rep; the FIM of reps 1..r is the running
            # sum of the per-repetition terms, so the whole sweep is free
            cum = np.cumsum(
                fim_terms(eps * v[idx], n, max_rep, t, h=eps * 1e-4, T1=T1, T2=T2),
                axis=0,
            )
            for r, rep in enumerate(rep_range):
                sub = cum[r][1:, 1:]
                svals = np.linalg.svd(sub, compute_uv=False)
                min_svals[idx, r] = svals[-1]
                max_svals[idx, r] = svals[0]
                off = sub - np.diag(np.diag(sub))
                print(
                    f"n={n}  rep={rep}  eps={eps:7.0e}  diag mean={np.diag(sub).mean():.6f}  "
                    f"diag spread={np.ptp(np.diag(sub)):.2e}  max|offdiag|={np.abs(off).max():.2e}  "
                    f"min sval={svals[-1]:.2e} max sval={svals[0]:.2e}"
                )
        axes[0].loglog(rep_range, min_svals.mean(0), marker="o", label=f"n={n}")
        axes[1].loglog(rep_range, max_svals.mean(0), marker="o", label=f"n={n}")

    for ax in axes.flat:
        ax.set_xlabel("Number of repetitions (rep) (log scale)")
        ax.set_ylabel("Mean FIM smallest singular value (log scale)")
        ax.grid(True, which="both")
    axes[0].set_title(
        f"Mean smallest singular value of FIM vs repetitions (log-log)\n"
        f"T1={T1}, T2={T2}"
    )
    axes[1].set_title(
        f"Max smallest singular value of FIM vs repetitions (log-log)\n"
        f"T1={T1}, T2={T2}"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.95))
    plt.tight_layout()
    outfile = outdir / f"fim_rep_n_T1={T1:g}_T2={T2:g}.pdf"
    plt.savefig(outfile)
    print(f"\nsaved {outfile}")
    plt.show()
    # print("prediction 4t^2 =", 4 * t**2)

    # print("\n=== identity-parameter row (theta_0, global phase) ===")
    # J = jac_fd(1e-3 * v, n, 1, t, h=1e-7)
    # print("max |df_k/dtheta_0| =", np.abs(J[..., 0]).max())
