"""Two-qubit DB model: parameters, db_set definitions and outcome probabilities.

Interface:
    PARAMS, PARAM_NAMES, PHASE_NAMES -- the fit parameters, in the one order every
        array of them uses.
    DbSet, DB_SETS -- one entry per experiment: its pulse sequence, readout basis,
        Pauli names, fixed parameters, bounds and initial values.
    MODEL -- which model `probabilities` uses: "mix" (first-order toggling-frame
        average of the sequence) or "model_dd" (the sequence gate by gate).
    probabilities(entry, params, n) -- P(00), P(01), P(10), P(11) after n repetitions.
    construct_unit_op(entry, params) -- superoperator of one repetition ("model_dd").

`params` is always a {name: value} dict over PARAM_NAMES.

eta, eps, kap are the coefficients of the db_set's Pauli error triple (ZZ, ZI, IZ for
set1); d1, d2, r1, r2 the dephasing and relaxation rates; ep1, em1, ep2, em2 the readout
confusion; z1, z2, z12 the un-refocused ZI, IZ, ZZ residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import NamedTuple

import numpy as np
from scipy.linalg import expm

# Which model `probabilities` uses unless a caller asks for one: "mix" or "model_dd".
MODEL = "mix"


############
# Parameters
############
class Param(NamedTuple):
    name: str
    group: str  # "phase", "decay", "spam" or "residual"
    restart: tuple  # uniform range a fresh random restart draws it from


Z_INIT_SCALE = 0.2

PARAMS = (
    Param("eta", "phase", (-0.02, 0.02)),
    Param("eps", "phase", (-0.02, 0.02)),
    Param("kap", "phase", (-0.02, 0.02)),
    Param("d1", "decay", (0.0, 1 / 100)),
    Param("d2", "decay", (0.0, 1 / 100)),
    Param("r1", "decay", (0.0, 1 / 100)),
    Param("r2", "decay", (0.0, 1 / 100)),
    Param("ep1", "spam", (0.0, 0.2)),
    Param("em1", "spam", (0.0, 0.2)),
    Param("ep2", "spam", (0.0, 0.2)),
    Param("em2", "spam", (0.0, 0.2)),
    Param("z1", "residual", (-Z_INIT_SCALE, Z_INIT_SCALE)),
    Param("z2", "residual", (-Z_INIT_SCALE, Z_INIT_SCALE)),
    Param("z12", "residual", (-Z_INIT_SCALE, Z_INIT_SCALE)),
)
PARAM_NAMES = [p.name for p in PARAMS]
# Every coefficient of a term in the coherent-error Hamiltonian.
PHASE_NAMES = [p.name for p in PARAMS if p.group in ("phase", "residual")]
# The generator is linear in every non-SPAM parameter; this is the order of its basis.
_GENERATOR_NAMES = [p.name for p in PARAMS if p.group != "spam"]
_SLOTS = {
    group: [_GENERATOR_NAMES.index(p.name) for p in PARAMS if p.group == group]
    for group in ("phase", "decay", "residual")
}

# Probabilities below this are treated as this value when they set the weights, so a
# model prediction that runs to zero cannot produce an infinite weight.
MIN_WEIGHT_PROB = 1e-12


############
# Operators
############
I2 = np.eye(2)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]])
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]])
H = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]])
SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]])  # lowering operator


def _on(m: np.ndarray, qubit: int) -> np.ndarray:
    """Single-qubit operator `m` acting on `qubit` (0 = left factor) of the pair."""
    return np.kron(m, I2) if qubit == 0 else np.kron(I2, m)


def _hamiltonian_super(h: np.ndarray) -> np.ndarray:
    """Superoperator of -i [h, .], using vec(A rho B) = kron(A, B.T)."""
    eye = np.eye(len(h))
    return -1j * (np.kron(h, eye) - np.kron(eye, h.T))


def _dissipator_super(c: np.ndarray) -> np.ndarray:
    """Superoperator of c rho c^dag - {c^dag c, rho} / 2, for jump operator `c`."""
    eye = np.eye(len(c))
    num = c.conj().T @ c  # c^dag c
    return np.kron(c, c.conj()) - 0.5 * (np.kron(num, eye) + np.kron(eye, num.T))


def _super(u: np.ndarray) -> np.ndarray:
    """Superoperator of rho -> u rho u^dag.

    Building a pulse this way (rather than kron(U, U) by hand) keeps the conjugate on
    the imaginary Pauli-Y pulses, whose sign would otherwise only cancel by accident
    because each appears an even number of times per half-repetition.
    """
    return np.kron(u, u.conj())


############
# Level structure
############
QUBIT_LEVELS = (2, 2)
# Levels kept on (control, target)
LEVELS = QUBIT_LEVELS
DIM = np.prod(LEVELS)

# Readout bins every leakage level with this computational level: a transmon
# discriminator lands |2> in the |1> window far more often than the |0> one.
LEAK_READOUT_LEVEL = 1


def _number(d: int) -> np.ndarray:
    """Number operator diag(0, 1, ..., d-1).

    Its dissipator dephases the 0-1 coherence at exactly the rate 0.5 * Z did (a
    dissipator only sees *differences* of a diagonal jump operator, and both have
    c_0 - c_1 = 1), and a leakage level dephases at the usual level-squared rate.
    """
    return np.diag(np.arange(d, dtype=float))


def _lower(d: int) -> np.ndarray:
    """Bosonic lowering operator sum_k sqrt(k) |k-1><k|."""
    return np.diag(np.sqrt(np.arange(1, d)), 1)


def construct_decay_basis(levels: tuple[int, int] = QUBIT_LEVELS) -> np.ndarray:
    """The four dissipator superoperators scaled by (d1, d2, r1, r2)."""
    la, lb = levels
    eye_a, eye_b = np.eye(la), np.eye(lb)
    return np.array(
        [
            _dissipator_super(np.kron(_number(la), eye_b)),  # scaled by d1
            _dissipator_super(np.kron(eye_a, _number(lb))),  # scaled by d2
            _dissipator_super(2.0 * np.kron(_lower(la), eye_b)),  # scaled by r1
            _dissipator_super(2.0 * np.kron(eye_a, _lower(lb))),  # scaled by r2
        ],
        dtype=complex,
    )


def construct_cz(levels: tuple[int, int] = QUBIT_LEVELS) -> np.ndarray:
    """CZ on the computational block, identity on every leakage level.

    The conditional phase the *ideal* gate applies is the one on |11>; whatever phase
    the real pulse leaves on |02> belongs in the coherent-error Hamiltonian.
    """
    la, lb = levels
    diag = np.ones(la * lb)
    diag[lb + 1] = -1.0  # |11>
    # diag[lb - 1] = -1.0  # |02>
    return np.diag(diag)


def construct_readout_rotation(levels: tuple[int, int] = QUBIT_LEVELS) -> np.ndarray:
    """Basis-change for an X-basis readout, one Hadamard per subsystem.

    The pre-measurement pulse only drives the computational transition, so it acts as
    the identity on a leakage level.
    """
    mats = []
    for d in levels:
        m = np.eye(d)
        m[:2, :2] = H
        mats.append(m)
    return np.kron(mats[0], mats[1])


def _readout_bin(level: int) -> int:
    """The computational outcome a raw level is reported as."""
    return level if level < 2 else LEAK_READOUT_LEVEL


def construct_ideal_msmt_ops(
    rot: np.ndarray = None, levels: tuple[int, int] = QUBIT_LEVELS
) -> np.ndarray:
    """The four ideal outcome operators, rotated into the measured basis by `rot`.

    `rot = construct_readout_rotation(levels)` gives the X basis, `rot = None` (the
    default) the Z basis.

    A leakage level is counted in the outcome named by LEAK_READOUT_LEVEL.
    """
    la, lb = levels
    dim = la * lb
    ops = np.zeros((4, dim, dim), dtype=complex)
    for a in range(la):
        for b in range(lb):
            idx = a * lb + b
            ops[2 * _readout_bin(a) + _readout_bin(b), idx, idx] = 1.0
    if rot is not None:
        ops = rot.conj().T @ ops @ rot
    return ops.reshape(4, -1)


def construct_init_state(
    rot: np.ndarray = None, levels: tuple[int, int] = QUBIT_LEVELS
):
    """vec of the prepared state |00>, rotated by `rot`."""
    dim = levels[0] * levels[1]
    op = np.zeros((dim, dim), dtype=complex)
    op[0, 0] = 1.0
    if rot is not None:
        op = rot.conj().T @ op @ rot
    return op.reshape(-1)


def construct_msmt_op(
    ep1, em1, ep2, em2, rot: np.ndarray = None, levels: tuple[int, int] = QUBIT_LEVELS
):
    """Outcome operators with the 4x4 readout confusion matrix folded in.

    The confusion matrix stays 4x4 because it acts on the reported outcomes, which
    `construct_ideal_msmt_ops` has already binned down to four.
    """
    a = np.array([[1 - ep1, em1], [ep1, 1 - em1]])
    b = np.array([[1 - ep2, em2], [ep2, 1 - em2]])
    confusion = (a[:, None, :, None] * b[None, :, None, :]).reshape(4, 4)
    return confusion @ construct_ideal_msmt_ops(rot, levels)


# The lab-frame generator basis of the "synthetic" db_set, in _GENERATOR_NAMES order:
# the Hamiltonian terms carry one phase each, and each jump operator is a fixed matrix
# scaled by its rate. The generator is then a single (256, 10) @ (10,) product.
_GENERATOR_BASIS = np.array(
    [
        _hamiltonian_super(np.kron(SIGMA_Z, SIGMA_Z)),  # eta
        _hamiltonian_super(_on(SIGMA_Z, 0)),  # eps
        _hamiltonian_super(_on(SIGMA_Z, 1)),  # kap
        _dissipator_super(0.5 * _on(SIGMA_Z, 0)),  # scaled by d1
        _dissipator_super(0.5 * _on(SIGMA_Z, 1)),  # scaled by d2
        _dissipator_super(2.0 * _on(SIGMA_MINUS, 0)),  # scaled by r1
        _dissipator_super(2.0 * _on(SIGMA_MINUS, 1)),  # scaled by r2
        _hamiltonian_super(_on(SIGMA_Z, 0)),  # z1
        _hamiltonian_super(_on(SIGMA_Z, 1)),  # z2
        _hamiltonian_super(np.kron(SIGMA_Z, SIGMA_Z)),  # z12
    ],
    dtype=complex,
)


############
# Propagators
############
# cond(V) past which an `eig` basis is too ill-conditioned to exponentiate through,
# so `evolve` falls back to scaling-and-squaring instead.
_MAX_EIGENBASIS_COND = 1e8


@lru_cache(maxsize=8)
def _eigendecompose(
    key: bytes, dim: int, kind: str
) -> tuple[np.ndarray, np.ndarray, str]:
    """Eigendecomposition of the (dim, dim) complex matrix whose buffer is `key`.

    Returns (eigenvalues, eigenvectors, mode), where mode tells `evolve` how to
    reassemble the propagator: "unitary" if the eigenbasis is orthonormal, "solve" if
    it has to be inverted, "expm" if it is too ill-conditioned to use at all (then the
    first two entries are None).

    `kind` is what the caller already knows about the matrix: "hamiltonian" for the
    anti-Hermitian generator of a coherent rotation, "dissipator" for one that is
    neither, "unknown" to work it out.

    Keyed on the raw bytes so that repeated calls with the *same* Liouvillian at
    different times share one decomposition; the arrays are handed out read-only
    because every caller gets the same objects back.
    """
    L = np.frombuffer(key, dtype=complex).reshape(dim, dim)
    # Use `eigh` on Hermitian and anti-Hermitian superoperators to avoid singularities
    # in calculating eigenvalues.
    anti_hermitian = kind == "hamiltonian" or (
        kind == "unknown" and np.allclose(L, -L.conj().T)
    )
    if anti_hermitian:
        mu, eigenvectors = np.linalg.eigh(1j * L)
        eigenvalues, mode = -1j * mu, "unitary"
    elif kind == "unknown" and np.allclose(L, L.conj().T):
        values, eigenvectors = np.linalg.eigh(L)
        eigenvalues, mode = values.astype(complex), "unitary"
    else:
        # Usually for the dissipator.
        eigenvalues, eigenvectors = np.linalg.eig(L)
        # 1-norm rather than the default 2-norm: it is an LU rather than an SVD, and
        # the two agree to within a factor of dim, which decides nothing at 1e8.
        if np.linalg.cond(eigenvectors, 1) > _MAX_EIGENBASIS_COND:
            return None, None, "expm"
        mode = "solve"
    eigenvalues.flags.writeable = False
    eigenvectors.flags.writeable = False
    return eigenvalues, eigenvectors, mode


def evolve(L: np.ndarray, t: float, kind: str = "unknown"):
    """Given a time-constant Louivillian in superoperator form (d^2 x d^2), get the propogator for
    corresponding to time t"""
    # exp(L t) = V diag(exp(w t)) V^-1
    # Scale the columns of V by exp(w t) and solve against V.T rather than forming
    # V^-1. Same trick as `probabilities`.
    #
    # It also amortizes across calls: `construct_unit_op` evolves one dissipator at
    # both of the sequence's dwell times, and the cache turns those into a single eig.
    L = np.ascontiguousarray(L, dtype=complex)
    eigenvalues, eigenvectors, mode = _eigendecompose(L.tobytes(), L.shape[0], kind)
    if mode == "expm":
        return expm(L * t)
    scaled = eigenvectors * np.exp(eigenvalues * t)
    if mode == "unitary":
        return scaled @ eigenvectors.conj().T
    return np.linalg.solve(eigenvectors.T, scaled.T).T


CZ = construct_cz(LEVELS)
# vec form of rho -> CZ rho CZ^dag, i.e. kron(CZ, CZ.conj()).
CZ_SUPER = np.kron(CZ, CZ.conj())
# Dissipators scaled by (d1, d2, r1, r2), in the units `_decay_init` produces.
DECAY_BASIS = construct_decay_basis(LEVELS)
SQ_GT, TQ_GT, TQ_ID = 32, 60, 20


def _decay_super(d1, d2, r1, r2) -> np.ndarray:
    """The dissipator at rates (d1, d2, r1, r2), which are given in 1/us."""
    return 1e-3 * (
        d1 * DECAY_BASIS[0]
        + d2 * DECAY_BASIS[1]
        + r1 * DECAY_BASIS[2]
        + r2 * DECAY_BASIS[3]
    )


############
# Pulses and error terms
############
_XI = np.kron(SIGMA_X, np.eye(2))
_IX = np.kron(np.eye(2), SIGMA_X)
_YI = np.kron(SIGMA_Y, np.eye(2))
_IY = np.kron(np.eye(2), SIGMA_Y)
_IZ = np.kron(np.eye(2), SIGMA_Z)
_ZI = np.kron(SIGMA_Z, np.eye(2))
_XY = _XI @ _IY
_YX = _YI @ _IX
_ZX = _ZI @ _IX
_XZ = _XI @ _IZ
_HH = np.kron(H, H)
_IH = np.kron(np.eye(2), H)
_HI = np.kron(H, np.eye(2))
_II = np.eye(DIM)

XI, IX, YI, IY = _super(_XI), _super(_IX), _super(_YI), _super(_IY)
XY, YX = _super(_XY), _super(_YX)
HH, IH, HI = _super(_HH), _super(_IH), _super(_HI)

_ZZ_ERR = (
    np.kron(SIGMA_Z, SIGMA_Z),
    _on(SIGMA_Z, 0),
    _on(SIGMA_Z, 1),
)
_YY_ERR = (
    np.kron(SIGMA_Y, SIGMA_Y),
    _on(SIGMA_Y, 0),
    _on(SIGMA_Y, 1),
)
_XX_ERR = (
    np.kron(SIGMA_X, SIGMA_X),
    _on(SIGMA_X, 0),
    _on(SIGMA_X, 1),
)
_SET4_ERR = (
    np.kron(SIGMA_Z, SIGMA_X),
    np.kron(SIGMA_X, SIGMA_Y),
    np.kron(SIGMA_Y, SIGMA_Z),
)
_SET5_ERR = (
    np.kron(SIGMA_X, SIGMA_Z),
    np.kron(SIGMA_Y, SIGMA_X),
    np.kron(SIGMA_Z, SIGMA_Y),
)
# The un-refocused residual, in (z1, z2, z12) order.
_Z_RESIDUAL = np.array([_on(SIGMA_Z, 0), _on(SIGMA_Z, 1), np.kron(SIGMA_Z, SIGMA_Z)])


def _cz_block(err_ops, ideal=_II):
    return (ideal, err_ops, TQ_GT)


def _id_block():
    return (np.eye(4), None, TQ_ID)


def _pulse_block(u):
    return (u, None, SQ_GT)


############
# db_sets
############
class _Compiled(NamedTuple):
    """A sequence with everything independent of the fit parameters worked out.

    `construct_unit_op` runs on every residual and every finite-difference column of
    every restart, so it should not be rebuilding the same fixed matrices each time.
    """

    steps: tuple  # (pulse superoperator, is_cz, dwell_ns)
    err_supers: tuple  # the (eta, eps, kap) Hamiltonian superoperators
    dwells: tuple  # the distinct dwell times, of which there are only two
    residual_supers: tuple  # the (z1, z2, z12) Hamiltonian superoperators, or ()


@dataclass(frozen=True, eq=False)
class DbSet:
    """One experiment: how it is played, read out, labelled and fit.

    blocks: one half repetition of the DD sequence, in time order, as
        (pulse 4x4, err_ops or None, dwell_ns). `pulse` is the ideal gate of the
        block; `err_ops` the (eta, eps, kap) Pauli triple a CZ block's coherent error
        is written in, or None for a single-qubit pulse (taken to be error free);
        `dwell_ns` how long the dissipator acts for. None for an entry defined by its
        generator basis alone.
    generator_basis: the "mix" generator basis, in _GENERATOR_NAMES order. None means
        derive it from `blocks` (`mix_basis`).
    readout_rot: pre-measurement rotation, None for the Z basis.
    residual_ops: the (z1, z2, z12) residual operators, or None.
    pauli_labels: the Pauli term each phase parameter multiplies.
    fixed: parameters held at a value during the fit.
    lower, upper: fit bounds for every parameter.
    init: the fit's first starting point, per parameter: ("uniform", low, high) is drawn,
        ("value", v) is taken as is. Every parameter is drawn (in PARAM_NAMES order)
        even when fixed, so the random stream does not depend on what is fixed.

    Hashed by identity, so `dataclasses.replace` gives an entry with its own caches.
    """

    name: str
    blocks: tuple | None
    generator_basis: np.ndarray | None
    readout_rot: np.ndarray | None
    residual_ops: np.ndarray | None
    pauli_labels: dict
    fixed: dict
    lower: dict
    upper: dict
    init: dict

    @cached_property
    def compiled(self) -> _Compiled:
        """The "model_dd" form of `blocks`."""
        if self.blocks is None:
            raise ValueError(f"db_set {self.name!r} has no pulse sequence")
        err_ops = next(ops for _, ops, _ in self.blocks if ops is not None)
        return _Compiled(
            steps=tuple(
                (_super(pulse), ops is not None, dwell)
                for pulse, ops, dwell in self.blocks
            ),
            err_supers=tuple(_hamiltonian_super(op) for op in err_ops),
            dwells=tuple({dwell for _, _, dwell in self.blocks}),
            residual_supers=(
                ()
                if self.residual_ops is None
                else tuple(_hamiltonian_super(op) for op in self.residual_ops)
            ),
        )

    @cached_property
    def mix_basis(self) -> np.ndarray:
        """The "mix" generator basis: `generator_basis`, or the sequence's average.

        The "mix" model has no DD pulses: it exponentiates one 16x16 generator for
        `4 * n`, four gates per repetition. We want that generator to approximate
        `construct_unit_op`, the simplest version of which is the first-order Magnus
        (toggling-frame) average. Each block's contribution is conjugated by the
        pulses played before it, weighted by how long the block lasts:

            G_eff = (1 / 4) sum_j W_j^dag (L_err_j + dwell_j L_diss) W_j,

        where W_j is the cumulative pulse superoperator through block j. For the
        coherent part it just reproduces the surviving Pauli triple in the frame the
        pulses leave it in. Relaxation operators get symmetrized to some extent.
        """
        if self.generator_basis is not None:
            return self.generator_basis
        # The whole repetition, not one half doubled: the pulses of the first half do
        # not compose back to the identity, so the second half runs in a frame the
        # first half left rotated. For set2 that is the difference between relaxation
        # coming out 1/3 sigma^- and 2/3 sigma^+ and its true even split.
        basis = np.zeros((len(_GENERATOR_NAMES), DIM**2, DIM**2), dtype=complex)
        frame = np.eye(DIM**2, dtype=complex)
        for pulse, err_ops, dwell in self.blocks * 2:
            frame = _super(pulse) @ frame
            if err_ops is not None:
                for slot, op in zip(_SLOTS["phase"], err_ops):
                    basis[slot] += frame.conj().T @ _hamiltonian_super(op) @ frame
            for k, slot in enumerate(_SLOTS["decay"]):
                basis[slot] += 1e-3 * dwell * (frame.conj().T @ DECAY_BASIS[k] @ frame)
            if self.residual_ops is not None:
                for slot, op in zip(_SLOTS["residual"], self.residual_ops):
                    basis[slot] += _hamiltonian_super(op)
        # `probabilities` exponentiates for 4 * n, four gates per repetition.
        basis /= 4
        # Every caller shares the cached array.
        basis.flags.writeable = False
        return basis


def _decay_init(t2: list, t1: list) -> dict:
    """Initial (d1, d2, r1, r2) values, in 1/us.

    Both methods integrate the dissipator over the real gate durations in ns, so d
    and r are absolute rates rather than per-repetition fractions:
    d = 1000 / T_phi and r = 1000 / T1, where 1 / T_phi = 1 / T2 - 1 / (2 T1).
    """
    values = [
        1000 * (1 / t2[0] - 1 / (2 * t1[0])),
        1000 * (1 / t2[1] - 1 / (2 * t1[1])),
        1000 / t1[0],
        1000 / t1[1],
    ]
    return {name: ("value", v) for name, v in zip(("d1", "d2", "r1", "r2"), values)}


# Bounds and starting points the entries below share.
_LOWER = {
    **{name: -0.3 for name in ("eta", "eps", "kap")},
    **{name: 0.0 for name in ("d1", "d2", "r1", "r2")},
    **{name: 0.0 for name in ("ep1", "em1", "ep2", "em2")},
    **{name: -0.3 for name in ("z1", "z2", "z12")},
}
_UPPER = {
    **{name: 0.3 for name in ("eta", "eps", "kap")},
    **{name: 3.0 for name in ("d1", "d2", "r1", "r2")},
    **{name: 0.3 for name in ("ep1", "em1", "ep2", "em2")},
    **{name: 0.3 for name in ("z1", "z2", "z12")},
}
_T2, _T1 = [11000, 21000], [10000, 30000]
_SPAM_AND_RESIDUAL_INIT = {
    **{name: ("uniform", 0, 0.1) for name in ("ep1", "em1", "ep2", "em2")},
    **{name: ("uniform", -Z_INIT_SCALE, Z_INIT_SCALE) for name in ("z1", "z2", "z12")},
}
_INIT = {
    **{name: ("uniform", 0, 0.01) for name in ("eta", "eps", "kap")},
    **_decay_init(_T2, _T1),
    **_SPAM_AND_RESIDUAL_INIT,
}
_INIT_SET1 = {
    **{name: ("uniform", 0, 0.3) for name in ("eta", "eps", "kap")},
    **_decay_init(_T2, _T1),
    **_SPAM_AND_RESIDUAL_INIT,
}
_Z_FIXED = {"z1": 0.0, "z2": 0.0, "z12": 0.0}
_ZZ_LABELS = {"eta": "ZZ", "eps": "ZI", "kap": "IZ", "z1": "ZI", "z2": "IZ", "z12": "ZZ"}

# Each experiment set is one half repetition of a DD sequence.
#   set1 -> ZZ, ZI, IZ      set2 -> YY, YI, IY      set3 -> XX, XI, IX
#   set4 -> ZX, XY, YZ      set5 -> XZ, YX, ZY
DB_SETS = {
    entry.name: entry
    for entry in (
        # H CZ H  X  H CZ H  X, with the ideal CZ kept (it commutes with the error).
        DbSet(
            name="qubit_pairq3-6",
            blocks=(
                _id_block(),
                _cz_block(_ZZ_ERR, ideal=CZ),
                _id_block(),
                _id_block(),
                _cz_block(_ZZ_ERR, ideal=CZ),
                _id_block(),
            ),
            generator_basis=None,
            readout_rot=construct_readout_rotation(LEVELS),
            residual_ops=_Z_RESIDUAL,
            pauli_labels=_ZZ_LABELS,
            fixed=_Z_FIXED,
            lower=_LOWER,
            upper=_UPPER,
            init=_INIT,
        ),
        DbSet(
            name="db_set0_qubit_pairq3-6",
            blocks=(
                _id_block(),
                _cz_block(_ZZ_ERR, ideal=CZ),
                _id_block(),
                _id_block(),
                _cz_block(_ZZ_ERR, ideal=CZ),
                _id_block(),
            ),
            generator_basis=None,
            readout_rot=construct_readout_rotation(LEVELS),
            residual_ops=_Z_RESIDUAL,
            pauli_labels=_ZZ_LABELS,
            fixed=_Z_FIXED,
            lower=_LOWER,
            upper=_UPPER,
            init=_INIT,
        ),
        DbSet(
            name="db_set1_qubit_pairq3-6",
            blocks=(
                _id_block(),
                _cz_block(_ZZ_ERR, ideal=CZ),
                _id_block(),
                _pulse_block(_ZI),
                _id_block(),
                _cz_block(_ZZ_ERR, ideal=CZ),
                _id_block(),
                _pulse_block(_IZ),
            ),
            generator_basis=None,
            readout_rot=construct_readout_rotation(LEVELS),
            residual_ops=_Z_RESIDUAL,
            pauli_labels=_ZZ_LABELS,
            fixed=_Z_FIXED,
            lower=_LOWER,
            upper=_UPPER,
            init=_INIT_SET1,
        ),
        DbSet(
            name="db_set2_qubit_pairq3-6",
            blocks=(
                _id_block(),
                _cz_block(_YY_ERR),
                _id_block(),
                _pulse_block(_YI),
                _id_block(),
                _cz_block(_YY_ERR),
                _id_block(),
                _pulse_block(_IY),
            ),
            generator_basis=None,
            readout_rot=None,
            residual_ops=_Z_RESIDUAL,
            pauli_labels={
                "eta": "YY",
                "eps": "YI",
                "kap": "IY",
                "z1": "ZI",
                "z2": "IZ",
                "z12": "ZZ",
            },
            fixed={},
            lower=_LOWER,
            upper=_UPPER,
            init=_INIT,
        ),
        DbSet(
            name="db_set3_qubit_pairq3-6",
            blocks=(
                _id_block(),
                _cz_block(_XX_ERR),
                _id_block(),
                _pulse_block(_XI),
                _id_block(),
                _cz_block(_XX_ERR),
                _id_block(),
                _pulse_block(_IX),
            ),
            generator_basis=None,
            readout_rot=None,
            residual_ops=_Z_RESIDUAL,
            pauli_labels={
                "eta": "XX",
                "eps": "XI",
                "kap": "IX",
                "z1": "ZI",
                "z2": "IZ",
                "z12": "ZZ",
            },
            fixed={},
            lower=_LOWER,
            upper=_UPPER,
            init=_INIT,
        ),
        DbSet(
            name="db_set4_qubit_pairq3-6",
            blocks=(
                _id_block(),
                _cz_block(_SET4_ERR),
                _id_block(),
                _pulse_block(_ZX),
                _id_block(),
                _cz_block(_SET4_ERR),
                _id_block(),
                _pulse_block(_XY),
            ),
            generator_basis=None,
            readout_rot=None,
            residual_ops=_Z_RESIDUAL,
            pauli_labels={
                "eta": "ZX",
                "eps": "XY",
                "kap": "YZ",
                "z1": "ZI",
                "z2": "IZ",
                "z12": "ZZ",
            },
            fixed=_Z_FIXED,
            lower=_LOWER,
            upper=_UPPER,
            init=_INIT,
        ),
        DbSet(
            name="db_set5_qubit_pairq3-6",
            blocks=(
                _id_block(),
                _cz_block(_SET5_ERR),
                _id_block(),
                _pulse_block(_XZ),
                _id_block(),
                _cz_block(_SET5_ERR),
                _id_block(),
                _pulse_block(_YX),
            ),
            generator_basis=None,
            readout_rot=None,
            residual_ops=_Z_RESIDUAL,
            pauli_labels={
                "eta": "XZ",
                "eps": "YX",
                "kap": "ZY",
                "z1": "ZI",
                "z2": "IZ",
                "z12": "ZZ",
            },
            fixed=_Z_FIXED,
            lower=_LOWER,
            upper=_UPPER,
            init=_INIT,
        ),
        # Simulated data for the error-bar scripts: lab-frame ZZ/ZI/IZ, no pulses, Z
        # readout. "mix" only; its rates are per step rather than 1/us.
        DbSet(
            name="synthetic",
            blocks=None,
            generator_basis=_GENERATOR_BASIS,
            readout_rot=None,
            residual_ops=_Z_RESIDUAL,
            pauli_labels=_ZZ_LABELS,
            fixed=_Z_FIXED,
            lower=_LOWER,
            upper=_UPPER,
            init=_INIT,
        ),
    )
}


############
# Outcome probabilities
############
def construct_unit_op(entry: DbSet, params: dict) -> np.ndarray:
    """Superoperator of one full repetition of `entry`'s sequence.

    Pass a `dataclasses.replace` of an entry to play different pulses (fim_check.py).
    """
    sequence = entry.compiled
    dissipator = _decay_super(params["d1"], params["d2"], params["r1"], params["r2"])
    # Every block dwells for one of two durations, and every CZ carries the same
    # coherent error, so three propagators cover the whole repetition.
    decay = {
        dwell: evolve(dissipator, dwell, "dissipator") for dwell in sequence.dwells
    }
    error = evolve(
        params["eta"] * sequence.err_supers[0]
        + params["eps"] * sequence.err_supers[1]
        + params["kap"] * sequence.err_supers[2],
        1,
        "hamiltonian",
    )
    residual_gen = (
        params["z1"] * sequence.residual_supers[0]
        + params["z2"] * sequence.residual_supers[1]
        + params["z12"] * sequence.residual_supers[2]
        if sequence.residual_supers
        else None
    )
    frame = np.eye(DIM**2, dtype=complex)
    unit = np.eye(DIM**2, dtype=complex)
    for pulse, is_cz, dwell in sequence.steps * 2:
        block = error @ pulse if is_cz else pulse
        frame = pulse @ frame
        step = decay[dwell] @ block
        if residual_gen is not None:
            step = (
                evolve(frame @ residual_gen @ frame.conj().T, 1, "hamiltonian") @ step
            )
        unit = step @ unit
    return unit


def probabilities(
    entry: DbSet, params: dict, n, model: str | None = None
) -> np.ndarray:
    """P(00), P(01), P(10), P(11) after each of `n` repetitions, shape (len(n), 4).

    n may be an int (meaning 0..n-1) or an array. `model` overrides MODEL.
    """
    if isinstance(n, (int, np.integer)):
        n = np.arange(n)
    n = np.asarray(n, dtype=float)
    model = MODEL if model is None else model
    rot = entry.readout_rot
    if model == "mix":
        # exp(G t) s0 = V diag(exp(w t)) V^-1 s0, so one eigendecomposition of the
        # (16, 16) generator gives *every* time step: fold V^-1 s0 and msmt_ops @ V
        # into a single (16, 4) weight matrix, and the whole trajectory is one
        # (len(n), 16) @ (16, 4) product.
        coefficients = np.array([params[k] for k in _GENERATOR_NAMES], dtype=complex)
        basis = entry.mix_basis
        generator = (basis.reshape(len(_GENERATOR_NAMES), -1).T @ coefficients).reshape(
            DIM**2, DIM**2
        )
        state = construct_init_state(rot).astype(complex)
        msmt_ops = construct_msmt_op(
            params["ep1"], params["em1"], params["ep2"], params["em2"], rot
        )
        eigenvalues, eigenvectors = np.linalg.eig(generator)
        weights = (msmt_ops @ eigenvectors) * np.linalg.solve(eigenvectors, state)
        return np.real(np.exp(np.multiply.outer(4 * n, eigenvalues)) @ weights.T)
    if model == "model_dd":
        # `unit_op` is the propagator of one repetition, so n repetitions is the
        # matrix *power* unit_op**n = V diag(w**n) V^-1.
        unit_op = construct_unit_op(entry, params)
        state = construct_init_state(rot, LEVELS).astype(complex)
        msmt_ops = construct_msmt_op(
            params["ep1"],
            params["em1"],
            params["ep2"],
            params["em2"],
            rot=rot,
            levels=LEVELS,
        )
        eigenvalues, eigenvectors = np.linalg.eig(unit_op)
        weights = (msmt_ops @ eigenvectors) * np.linalg.solve(eigenvectors, state)
        return np.real((eigenvalues ** n[:, None]) @ weights.T)
    raise ValueError(f"unknown model {model!r}; expected 'mix' or 'model_dd'")
