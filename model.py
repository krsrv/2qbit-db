"""Two-qubit DB model.

PARAMS, PARAM_NAMES, PHASE_NAMES: the fit parameters, in array order.
DB_SETS: one DbSet per experiment.
MODEL: "mix" (toggling-frame average) or "model_dd" (gate by gate).
probabilities(entry, params, n): outcome probabilities; `params` is a name->value dict.
construct_unit_op(entry, params): superoperator of one repetition.

eta, eps, kap: the set's error triple; d, r: dephasing, relaxation; ep, em: readout
confusion; z1, z2, z12: un-refocused ZI, IZ, ZZ.
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
# Coherent-error coefficients.
PHASE_NAMES = [p.name for p in PARAMS if p.group in ("phase", "residual")]
# Order of the generator basis: every non-SPAM parameter.
_GENERATOR_NAMES = [p.name for p in PARAMS if p.group != "spam"]
_SLOTS = {
    group: [_GENERATOR_NAMES.index(p.name) for p in PARAMS if p.group == group]
    for group in ("phase", "decay", "residual")
}

# Floor on probabilities used as fit weights.
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
    """Superoperator of rho -> u rho u^dag (keeps the conjugate on Y pulses)."""
    return np.kron(u, u.conj())


############
# Level structure
############
QUBIT_LEVELS = (2, 2)
# Levels kept on (control, target)
LEVELS = QUBIT_LEVELS
DIM = np.prod(LEVELS)

# Readout reports a leakage level as this computational level.
LEAK_READOUT_LEVEL = 1


def _number(d: int) -> np.ndarray:
    """diag(0, 1, ..., d-1); dephases 0-1 at the same rate as 0.5 * Z."""
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
    """CZ on the computational block, identity on leakage levels."""
    la, lb = levels
    diag = np.ones(la * lb)
    diag[lb + 1] = -1.0  # |11>
    # diag[lb - 1] = -1.0  # |02>
    return np.diag(diag)


def construct_readout_rotation(levels: tuple[int, int] = QUBIT_LEVELS) -> np.ndarray:
    """X-basis readout: a Hadamard on each qubit's computational levels."""
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
    """The four ideal outcome operators in the basis `rot` (None: Z basis)."""
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
    """Outcome operators with the 4x4 readout confusion matrix folded in."""
    a = np.array([[1 - ep1, em1], [ep1, 1 - em1]])
    b = np.array([[1 - ep2, em2], [ep2, 1 - em2]])
    confusion = (a[:, None, :, None] * b[None, :, None, :]).reshape(4, 4)
    return confusion @ construct_ideal_msmt_ops(rot, levels)


# Lab-frame generator basis of the "synthetic" entry, in _GENERATOR_NAMES order.
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
# Above this cond(V), `evolve` falls back to expm.
_MAX_EIGENBASIS_COND = 1e8


@lru_cache(maxsize=8)
def _eigendecompose(
    key: bytes, dim: int, kind: str
) -> tuple[np.ndarray, np.ndarray, str]:
    """Cached eigendecomposition of the matrix whose bytes are `key`.

    Returns (eigenvalues, eigenvectors, mode) with mode "unitary", "solve" or "expm"
    (too ill-conditioned; arrays None). `kind`: "hamiltonian", "dissipator", "unknown".
    """
    L = np.frombuffer(key, dtype=complex).reshape(dim, dim)
    # eigh for (anti-)Hermitian matrices.
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
        # 1-norm cond: an LU rather than an SVD.
        if np.linalg.cond(eigenvectors, 1) > _MAX_EIGENBASIS_COND:
            return None, None, "expm"
        mode = "solve"
    eigenvalues.flags.writeable = False
    eigenvectors.flags.writeable = False
    return eigenvalues, eigenvectors, mode


def evolve(L: np.ndarray, t: float, kind: str = "unknown"):
    """Propagator exp(L t) of the constant Liouvillian `L`."""
    # exp(L t) = V diag(exp(w t)) V^-1, solving against V rather than inverting it.
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
# Dissipators scaled by (d1, d2, r1, r2).
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
    return (ideal, err_ops, TQ_GT, "op")


def _id_block():
    return (np.eye(4), None, TQ_ID, "idle")


def _pulse_block(u):
    return (u, None, SQ_GT, "dd")


############
# db_sets
############
class _Compiled(NamedTuple):
    """A sequence with its parameter-independent matrices precomputed."""

    steps: tuple  # (pulse superoperator, is_cz, dwell_ns)
    err_supers: tuple  # the (eta, eps, kap) Hamiltonian superoperators
    dwells: tuple  # the distinct dwell times, of which there are only two
    residual_supers: tuple  # the (z1, z2, z12) Hamiltonian superoperators, or ()


@dataclass(frozen=True, eq=False)
class DbSet:
    """One experiment.

    blocks: half a repetition in time order, as (pulse, err_ops, dwell_ns, kind);
        err_ops is a CZ block's (eta, eps, kap) Pauli triple, None for an error-free
        pulse; kind is "op" (CZ), "idle" or "dd".
    generator_basis: explicit "mix" basis; None derives it from `blocks`.
    readout_rot: pre-measurement rotation, None for the Z basis.
    residual_ops: the (z1, z2, z12) operators.
    pauli_labels: the Pauli term each phase parameter multiplies.
    fixed, lower, upper: fixed values and fit bounds.
    init: first starting point, ("uniform", low, high) or ("value", v) per parameter.

    Hashed by identity, so a `dataclasses.replace` copy gets its own caches.
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
        err_ops = next(ops for _, ops, _, _ in self.blocks if ops is not None)
        return _Compiled(
            steps=tuple(
                (_super(pulse), ops is not None, dwell)
                for pulse, ops, dwell, _ in self.blocks
            ),
            err_supers=tuple(_hamiltonian_super(op) for op in err_ops),
            dwells=tuple({dwell for _, _, dwell, _ in self.blocks}),
            residual_supers=(
                ()
                if self.residual_ops is None
                else tuple(_hamiltonian_super(op) for op in self.residual_ops)
            ),
        )

    @cached_property
    def mix_basis(self) -> np.ndarray:
        """`generator_basis`, or the sequence's first-order toggling-frame average

            G_eff = (1 / 4) sum_j W_j^dag (L_err_j + dwell_j L_diss) W_j,

        with W_j the pulses played through block j.
        """
        if self.generator_basis is not None:
            return self.generator_basis
        # Both halves: the first leaves the frame rotated.
        basis = np.zeros((len(_GENERATOR_NAMES), DIM**2, DIM**2), dtype=complex)
        frame = np.eye(DIM**2, dtype=complex)
        for pulse, err_ops, dwell, _ in self.blocks * 2:
            frame = _super(pulse) @ frame
            if err_ops is not None:
                for slot, op in zip(_SLOTS["phase"], err_ops):
                    basis[slot] += frame.conj().T @ _hamiltonian_super(op) @ frame
            for k, slot in enumerate(_SLOTS["decay"]):
                basis[slot] += 1e-3 * dwell * (frame.conj().T @ DECAY_BASIS[k] @ frame)
            if self.residual_ops is not None:
                for slot, op in zip(_SLOTS["residual"], self.residual_ops):
                    basis[slot] += _hamiltonian_super(op)
        # Four gates per repetition.
        basis /= 4
        basis.flags.writeable = False
        return basis


def _decay_init(t2: list, t1: list) -> dict:
    """Initial (d1, d2, r1, r2) in 1/us from T2, T1 in ns: 1000/T_phi and 1000/T1."""
    values = [
        1000 * (1 / t2[0] - 1 / (2 * t1[0])),
        1000 * (1 / t2[1] - 1 / (2 * t1[1])),
        1000 / t1[0],
        1000 / t1[1],
    ]
    return {name: ("value", v) for name, v in zip(("d1", "d2", "r1", "r2"), values)}


# Shared by the entries below.
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
_ZZ_LABELS = {
    "eta": "ZZ",
    "eps": "ZI",
    "kap": "IZ",
    "z1": "ZI",
    "z2": "IZ",
    "z12": "ZZ",
}

# Each experiment set is one half repetition of a DD sequence.
#   set1 -> ZZ, ZI, IZ      set2 -> YY, YI, IY      set3 -> XX, XI, IX
#   set4 -> ZX, XY, YZ      set5 -> XZ, YX, ZY
DB_SETS = {
    entry.name: entry
    for entry in (
        # The plain CZ node.
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
                _cz_block(_YY_ERR, ideal=CZ),
                _id_block(),
                _pulse_block(_YI),
                _id_block(),
                _cz_block(_YY_ERR, ideal=CZ),
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
            fixed=_Z_FIXED,
            lower=_LOWER,
            upper=_UPPER,
            init=_INIT,
        ),
        DbSet(
            name="db_set3_qubit_pairq3-6",
            blocks=(
                _id_block(),
                _cz_block(_XX_ERR, ideal=CZ),
                _id_block(),
                _pulse_block(_XI),
                _id_block(),
                _cz_block(_XX_ERR, ideal=CZ),
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
            fixed=_Z_FIXED,
            lower=_LOWER,
            upper=_UPPER,
            init=_INIT,
        ),
        DbSet(
            name="db_set4_qubit_pairq3-6",
            blocks=(
                _id_block(),
                _cz_block(_SET4_ERR, ideal=CZ),
                _id_block(),
                _pulse_block(_ZX),
                _id_block(),
                _cz_block(_SET4_ERR, ideal=CZ),
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
                _cz_block(_SET5_ERR, ideal=CZ),
                _id_block(),
                _pulse_block(_XZ),
                _id_block(),
                _cz_block(_SET5_ERR, ideal=CZ),
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
        # For the error-bar scripts: lab-frame ZZ/ZI/IZ, Z readout, "mix" only.
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
    """Superoperator of one repetition of `entry`'s sequence ("model_dd")."""
    sequence = entry.compiled
    dissipator = _decay_super(params["d1"], params["d2"], params["r1"], params["r2"])
    # Two dwell times and one CZ error: three propagators cover the repetition.
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

    `n` is an int (0..n-1) or an array; `model` overrides MODEL.
    """
    if isinstance(n, (int, np.integer)):
        n = np.arange(n)
    n = np.asarray(n, dtype=float)
    model = MODEL if model is None else model
    rot = entry.readout_rot
    if model == "mix":
        # One eig of the generator gives every time step.
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
        # unit_op ** n via its eigendecomposition.
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
