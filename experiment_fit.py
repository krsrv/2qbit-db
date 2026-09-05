import itertools
import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from scipy.linalg import expm
from scipy.optimize import least_squares

from get_error_bars import PHASE_NAMES
from least_squares import (
    _GENERATOR_BASIS,
    MIN_WEIGHT_PROB,
    PARAM_NAMES,
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
    H,
    _hamiltonian_super,
    _on,
    construct_cz,
    construct_decay_basis,
    construct_full_params,
    construct_init_state,
    construct_msmt_op,
    construct_readout_rotation,
    get_fidelities,
    residuals,
)

############
# Constants
############

HERE = Path().cwd()

JOINT_STATES = ("00", "01", "10", "11")
SHOT_DIM_CANDIDATES = ("shot", "n", "N")
_SET_VAR_RE = re.compile(r"^state_(control|target)_s(\d+)_(\d+)$")


############
# Daria's code
############
def discover_set_indices(ds: xr.Dataset) -> list[int]:
    sets = set()
    for name in ds.data_vars:
        m = _SET_VAR_RE.match(name)
        if m:
            sets.add(int(m.group(2)))
            continue
        m2 = re.match(r"^state_(?:control|target)_s(\d+)_$", name)
        if m2:
            sets.add(int(m2.group(1)))
    return sorted(sets) if sets else list(range(1, 6))


def concat_pair_streams(
    ds: xr.Dataset, role: str, set_idx: int, pair_names: np.ndarray
):
    """Build state_{role} for one experiment set with a qubit_pair dimension."""
    stacked = f"state_{role}_s{set_idx}_"
    if stacked in ds.data_vars:
        return ds[stacked]

    pair_vars = []
    for name in ds.data_vars:
        m = _SET_VAR_RE.match(name)
        if m and m.group(1) == role and int(m.group(2)) == set_idx:
            pair_vars.append((int(m.group(3)), name))
    if not pair_vars:
        alt = f"state_{role}_s{set_idx}"
        return ds[alt] if alt in ds.data_vars else None

    pair_vars = sorted(pair_vars, key=lambda x: x[0])
    state_list = [ds[name] for _, name in pair_vars]
    names = pair_names[: len(state_list)]
    return xr.concat(state_list, dim="qubit_pair").assign_coords(
        qubit_pair=("qubit_pair", names)
    )


def probs_from_shots(
    state_c: xr.DataArray, state_t: xr.DataArray
) -> dict[str, xr.DataArray]:
    """Mean and SEM of joint-state indicators over the shot dimension."""
    shot_dim = next((d for d in SHOT_DIM_CANDIDATES if d in state_c.dims), None)
    out = {}
    for ss in JOINT_STATES:
        s_c, s_t = int(ss[0]), int(ss[1])
        indicator = ((state_c == s_c) & (state_t == s_t)).astype(float)
        if shot_dim is None:
            out[f"P_{ss}"] = indicator
            out[f"P_{ss}_err"] = xr.full_like(indicator, np.nan)
        else:
            n_shots = max(int(state_c.sizes[shot_dim]), 1)
            out[f"P_{ss}"] = indicator.mean(dim=shot_dim)
            out[f"P_{ss}_err"] = indicator.std(dim=shot_dim, ddof=1) / np.sqrt(n_shots)
    return out


def process_ds_raw(ds: xr.Dataset) -> xr.Dataset:
    """Recompute P_ss / P_ss_err from per-shot state streams (ignore any stored P_*)."""
    if "qubit_pair" in ds.coords:
        pair_names = np.asarray(ds.qubit_pair.values)
    else:
        # Infer pair count from stream suffixes *_1, *_2, …
        idxs = sorted(
            {int(m.group(3)) for name in ds.data_vars if (m := _SET_VAR_RE.match(name))}
        )
        pair_names = np.array([f"pair_{i}" for i in idxs]) or np.array(["pair_1"])

    set_indices = discover_set_indices(ds)
    state_c_sets, state_t_sets, used = [], [], []
    for k in set_indices:
        sc = concat_pair_streams(ds, "control", k, pair_names)
        st = concat_pair_streams(ds, "target", k, pair_names)
        if sc is None or st is None:
            print(f"warning: missing streams for set {k}, skipping")
            continue
        # Ensure qubit_pair dim exists for single-pair streams (shot, n_ops)
        if "qubit_pair" not in sc.dims:
            sc = sc.expand_dims(qubit_pair=[str(pair_names[0])])
            st = st.expand_dims(qubit_pair=[str(pair_names[0])])
        state_c_sets.append(sc)
        state_t_sets.append(st)
        used.append(k)

    if not used:
        raise RuntimeError("No state_control_s* / state_target_s* streams found")

    state_c = xr.concat(state_c_sets, dim="db_set").assign_coords(db_set=used)
    state_t = xr.concat(state_t_sets, dim="db_set").assign_coords(db_set=used)
    probs = probs_from_shots(state_c, state_t)
    return xr.Dataset({"state_control": state_c, "state_target": state_t, **probs})


############
# Analysis functions
############
# eta, eps, kap | d1, d2 | r1, r2 | ep1, em1, ep2, em2 | phi
# Params: Coherent errors, sigma_z decay rate, sigma_- decay rate, readout + prep error
# rates, leakage coupling.
LOWER_BOUNDS = np.array([-np.pi] * 3 + [0.0] * 4 + [0.0] * 4 + [-0.3])
UPPER_BOUNDS = np.array([np.pi] * 3 + [1.0] * 4 + [0.3] * 4 + [0.3])

N_RESTARTS = 20
GLS_PASSES = 2
GLS_TOL = 1e-9

# Warm-chained fits use a growing prefix of the time steps, as get_error_bars.py does.
MIN_REPETITIONS = 10
REPETITION_STEP = 5

OUTPUT_DIR = HERE / "output"

# Number of points to plot using fitted formula
PLOT_POINTS = 400

DATA_COLUMNS = [f"P_{ss}" for ss in JOINT_STATES]
ERR_COLUMNS = [f"P_{ss}_err" for ss in JOINT_STATES]


class Family(NamedTuple):
    """One independent experiment: a (len(n), 4) probability table and its noise scale."""

    label: str
    coords: dict
    n: np.ndarray
    data: np.ndarray
    errs: np.ndarray
    shots: int


def infer_shots(ds: xr.Dataset, default: int = 1000) -> int:
    """Shots per time step, read off the per-shot dimension of the state streams."""
    for name in ds.data_vars:
        if not str(name).startswith("state_"):
            continue
        shot_dim = next((d for d in SHOT_DIM_CANDIDATES if d in ds[name].dims), None)
        if shot_dim is not None:
            return int(ds.sizes[shot_dim])
    print(f"warning: no per-shot dimension found, assuming shots={default}")
    return default


def prepare_dataset(ds_raw: xr.Dataset) -> xr.Dataset:
    """Recompute P_ss from per-shot streams where they exist, else keep the stored ones.

    `process_ds_raw` expects the general-protocol layout (state_control_s{set}_{pair}).
    The plain DB node stores a single un-suffixed pair of streams instead, which it
    rejects; that dataset already carries P_ss / P_ss_err, so fall through to those.
    """
    try:
        return process_ds_raw(ds_raw)
    except RuntimeError as exc:
        missing = [c for c in DATA_COLUMNS if c not in ds_raw.data_vars]
        if missing:
            raise RuntimeError(
                f"{exc}; and no stored {missing} to fall back on"
            ) from exc
        print(f"note: {exc}; using the P_ss stored in the file instead")
        return ds_raw


def iter_families(ds: xr.Dataset) -> list[Family]:
    """Split `ds` into one Family per (db_set, qubit_pair, ...) combination.

    Every dimension of P_00 other than the time axis indexes an independent experiment,
    so the families are the points of their cross product.
    """
    shots = infer_shots(ds)
    n = np.asarray(ds["number_of_operations"].values, dtype=float)
    family_dims = [d for d in ds[DATA_COLUMNS[0]].dims if d != "number_of_operations"]
    grids = [ds[DATA_COLUMNS[0]][d].values for d in family_dims]

    families = []
    for point in itertools.product(*grids) if family_dims else [()]:
        coords = dict(zip(family_dims, point))
        selected = ds.sel(coords)
        data = np.stack([selected[c].values for c in DATA_COLUMNS], axis=-1)
        if all(c in selected.data_vars for c in ERR_COLUMNS):
            errs = np.stack([selected[c].values for c in ERR_COLUMNS], axis=-1)
        else:
            errs = np.full_like(data, np.nan)
        label = "_".join(f"{d}{v}" for d, v in coords.items()) or "all"
        families.append(Family(label, coords, n, data, errs, shots))
    return families


def _decay_init(t2: list, t1: list) -> list:
    """Initial (d1, d2, r1, r2) values, in 1/us.

    Both methods integrate the dissipator over the real gate durations in ns, so d
    and r are absolute rates rather than per-repetition fractions:
    d = 1000 / T_phi and r = 1000 / T1, where 1 / T_phi = 1 / T2 - 1 / (2 T1).
    """
    return [
        1000 * (1 / t2[0] - 1 / (2 * t1[0])),
        1000 * (1 / t2[1] - 1 / (2 * t1[1])),
        1000 / t1[0],
        1000 / t1[1],
    ]


def construct_init_values(
    family: Family, rng: np.random.Generator, fixed_params: dict
) -> np.ndarray:
    """
    Generate an initial guess for the fit parameters for a given experiment family.

    Args:
        family (Family): The experimental Family grouping currently being fit.
        rng: np.random.Generator
        fixed_params (dict): Any parameter names and values that will be held fixed during optimization.

    Returns:
        np.ndarray: Array of initial parameter values (with fixed-value entries removed).
    """
    # eta, eps, kap | d1, d2, r1, r2 | ep1, em1, ep2, em2 | phi
    t2, t1 = [11000, 21000], [10000, 30000]
    discard_idx = [i for i, name in enumerate(PARAM_NAMES) if name in fixed_params]
    if "set1" in family.label:
        zi, iz = (0.9725 - 1) * 2 * np.pi, 0.2330 * 2 * np.pi
        params = np.concatenate(
            [
                rng.uniform(0, 0.01, size=1),
                rng.uniform(0, 0.1, size=2),
                # zi,
                # iz,
                _decay_init(t2, t1),
            ]
        )

    elif (
        ("set2" in family.label)
        or ("set3" in family.label)
        or ("set4" in family.label)
        or ("set5" in family.label)
        or (family.label == "qubit_pairq3-6")
    ):
        params = np.concatenate(
            [
                rng.uniform(0, 0.01, size=3),
                _decay_init(t2, t1),
            ]
        )
    elif "ibm" in family.label:
        t2, t1 = [100000, 100000], [100000, 100000]
        params = np.concatenate(
            [
                rng.uniform(0, 0.02, size=3),
                _decay_init(t2, t1),
            ]
        )
    params = np.concatenate(
        [
            params,
            rng.uniform(0, 0.1, size=4),
            rng.uniform(-PHI_INIT_SCALE, PHI_INIT_SCALE, size=1),
        ]
    )
    if discard_idx:
        params = np.delete(params, discard_idx)
    return params


def get_decay_timescale(d1, d2, r1, r2, label: str) -> np.ndarray:
    """(T1, T2) in ns from the fitted rates, which are in 1/us for both methods."""
    t1 = 1 / np.array([r1, r2])
    t2 = 1 / np.array([r1 / 2 + d1, r2 / 2 + d2])
    return 1e3 * t1, 1e3 * t2


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
    # V^-1. Same trick as `get_fidelities`.
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


# Levels kept on (control, target)
LEVELS = (2, 2)
DIM = np.prod(LEVELS)

CZ = construct_cz(LEVELS)
# vec form of rho -> CZ rho CZ^dag, i.e. kron(CZ, CZ.conj()).
CZ_SUPER = np.kron(CZ, CZ.conj())
# X-basis readout;
READOUT_ROT = construct_readout_rotation(LEVELS)
# Dissipators scaled by (d1, d2, r1, r2), in the units `_decay_init` produces.
DECAY_BASIS = construct_decay_basis(LEVELS)
SQ_GT, TQ_GT = 30, 60

# Number operators of the two subsystems, in units of the computational splitting
# E2's third entry carries the anharmonicity: |2> sits at 2 - 0.05.
E1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
E2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0 - 0.05]], dtype=complex)
# INT = np.zeros((DIM, DIM), dtype=complex)
# INT[4, 2] = INT[2, 4] = 1.0

PHI_INIT_SCALE = 0.02


# DD pulses, as 4x4 unitaries. `_super` turns one into the superoperator
# rho -> U rho U^dag, i.e. kron(U, U.conj()); building them that way (rather than
# kron(U, U) by hand) keeps the conjugate on the imaginary Pauli-Y pulses, whose
# sign would otherwise only cancel by accident because each appears an even number
# of times per half-repetition.
def _super(u: np.ndarray) -> np.ndarray:
    """Superoperator of rho -> u rho u^dag."""
    return np.kron(u, u.conj())


_XI = np.kron(SIGMA_X, np.eye(2))
_IX = np.kron(np.eye(2), SIGMA_X)
_YI = np.kron(SIGMA_Y, np.eye(2))
_IY = np.kron(np.eye(2), SIGMA_Y)
_XY = _XI @ _IY
_YX = _YI @ _IX
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


class Sequence(NamedTuple):
    """One half-repetition of a DD sequence plus the basis it is read out in."""

    blocks: tuple  # (pulse 4x4, err_ops or None, dwell_ns)
    rot: np.ndarray | None  # pre-measurement rotation, None for the Z basis


def _cz_block(err_ops, ideal=_II):
    return (ideal, err_ops, TQ_GT)


def _pulse_block(u):
    return (u, None, SQ_GT)


############
# The five DD sequences
############
# Each experiment set is one half repetition of a DD sequence, written here as the
# blocks it plays in time order (first played first):
#
#   pulse     -- the ideal gate of the block, as a 4x4 unitary.
#   err_ops   -- the (eta, eps, kap) Pauli triple the block's coherent error is
#                written in, or None for a single-qubit pulse (taken to be error free).
#   dwell_ns  -- how long the block occupies the qubits, and so how long the
#                dissipator acts for.
#   set1 -> ZZ, ZI, IZ      set2 -> YY, YI, IY      set3 -> XX, XI, IX
#   set4 -> ZX, XY, YZ      set5 -> XZ, YX, ZY
SEQUENCES = {
    # H CZ H  X  H CZ H  X, with the ideal CZ kept (it commutes with the error).
    "set1": Sequence(
        (
            _pulse_block(_HH),
            _cz_block(_ZZ_ERR, ideal=CZ),
            _pulse_block(_HH),
            _pulse_block(_XI),
            _pulse_block(_HH),
            _cz_block(_ZZ_ERR, ideal=CZ),
            _pulse_block(_HH),
            _pulse_block(_IX),
        ),
        None,
    ),
    "set2": Sequence(
        (
            _cz_block(_YY_ERR),
            _pulse_block(_YI),
            _cz_block(_YY_ERR),
            _pulse_block(_IY),
        ),
        None,
    ),
    "set3": Sequence(
        (
            _cz_block(_XX_ERR),
            _pulse_block(_XI),
            _cz_block(_XX_ERR),
            _pulse_block(_IX),
        ),
        None,
    ),
    "set4": Sequence(
        (
            _pulse_block(_IH),
            _cz_block(_SET4_ERR),
            _pulse_block(_IH),
            _pulse_block(_YX),
            _pulse_block(_IH),
            _cz_block(_SET4_ERR),
            _pulse_block(_IH),
            _pulse_block(_XY),
        ),
        construct_readout_rotation(LEVELS),
    ),
    "set5": Sequence(
        (
            _pulse_block(_HI),
            _cz_block(_SET5_ERR),
            _pulse_block(_HI),
            _pulse_block(_YX),
            _pulse_block(_HI),
            _cz_block(_SET5_ERR),
            _pulse_block(_HI),
            _pulse_block(_XY),
        ),
        construct_readout_rotation(LEVELS),
    ),
}


def sequence_for(label: str) -> Sequence:
    """The DD sequence the family `label` was taken with."""
    for key, sequence in SEQUENCES.items():
        if key in label:
            return sequence
    # The plain (no-DD) node and the IBM CSVs both run set1's ZZ/ZI/IZ experiment.
    if label == "qubit_pairq3-6" or "ibm" in label:
        return SEQUENCES["set1"]
    raise RuntimeError(f"Unrecognized family label: {label}")


def _decay_super(d1, d2, r1, r2) -> np.ndarray:
    """The dissipator at rates (d1, d2, r1, r2), which are given in 1/us."""
    return 1e-3 * (
        d1 * DECAY_BASIS[0]
        + d2 * DECAY_BASIS[1]
        + r1 * DECAY_BASIS[2]
        + r2 * DECAY_BASIS[3]
    )


class _Compiled(NamedTuple):
    """A `Sequence` with everything independent of the fit parameters worked out.

    `construct_unit_op` runs on every residual and every finite-difference column of
    every restart, so it should not be rebuilding the same fixed matrices each time.
    """

    steps: tuple  # (pulse superoperator, is_cz, dwell_ns)
    err_supers: tuple  # the (eta, eps, kap) Hamiltonian superoperators
    dwells: tuple  # the distinct dwell times, of which there are only two


@lru_cache(maxsize=None)
def _compiled(label: str) -> _Compiled:
    blocks = sequence_for(label).blocks
    err_ops = next(ops for _, ops, _ in blocks if ops is not None)
    return _Compiled(
        steps=tuple(
            (_super(pulse), ops is not None, dwell) for pulse, ops, dwell in blocks
        ),
        err_supers=tuple(_hamiltonian_super(op) for op in err_ops),
        dwells=tuple({dwell for _, _, dwell in blocks}),
    )


def construct_unit_op(label: str, eta, eps, kap, d1, d2, r1, r2) -> np.ndarray:
    """Superoperator of one full repetition of `label`'s sequence."""
    sequence = _compiled(label)
    dissipator = _decay_super(d1, d2, r1, r2)
    # Every block dwells for one of two durations, and every CZ carries the same
    # coherent error, so three propagators cover the whole repetition.
    decay = {
        dwell: evolve(dissipator, dwell, "dissipator") for dwell in sequence.dwells
    }
    error = evolve(
        eta * sequence.err_supers[0]
        + eps * sequence.err_supers[1]
        + kap * sequence.err_supers[2],
        1,
        "hamiltonian",
    )
    half = np.eye(DIM**2, dtype=complex)
    for pulse, is_cz, dwell in sequence.steps:
        block = error @ pulse if is_cz else pulse
        half = (decay[dwell] @ block) @ half
    return half @ half


@lru_cache(maxsize=None)
def effective_generator_basis(label: str) -> np.ndarray:
    """`label`'s sequence collapsed to the single generator the "mix" path uses.

    The "mix" model has no DD pulses: it exponentiates one 16x16 generator for
    `4 * n`, four gates per repetition. We want the generator in the "mix" path
    to be an approximation of `construct_unit_op`, the simplest of which is the
    first-order Magnus (toggling-frame) average.
    Each block's contribution is conjugated by the pulses played before it,
    weighted by how long the block lasts:

        G_eff = (1 / 4) sum_j W_j^dag (L_err_j + dwell_j L_diss) W_j,

    where W_j is the cumulative pulse superoperator through block j. For the coherent
    part it just reproduces the surviving Pauli triple in the frame the pulses leave
    it in.
    The dissipative part needs to be derived. For instance, relaxation operators
    will be symmetrized to some extent due to this approach.
    """
    # The whole repetition, not one half doubled: the pulses of the first half do
    # not compose back to the identity, so the second half runs in a frame the first
    # half left rotated. For set2 that is the difference between relaxation coming
    # out 1/3 sigma^- and 2/3 sigma^+ and its true even split.
    blocks = sequence_for(label).blocks * 2
    basis = np.zeros((7, DIM**2, DIM**2), dtype=complex)
    frame = np.eye(DIM**2, dtype=complex)
    for pulse, err_ops, dwell in blocks:
        frame = _super(pulse) @ frame
        if err_ops is not None:
            for k, op in enumerate(err_ops):
                basis[k] += frame.conj().T @ _hamiltonian_super(op) @ frame
        for k in range(4):
            basis[3 + k] += 1e-3 * dwell * (frame.conj().T @ DECAY_BASIS[k] @ frame)
    # `get_fidelities` exponentiates for 4 * n, four gates per repetition.
    basis /= 4
    # Every caller shares the cached array.
    basis.flags.writeable = False
    return basis


def generator_basis_for(label: str) -> np.ndarray:
    """The Hamiltonian basis each db_set drives, as a generator basis."""
    return effective_generator_basis(label)


def readout_basis_for(label: str) -> np.ndarray | None:
    """The pre-measurement rotation each db_set reads out in."""
    return sequence_for(label).rot


def new_gate_fidelities(
    n,
    eta,
    eps,
    kap,
    d1,
    d2,
    r1,
    r2,
    ep1,
    em1,
    ep2,
    em2,
    phi,
    generator_basis: np.ndarray = _GENERATOR_BASIS,
    readout_basis: np.ndarray = None,
    label: str | None = None,
):
    """Outcome probabilities from the DD sequence `label` was taken with.

    Builds the propagator of one repetition gate by gate out of `SEQUENCES`.
    `generator_basis` and `readout_basis` are ignored and are accepted
    only so this shares a signature with `get_fidelities`.
    """
    if isinstance(n, (int, np.integer)):
        n = np.arange(n)
    n = np.asarray(n, dtype=float)

    unit_op = construct_unit_op(label, eta, eps, kap, d1, d2, r1, r2)
    rot = readout_basis_for(label)

    state = construct_init_state(rot, LEVELS).astype(complex)
    msmt_ops = construct_msmt_op(ep1, em1, ep2, em2, rot=rot, levels=LEVELS)

    # `unit_op` is the propagator of one repetition, so n repetitions is the matrix
    # *power* unit_op**n = V diag(w**n) V^-1
    eigenvalues, eigenvectors = np.linalg.eig(unit_op)
    weights = (msmt_ops @ eigenvectors) * np.linalg.solve(eigenvectors, state)
    return np.real((eigenvalues ** n[:, None]) @ weights.T)


def gate_residuals(
    x: np.ndarray,
    n: np.ndarray,
    data: np.ndarray,
    shots: int,
    weight_probs: np.ndarray | None = None,
    generator_basis: np.ndarray = _GENERATOR_BASIS,
    readout_basis: np.ndarray = None,
    fixed_params: dict | None = None,
    label: str | None = None,
) -> np.ndarray:
    x_full = construct_full_params(x, fixed_params)
    model_data = new_gate_fidelities(
        n,
        *x_full,
        generator_basis=generator_basis,
        readout_basis=readout_basis,
        label=label,
    ).real

    probs = model_data if weight_probs is None else weight_probs
    diffs = model_data - data
    # Same floor as `least_squares.residuals`, so the two methods' reduced_chi2 are
    # on the same scale and can be compared.
    return (
        np.sqrt(shots) * diffs / np.sqrt(np.clip(probs, MIN_WEIGHT_PROB, None))
    ).reshape(-1)


METHOD = "model_dd"
if METHOD == "model_dd":
    residual_fn = gate_residuals  # residuals
    fidelity_fn = new_gate_fidelities  # get_fidelities
elif METHOD == "mix":
    residual_fn = residuals
    fidelity_fn = get_fidelities


def _run_least_squares(
    x0: np.ndarray,
    n: np.ndarray,
    data: np.ndarray,
    shots: int,
    weight_probs: np.ndarray | None = None,
    generator_basis: np.ndarray = _GENERATOR_BASIS,
    readout_basis: np.ndarray = None,
    fixed_params: dict | None = None,
    lower_bounds: np.ndarray = LOWER_BOUNDS,
    upper_bounds: np.ndarray = UPPER_BOUNDS,
    label: str | None = None,
):
    return least_squares(
        residual_fn,
        x0,
        args=(
            n,
            data,
            shots,
            weight_probs,
            generator_basis,
            readout_basis,
            fixed_params,
            label,
        ),
        bounds=(lower_bounds, upper_bounds),
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )


def _gls_refine(
    result,
    n: np.ndarray,
    data: np.ndarray,
    shots: int,
    generator_basis: np.ndarray = _GENERATOR_BASIS,
    readout_basis: np.ndarray | None = None,
    fixed_params: dict | None = None,
    lower_bounds: np.ndarray = LOWER_BOUNDS,
    upper_bounds: np.ndarray = UPPER_BOUNDS,
    label: str | None = None,
):
    """Iterated GLS: refit with the weights frozen at the current model prediction.

    Holding the weights fixed within a pass keeps the solver from differentiating through them,
    so the fixed point solves the unbiased estimating equation.
    """
    for _ in range(GLS_PASSES):
        x = construct_full_params(result.x, fixed_params)
        weight_probs = fidelity_fn(
            n,
            *x,
            generator_basis=generator_basis,
            readout_basis=readout_basis,
            label=label,
        ).real
        refined = _run_least_squares(
            result.x,
            n,
            data,
            shots,
            weight_probs,
            generator_basis=generator_basis,
            readout_basis=readout_basis,
            fixed_params=fixed_params,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            label=label,
        )
        shift = np.max(np.abs(refined.x - result.x))
        result = refined
        if shift < GLS_TOL:
            break
    return result


def construct_x_trial(
    x0: np.ndarray | None,
    attempt: int,
    rng: np.random.Generator,
    fixed_params: dict | None,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
):
    """
    Construct a trial parameter vector `x0_trial` for optimization, supporting random
    initialization and jittered restarts, while respecting any fixed parameters.

    Args:
        x0 (np.ndarray | None): The initial parameter guess. If None, a random starting point
                                is generated; otherwise, this is the base for (potentially
                                jittered) restarts.
        attempt (int): The index of the current restart.
                       - 0 indicates the primary attempt (no perturbation).
                       - >0 indicates a random restart with jitter.
        rng (np.random.Generator).
        fixed_params (dict | None): Dictionary specifying the names and values of parameters fixed
                                    during this fit. These parameters are omitted from the trial vector.
        lower_bounds (np.ndarray): Lower bounds for each free parameter, used to clip the returned vector.
        upper_bounds (np.ndarray): Upper bounds for each free parameter, used to clip the returned vector.

    Returns:
        np.ndarray: A parameter vector suitable for passing to the solver.
    """
    # eta, eps, kap | d1, d2, r1, r2 | ep1, em1, ep2, em2 | phi
    discard_idx = [i for i, name in enumerate(PARAM_NAMES) if name in fixed_params]
    # d and r are rates in 1/us in both methods, so one set of scales covers both.
    if x0 is None:
        x0_trial = np.concatenate(
            [
                rng.uniform(-0.02, 0.02, size=3),
                rng.uniform(0.0, 1 / 100, size=4),
                rng.uniform(0.0, 0.2, size=4),
                rng.uniform(-PHI_INIT_SCALE, PHI_INIT_SCALE, size=1),
            ]
        )
        if discard_idx:
            x0_trial = np.delete(x0_trial, discard_idx)
    elif attempt == 0:
        x0_trial = np.asarray(x0, dtype=float)
    else:
        perturb = np.concatenate(
            [
                rng.uniform(-0.01, 0.01, size=3),
                rng.uniform(0, 0.01, size=4),
                rng.uniform(0, 0.001, size=4),
                rng.uniform(-PHI_INIT_SCALE, PHI_INIT_SCALE, size=1),
            ]
        )
        if discard_idx:
            perturb = np.delete(perturb, discard_idx)
        x0_trial = np.asarray(x0, dtype=float) + perturb
    return np.clip(x0_trial, lower_bounds, upper_bounds)


def fit_family(
    n: np.ndarray,
    data: np.ndarray,
    shots: int,
    rng: np.random.Generator,
    x0: np.ndarray | None = None,
    n_restarts: int = N_RESTARTS,
    generator_basis: np.ndarray = _GENERATOR_BASIS,
    readout_basis: np.ndarray | None = None,
    fixed_params: dict | None = None,
    lower_bounds: np.ndarray = LOWER_BOUNDS,
    upper_bounds: np.ndarray = UPPER_BOUNDS,
    label: str | None = None,
) -> dict:
    """Multi-start least-squares fit of all four curves at once, on a fixed budget.

    `x0` is the starting point for the first attempt; the remaining `n_restarts`
    attempts are jittered around it. The lowest-cost attempt is then GLS-refined.
    """
    best = None
    for attempt in range(n_restarts + 1):
        x0_trial = construct_x_trial(
            x0, attempt, rng, fixed_params, lower_bounds, upper_bounds
        )
        result = _run_least_squares(
            x0_trial,
            n,
            data,
            shots,
            generator_basis=generator_basis,
            readout_basis=readout_basis,
            fixed_params=fixed_params,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            label=label,
        )
        if best is None or result.cost < best.cost:
            best = result

    best = _gls_refine(
        best,
        n,
        data,
        shots,
        generator_basis=generator_basis,
        readout_basis=readout_basis,
        fixed_params=fixed_params,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        label=label,
    )

    param_names = [name for name in PARAM_NAMES if name not in fixed_params]
    params = dict(zip(param_names, best.x))
    params["true_cost"] = 0.5 * np.sum(
        residual_fn(
            best.x,
            n,
            data,
            shots,
            generator_basis=generator_basis,
            readout_basis=readout_basis,
            fixed_params=fixed_params,
            label=label,
        )
        ** 2
    )
    # 4 residual entries per time step but only 3 independent ones (the rows sum to 1).
    dof = 3 * len(n) - (len(PARAM_NAMES) - len(fixed_params))
    params["rmse"] = float(np.sqrt(2 * params["true_cost"] / dof))
    params["reduced_chi2"] = float(2 * params["true_cost"] / dof)
    params["at_bound"] = int(
        np.sum(
            np.isclose(best.x, lower_bounds, atol=1e-9)
            | np.isclose(best.x, upper_bounds, atol=1e-9)
        )
    )
    params["result"] = best
    return params


def fixed_params_for(label: str) -> dict:
    if "ibm" in label:
        return {"phi": 0.0}
    if label == "qubit_pairq3-6":
        return {"phi": 0.0}
    return {"phi": 0.0}


def bounds_for(label: str) -> tuple[np.ndarray, np.ndarray]:
    fixed = fixed_params_for(label)
    keep_idx = [i for i, name in enumerate(PARAM_NAMES) if name not in fixed]
    lower_bounds = LOWER_BOUNDS[keep_idx]
    upper_bounds = UPPER_BOUNDS[keep_idx]
    if label == "qubit_pairq3-6":
        # free_names = [name for name in PARAM_NAMES if name not in fixed]
        # decay_floors = {"d1": 0.16, "d2": 0.16, "r1": 0.08, "r2": 0.08}
        # for name, floor in decay_floors.items():
        #     if name in free_names:
        #         lower_bounds[free_names.index(name)] = floor

        # decay_caps = {"d1": 1 / 6, "d2": 1 / 6, "r1": 1 / 6, "r2": 1 / 6}
        # for name, cap in decay_caps.items():
        #     if name in free_names:
        #         upper_bounds[free_names.index(name)] = cap
        assert np.all(lower_bounds < upper_bounds)
    return lower_bounds, upper_bounds


def _phase_flip_is_a_symmetry(family: Family, fixed_params: dict, params: dict) -> bool:
    """Whether negating every phase leaves this family's model curve unchanged.

    `canonicalize_signs` picks the eps >= 0 branch of a phases -> -phases degeneracy.
    Which sign flips are degeneracies depends on the set: sets 2 and 3 are blind to
    all of them (a Z-basis prep and readout commute with ZI and IZ, so only the three
    magnitudes are identifiable at all), set1 is blind only to the global flip, and
    sets 4 and 5 are blind to the global flip but *not* to flipping eps or kap on
    their own. Flip one that is not a degeneracy and the row written to the CSV stops
    reproducing the fit it came from -- and `plot_family`, which draws that row, plots
    a curve that misses the data. Rather than tabulate which sets qualify, ask the
    model.

    "Degenerate" here means the two curves differ by less than the standard error of
    a single measured probability, since a difference smaller than that is not what
    the fit resolved: the DD sequences break the global flip in sets 4 and 5 at the
    1e-3 level through terms that do not commute with the dissipator, which is real
    but three times under the noise floor at 5000 shots.
    """
    x = np.array([params[name] for name in PARAM_NAMES if name not in fixed_params])
    flipped = np.array(
        [
            -value if name in PHASE_NAMES else value
            for name, value in zip([n for n in PARAM_NAMES if n not in fixed_params], x)
        ]
    )
    kwargs = dict(
        generator_basis=generator_basis_for(family.label),
        readout_basis=readout_basis_for(family.label),
        label=family.label,
    )
    a = fidelity_fn(family.n, *construct_full_params(x, fixed_params), **kwargs).real
    b = fidelity_fn(
        family.n, *construct_full_params(flipped, fixed_params), **kwargs
    ).real
    # 0.5 / sqrt(shots) is the largest standard error a probability can have.
    return bool(np.allclose(a, b, atol=0.5 / np.sqrt(family.shots), rtol=0))


def _canonicalize(fit_params: dict, family: Family, fixed_params: dict) -> dict:
    """`canonicalize_signs`, but branch-safe and only where the flip is a degeneracy.

    `canonicalize_signs` reads the branch off eps. That works while eps is clearly
    non-zero, but these fits routinely drive eps and kap to ~1e-8, and then the
    branch is decided by rounding noise -- two runs of the same data land on
    eta = +0.0049 and eta = -0.0049 and look like they disagree when they are the
    same point. Read the branch off the phase with the most magnitude behind it
    instead; for a fit where eps dominates this is the same rule.
    """
    free_phases = {
        name: value
        for name, value in fit_params.items()
        if name in PHASE_NAMES and name not in fixed_params
    }
    if not free_phases:
        return dict(fit_params)
    branch = max(free_phases, key=lambda name: abs(free_phases[name]))
    if free_phases[branch] >= 0:
        return dict(fit_params)
    if not _phase_flip_is_a_symmetry(family, fixed_params, fit_params):
        return dict(fit_params)
    return {
        name: -value if name in PHASE_NAMES else value
        for name, value in fit_params.items()
    }


def process_single_family(family: Family, rng) -> list[dict]:
    """Given data for a single family, run the fitting procedure. Use init_values
    as the initial guess.

    Fits a growing prefix of the time steps and warm-chains each solution into the
    next, so the expensive search happens once on the shortest prefix and every later
    fit is a local refinement of it. Returns one record per prefix; the last record is
    the fit over all the data.
    """
    max_reps = len(family.n)
    prefixes = [
        max_reps
    ]  # list(range(MIN_REPETITIONS, max_reps, REPETITION_STEP)) + [max_reps]
    rows = []
    prev_fit_params = None
    generator_basis = generator_basis_for(family.label)
    readout_basis = readout_basis_for(family.label)
    fixed_params = fixed_params_for(family.label)
    lower_bounds, upper_bounds = bounds_for(family.label)

    init_values = construct_init_values(family, rng, fixed_params)
    print(f"{family.label}: x0 = {np.round(init_values, 5).tolist()}")

    for repetitions in prefixes:
        n = family.n[:repetitions]
        data = family.data[:repetitions]
        fit_params = fit_family(
            n,
            data,
            family.shots,
            rng=rng,
            x0=(
                prev_fit_params["result"].x
                if prev_fit_params is not None
                else init_values
            ),
            generator_basis=generator_basis,
            readout_basis=readout_basis,
            fixed_params=fixed_params,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            label=family.label,
        )
        row = {
            "family": family.label,
            "repetitions": repetitions,
            "shots": family.shots,
        }
        row.update(_canonicalize(fit_params, family, fixed_params))
        rows.append(row)
        prev_fit_params = fit_params  # Warm-chaining solutions
        print(
            f"  {family.label}: repetitions={repetitions:3d} "
            f"reduced_chi2={fit_params['reduced_chi2']:8.2f} "
            f"rmse={fit_params['rmse']:.4f} at_bound={fit_params['at_bound']}"
        )
    return rows, fit_params


def plot_family(
    family: Family,
    params: np.ndarray,
    pdf: PdfPages,
    generator_basis: np.ndarray = _GENERATOR_BASIS,
    readout_basis: np.ndarray | None = None,
    fixed_params: dict | None = None,
    model: np.ndarray | None = None,
) -> None:
    """Measured probabilities against the fitted model, one panel per joint state.

    Appends one page to `pdf` so every family ends up in a single vector document.
    """
    dense_n = np.linspace(float(np.min(family.n)), float(np.max(family.n)), PLOT_POINTS)
    x = construct_full_params(params, fixed_params)
    if model is None:
        model = fidelity_fn(
            dense_n,
            *x,
            generator_basis=generator_basis,
            readout_basis=readout_basis,
            label=family.label,
        ).real
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharex=True, sharey=True)
    for idx, (ax, ss) in enumerate(zip(axes, JOINT_STATES)):
        errs = family.errs[:, idx] if family.errs is not None else None
        ax.errorbar(
            family.n,
            family.data[:, idx],
            yerr=None if errs is None or errs is np.all(np.isnan(errs)) else errs,
            fmt="o",
            ms=3,
            lw=1,
            capsize=2,
            label="data",
        )
        ax.plot(
            dense_n,
            model[:, idx],
            linestyle="-",
            marker=None,
            ms=2,
            lw=1.5,
            color="crimson",
            label="fit",
        )

        ax.set_title(f"P_{ss}")
        ax.set_xlabel("number_of_operations")
    axes[0].set_ylabel("probability")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{family.label} (shots={family.shots})")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def analyze_experiments(data_path: Path, seed: int = 1, output_dir: Path = OUTPUT_DIR):
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist. Pass --data pointing at an h5/csv file."
        )
    if data_path.suffix == ".h5":
        ds_raw = xr.open_dataset(data_path)
        print("raw vars:", list(ds_raw.data_vars))
        print("raw dims:", dict(ds_raw.sizes))

        ds = prepare_dataset(ds_raw)
        rng = np.random.default_rng(seed)
        families = iter_families(ds)
        print(f"fitting {len(families)} famil{'y' if len(families) == 1 else 'ies'}\n")
    elif data_path.suffix == ".csv":
        if not data_path.exists():
            raise FileNotFoundError(f"{data_path} does not exist.")
        df = pd.read_csv(data_path)
        n = df["n"].values
        shots = (
            df["shots"].iloc[0]
            if "shots" in df.columns and len(df["shots"]) > 0
            else 800
        )
        # Try to gather all families (we only support single family here)
        data = np.stack([df[c].values for c in ["00", "01", "10", "11"]], axis=-1)
        family = Family("ibm_" + data_path.name, {}, n, data, None, shots)
        families = [family]
    else:
        raise RuntimeError(f"Expected CSV or H5 file. Received {data_path.suffix}")

    rng = np.random.default_rng(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "experiment_fit.pdf"

    rows = []
    with PdfPages(pdf_path) as pdf:
        pdf.infodict().update(
            {
                "Title": f"Two-qubit DB fits: {data_path.parent.name}",
                "Subject": str(data_path),
            }
        )
        for idx, family in enumerate(families):
            family_rows, _ = process_single_family(family, rng)
            rows.extend(family_rows)

            final = family_rows[-1]
            params = np.array([final[name] for name in PARAM_NAMES if name in final])
            plot_family(
                family,
                params,
                pdf,
                generator_basis_for(family.label),
                readout_basis_for(family.label),
                fixed_params_for(family.label),
            )
            fixed_params = fixed_params_for(family.label)
            t1, t2 = get_decay_timescale(
                final["d1"], final["d2"], final["r1"], final["r2"], family.label
            )
            print(
                f"  -> {', '.join(f'{k}={final[k]:+.5f}' for k in PARAM_NAMES if k in final)}\n"
                f"  -> fixed: {', '.join(f'{k}={v:+.5f}' for k, v in fixed_params.items())}\n"
                f"  -> cost={final['true_cost']:.1f}\n"
                f"  -> {', '.join(f'{k}={final[k]/2/np.pi/60*1e6:+.5f}' for k in PHASE_NAMES if k in final)}\n"
                f"  -> t1: {t1}, t2: {t2}\n"
                f"reduced_chi2={final['reduced_chi2']:.2f} rmse={final['rmse']:.4f}\n"
            )

    csv_path = output_dir / "experiment_fit.csv"
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "result"} for r in rows])
    frame.to_csv(csv_path, index=False)
    print(
        f"wrote {csv_path} ({len(frame)} rows) and "
        f"{pdf_path} ({len(families)} page(s))"
    )
    return frame


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze experiment runs.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Input file path (h5)",
    )
    args = parser.parse_args()

    analyze_experiments(args.data)
