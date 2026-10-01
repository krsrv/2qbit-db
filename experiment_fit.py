import itertools
import re
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares

from model import (
    DB_SETS,
    MIN_WEIGHT_PROB,
    MODEL,
    PARAM_NAMES,
    PARAMS,
    PHASE_NAMES,
    TQ_GT,
    DbSet,
    probabilities,
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
    """Recompute P_ss from the per-shot streams, else keep the stored P_ss."""
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
    """One Family per point of P_00's non-time dimensions (db_set, qubit_pair, ...)."""
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


def construct_init_values(entry: DbSet, rng: np.random.Generator) -> np.ndarray:
    """First starting point over `entry`'s free parameters.

    Fixed parameters are drawn too, then dropped, so the random stream does not change.
    """
    values = {}
    for name in PARAM_NAMES:
        kind, *spec = entry.init[name]
        values[name] = rng.uniform(*spec) if kind == "uniform" else spec[0]
    return np.array([values[name] for name in _free_names(entry)])


def get_decay_timescale(d1, d2, r1, r2, label: str) -> np.ndarray:
    """(T1, T2) in ns from the fitted rates, which are in 1/us for both methods."""
    t1 = 1 / np.array([r1, r2])
    t2 = 1 / np.array([r1 / 2 + d1, r2 / 2 + d2])
    return 1e3 * t1, 1e3 * t2


############
# Fitting
############
def _free_names(entry: DbSet) -> list[str]:
    """The parameters `entry` fits, in PARAM_NAMES order: the layout of x."""
    return [name for name in PARAM_NAMES if name not in entry.fixed]


def _params(entry: DbSet, x: np.ndarray) -> dict:
    """The full {name: value} point for the free-parameter vector `x`."""
    params = dict(entry.fixed)
    params.update(zip(_free_names(entry), x))
    return params


def _bounds(entry: DbSet) -> tuple[np.ndarray, np.ndarray]:
    names = _free_names(entry)
    return (
        np.array([entry.lower[name] for name in names]),
        np.array([entry.upper[name] for name in names]),
    )


def _residuals(
    x: np.ndarray,
    entry: DbSet,
    n: np.ndarray,
    data: np.ndarray,
    shots: int,
    weight_probs: np.ndarray | None = None,
) -> np.ndarray:
    """Whitened residuals sqrt(shots) * (model - data) / sqrt(p).

    This is the multinomial chi^2, since each row's residuals sum to zero.
    `weight_probs` fixes p (GLS passes); None uses the model prediction.
    """
    model_data = probabilities(entry, _params(entry, x), n).real
    probs = model_data if weight_probs is None else weight_probs
    diffs = model_data - data
    return (
        np.sqrt(shots) * diffs / np.sqrt(np.clip(probs, MIN_WEIGHT_PROB, None))
    ).reshape(-1)


def _run_least_squares(
    x0: np.ndarray,
    entry: DbSet,
    n: np.ndarray,
    data: np.ndarray,
    shots: int,
    weight_probs: np.ndarray | None = None,
):
    return least_squares(
        _residuals,
        x0,
        args=(entry, n, data, shots, weight_probs),
        bounds=_bounds(entry),
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )


def _gls_refine(result, entry: DbSet, n: np.ndarray, data: np.ndarray, shots: int):
    """Iterated GLS: refit with weights frozen at the current prediction."""
    for _ in range(GLS_PASSES):
        weight_probs = probabilities(entry, _params(entry, result.x), n).real
        refined = _run_least_squares(result.x, entry, n, data, shots, weight_probs)
        shift = np.max(np.abs(refined.x - result.x))
        result = refined
        if shift < GLS_TOL:
            break
    return result


def construct_x_trial(
    entry: DbSet, x0: np.ndarray | None, attempt: int, rng: np.random.Generator
) -> np.ndarray:
    """Start of restart `attempt`: `x0` for attempt 0, else a fresh draw.

    Fixed parameters are drawn too, then dropped.
    """
    if x0 is None or attempt > 0:
        draws = {p.name: rng.uniform(*p.restart) for p in PARAMS}
        x0_trial = np.array([draws[name] for name in _free_names(entry)])
    else:
        x0_trial = np.asarray(x0, dtype=float)
    return np.clip(x0_trial, *_bounds(entry))


def fit_family(
    entry: DbSet,
    n: np.ndarray,
    data: np.ndarray,
    shots: int,
    rng: np.random.Generator,
    x0: np.ndarray | None = None,
    n_restarts: int = N_RESTARTS,
) -> dict:
    """Multi-start fit of all four curves; the best attempt is then GLS-refined.

    Returns the free parameters by name plus true_cost, rmse, reduced_chi2,
    at_bound and result.
    """
    best = None
    for attempt in range(n_restarts + 1):
        x0_trial = construct_x_trial(entry, x0, attempt, rng)
        result = _run_least_squares(x0_trial, entry, n, data, shots)
        if best is None or result.cost < best.cost:
            best = result

    best = _gls_refine(best, entry, n, data, shots)

    params = dict(zip(_free_names(entry), best.x))
    params["true_cost"] = 0.5 * np.sum(_residuals(best.x, entry, n, data, shots) ** 2)
    # Rows sum to 1: 3 independent residuals per time step.
    dof = 3 * len(n) - len(best.x)
    params["rmse"] = float(np.sqrt(2 * params["true_cost"] / dof))
    params["reduced_chi2"] = float(2 * params["true_cost"] / dof)
    lower_bounds, upper_bounds = _bounds(entry)
    params["at_bound"] = int(
        np.sum(
            np.isclose(best.x, lower_bounds, atol=1e-9)
            | np.isclose(best.x, upper_bounds, atol=1e-9)
        )
    )
    params["result"] = best
    return params


def _phase_flip_is_a_symmetry(entry: DbSet, family: Family, params: dict) -> bool:
    """Whether negating every free phase moves the curve by under one standard error.

    Which flips are degeneracies depends on the set, so ask the model rather than
    tabulate; flipping a non-degenerate branch would write a row that misses the data.
    """
    point = {**entry.fixed, **{name: params[name] for name in _free_names(entry)}}
    flipped = {
        name: -value if name in PHASE_NAMES and name not in entry.fixed else value
        for name, value in point.items()
    }
    a = probabilities(entry, point, family.n).real
    b = probabilities(entry, flipped, family.n).real
    # 0.5 / sqrt(shots) is the largest standard error a probability can have.
    return bool(np.allclose(a, b, atol=0.5 / np.sqrt(family.shots), rtol=0))


def _canonicalize(fit_params: dict, entry: DbSet, family: Family) -> dict:
    """Flip so the largest free phase is positive, when that flip is a degeneracy.

    Reading the branch off eps alone fails when eps fits to ~1e-8.
    """
    free_phases = {
        name: value
        for name, value in fit_params.items()
        if name in PHASE_NAMES and name not in entry.fixed
    }
    if not free_phases:
        return dict(fit_params)
    branch = max(free_phases, key=lambda name: abs(free_phases[name]))
    if free_phases[branch] >= 0:
        return dict(fit_params)
    if not _phase_flip_is_a_symmetry(entry, family, fit_params):
        return dict(fit_params)
    return {
        name: -value if name in PHASE_NAMES else value
        for name, value in fit_params.items()
    }


def process_single_family(family: Family, rng) -> list[dict]:
    """Fit `family` on each prefix of its time steps, warm-chaining the solutions.

    Returns one row per prefix (the last covers all the data) and the final fit.
    """
    max_reps = len(family.n)
    prefixes = [
        max_reps
    ]  # list(range(MIN_REPETITIONS, max_reps, REPETITION_STEP)) + [max_reps]
    rows = []
    prev_fit_params = None
    entry = DB_SETS[family.label]

    init_values = construct_init_values(entry, rng)
    print(f"{family.label}: x0 = {np.round(init_values, 5).tolist()}")

    for repetitions in prefixes:
        n = family.n[:repetitions]
        data = family.data[:repetitions]
        fit_params = fit_family(
            entry,
            n,
            data,
            family.shots,
            rng=rng,
            x0=(
                prev_fit_params["result"].x
                if prev_fit_params is not None
                else init_values
            ),
        )
        row = {
            "family": family.label,
            "repetitions": repetitions,
            "shots": family.shots,
            "model": MODEL,
        }
        row.update(_canonicalize(fit_params, entry, family))
        rows.append(row)
        prev_fit_params = fit_params  # Warm-chaining solutions
        print(
            f"  {family.label}: repetitions={repetitions:3d} "
            f"reduced_chi2={fit_params['reduced_chi2']:8.2f} "
            f"rmse={fit_params['rmse']:.4f} at_bound={fit_params['at_bound']}"
        )
    return rows, fit_params


def plot_family(family: Family, entry: DbSet, params: dict, pdf: PdfPages) -> None:
    """Append a page of data vs. fit to `pdf`; `params` is the full parameter point."""
    dense_n = np.linspace(float(np.min(family.n)), float(np.max(family.n)), PLOT_POINTS)
    model = probabilities(entry, params, dense_n).real
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
    fig.suptitle(
        f"{family.label} (shots={family.shots}, model={MODEL}, errors={entry.pauli_labels['eta'], entry.pauli_labels['eps'], entry.pauli_labels['kap']})"
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def analyze_experiments(data_path: Path, seed: int, output_prefix: Path):
    # A CSV path may be a glob, e.g. S*_theta_0.csv, giving one family per file.
    csv_paths = sorted(data_path.parent.glob(data_path.name))
    if not csv_paths:
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
        families = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            n = df["n"].values.astype(float)
            shots = int(df["shots"].iloc[0])
            # Each CSV holds a single family, S<k> -> db_set<k>.
            set_idx = int(df["family"].iloc[0].lstrip("S"))
            counts = np.stack(
                [df[f"count_{ss}"].values for ss in JOINT_STATES], axis=-1
            )
            data = counts / counts.sum(axis=1, keepdims=True)
            errs = np.sqrt(data * (1 - data) / shots)
            families.append(
                Family(f"db_set{set_idx}_qubit_pairq3-6", {}, n, data, errs, shots)
            )
        print(f"fitting {len(families)} famil{'y' if len(families) == 1 else 'ies'}\n")
    else:
        raise RuntimeError(f"Expected CSV or H5 file. Received {data_path.suffix}")

    rng = np.random.default_rng(seed)

    # Writes output_prefix.pdf and output_prefix.csv.
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")

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

            entry = DB_SETS[family.label]
            final = family_rows[-1]
            params = {
                **entry.fixed,
                **{name: final[name] for name in _free_names(entry)},
            }
            plot_family(family, entry, params, pdf)
            t1, t2 = get_decay_timescale(
                final["d1"], final["d2"], final["r1"], final["r2"], family.label
            )
            print(
                f"  -> {', '.join(f'{k}={final[k]:+.5f}' for k in PARAM_NAMES if k in final)}\n"
                f"  -> fixed: {', '.join(f'{k}={v:+.5f}' for k, v in entry.fixed.items())}\n"
                f"  -> cost={final['true_cost']:.1f}\n"
                f"  -> {', '.join(f'{entry.pauli_labels[k]}={final[k]/2/np.pi/TQ_GT*1e6:+.5f}' for k in PHASE_NAMES if k in final)}\n"
                f"  -> t1: {t1}, t2: {t2}\n"
                f"reduced_chi2={final['reduced_chi2']:.2f} rmse={final['rmse']:.4f}\n"
            )

    csv_path = output_prefix.with_suffix(".csv")
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
        help="Input file path (h5), or a quoted csv glob such as 'S*_theta_0.csv'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output file path (folder + prefix). The generated files will be .pdf and .csv",
    )
    args = parser.parse_args()

    analyze_experiments(args.data, args.seed, args.output_prefix)
