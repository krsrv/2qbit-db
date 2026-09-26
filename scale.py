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

from experiment_fit import (
    JOINT_STATES,
    Family,
    iter_families,
    prepare_dataset,
    probs_from_shots,
    process_single_family,
)
from model import (
    CZ,
    DB_SETS,
    DIM,
    I2,
    LEVELS,
    PARAM_NAMES,
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
    TQ_GT,
    DbSet,
    _decay_super,
    construct_init_state,
    construct_msmt_op,
    evolve,
)

# claude --resume "db-scaling"

############
# Constants
############
QUBIT_PAIR = "q3-6"

# Ground truth for the synthetic data, shared by every db_set: the coefficient of each
# lab-frame Pauli error on the CZ | d1, d2, r1, r2 (1/us) | ep1, em1, ep2, em2 |
# z1, z2, z12. A set's true (eta, eps, kap) is its own Pauli triple of these
# (`get_ideal_true_params_for_expt`).
TRUE_PARAMS = {
    "XX": 0.000,
    "XY": 0.015,
    "XZ": 0.000,
    "XI": 0.020,
    "YX": 0.000,
    "YY": 0.000,
    "YZ": 0.000,
    "YI": 0.000,
    "ZX": 0.000,
    "ZY": 0.000,
    "ZZ": 0.023,
    "ZI": 0.000,
    "IX": 0.000,
    "IY": 0.000,
    "IZ": 0.040,
    "d1": 0.04568,
    "d2": 0.04218,
    "r1": 0.029955,
    "r2": 0.032101,
    "ep1": 0.0,
    "em1": 0.0,
    "ep2": 0.0,
    "em2": 0.0,
    "z1": 0.0,
    "z2": 0.0,
    "z12": 0.0,
}


############
# Synthetic data
############


def set_label(set_idx: int) -> str:
    """The family label `iter_families` gives set `set_idx` of a general dataset."""
    return f"db_set{set_idx}_qubit_pair{QUBIT_PAIR}"


def sample_joint_states(
    probs: np.ndarray, shots: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw `shots` joint outcomes per time step from a (len(n), 4) probability table.

    Columns are in `JOINT_STATES` order, whose first digit is the control qubit.
    Returns (control, target) bit arrays of shape (shots, len(n)).
    """
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random((shots, len(probs)))
    idx = (u[..., None] > cdf[None]).sum(axis=-1)
    return idx // 2, idx % 2


def _construct_unit_op(entry, params):
    """Copy logic from `construct_unit_op` in model.py, changing only the error block."""
    sequence = entry.compiled
    dissipator = _decay_super(params["d1"], params["d2"], params["r1"], params["r2"])
    # Two dwell times and one CZ error: three propagators cover the repetition.
    decay = {
        dwell: evolve(dissipator, dwell, "dissipator") for dwell in sequence.dwells
    }
    err_str = [x + y for x in ["X", "Y", "Z", "I"] for y in ["X", "Y", "Z", "I"]]
    err_str = err_str[:-1]
    error = evolve(
        sum([params[err_str[i]] * sequence.err_supers[i] for i in range(len(err_str))]),
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


def get_simulated_probabilities(k: int, true_params: dict, n: np.ndarray):
    err_ops = [
        np.kron(x, y)
        for x in [SIGMA_X, SIGMA_Y, SIGMA_Z, I2]
        for y in [SIGMA_X, SIGMA_Y, SIGMA_Z, I2]
    ]
    err_ops = err_ops[:-1]
    cz_block = (CZ, err_ops, TQ_GT, "op")
    entry = DB_SETS[set_label(k)]
    # Replace the pulse block with the error CZ block
    entry = DbSet(
        name=entry.name,
        blocks=[x if x[-1] != "op" else cz_block for x in entry.blocks],
        generator_basis=entry.generator_basis,
        readout_rot=entry.readout_rot,
        # z1, z2, z12 are not simulated, yet `get_ideal_true_params_for_expt` records
        # them in ds_true: keep them 0 in TRUE_PARAMS.
        residual_ops=None,
        pauli_labels=None,
        init=entry.init,
        fixed=None,
        lower=None,
        upper=None,
    )

    # Repeat logic in `probabilities` for the `model_dd` branch
    unit_op = _construct_unit_op(entry, true_params)
    state = construct_init_state(entry.readout_rot, LEVELS).astype(complex)
    msmt_ops = construct_msmt_op(
        true_params["ep1"],
        true_params["em1"],
        true_params["ep2"],
        true_params["em2"],
        rot=entry.readout_rot,
        levels=LEVELS,
    )
    eigenvalues, eigenvectors = np.linalg.eig(unit_op)
    weights = (msmt_ops @ eigenvectors) * np.linalg.solve(eigenvectors, state)
    return np.real((eigenvalues ** n[:, None]) @ weights.T)


def get_ideal_true_params_for_expt(k: int, true_params: dict):
    params = {
        k: true_params[k]
        for k in ["em1", "em2", "ep1", "ep2", "z1", "z2", "z12", "d1", "d2", "r1", "r2"]
    }
    if k == 0 or k == 1:
        eta, eps, kap = true_params["ZZ"], true_params["ZI"], true_params["IZ"]
    elif k == 2:
        eta, eps, kap = true_params["YY"], true_params["YI"], true_params["IY"]
    elif k == 3:
        eta, eps, kap = true_params["XX"], true_params["XI"], true_params["IX"]
    elif k == 4:
        eta, eps, kap = true_params["ZX"], true_params["XY"], true_params["YZ"]
    elif k == 5:
        eta, eps, kap = true_params["XZ"], true_params["YX"], true_params["ZY"]
    params.update({"eta": eta, "eps": eps, "kap": kap})
    return params


def make_synthetic_dataset(
    true_params: dict, reps: int, shots: int, seed: int
) -> tuple[xr.Dataset, xr.Dataset]:
    """Shot data laid out like the `_general_` ds_raw.h5 files, and its ground truth."""
    rng = np.random.default_rng(seed)
    n = np.arange(reps + 1)
    sets = np.arange(1, 6)  # Sets 1 to 5

    shot_vars, exact = {}, []
    for k in sets:
        probs = get_simulated_probabilities(k, true_params, n)
        exact.append(probs)
        control, target = sample_joint_states(probs, shots, rng)
        dims = ("shot", "number_of_operations")
        shot_vars[f"state_control_s{k}_1"] = (dims, control.astype(np.int64))
        shot_vars[f"state_target_s{k}_1"] = (dims, target.astype(np.int64))

    coords = {
        "shot": np.arange(shots),
        "number_of_operations": n,
        "db_set": sets,
        "qubit_pair": [QUBIT_PAIR],
    }
    ds_raw = xr.Dataset(shot_vars, coords=coords)

    # The real files also carry the streams stacked over (db_set, qubit_pair), and
    # P_ss computed from them.
    def stack(role):
        return (
            xr.concat([ds_raw[f"state_{role}_s{k}_1"] for k in sets], dim="db_set")
            .assign_coords(db_set=sets)
            .expand_dims(qubit_pair=[QUBIT_PAIR], axis=1)
        )

    state_c, state_t = stack("control"), stack("target")
    ds_raw = ds_raw.assign(
        state_control=state_c,
        state_target=state_t,
        **probs_from_shots(state_c, state_t),
    )
    ds_raw.attrs.update({"synthetic": 1, "seed": seed, "shots": shots, "reps": reps})

    exact = np.stack(exact)[:, None]  # (db_set, qubit_pair, number_of_operations, 4)
    ds_true = xr.Dataset(
        {
            "true_value": (
                ("db_set", "param"),
                np.array(
                    [
                        [
                            get_ideal_true_params_for_expt(k, true_params)[p]
                            for p in PARAM_NAMES
                        ]
                        for k in sets
                    ]
                ),
            ),
            **{
                f"P_{ss}_true": (
                    ("db_set", "qubit_pair", "number_of_operations"),
                    exact[..., i],
                )
                for i, ss in enumerate(JOINT_STATES)
            },
        },
        coords={
            "db_set": sets,
            "param": PARAM_NAMES,
            "qubit_pair": [QUBIT_PAIR],
            "number_of_operations": n,
        },
        attrs=ds_raw.attrs,
    )
    return ds_raw, ds_true


def write_synthetic_dataset(out_dir: Path, reps: int, shots: int, seed: int):
    raw_path, true_path = out_dir / "ds_raw.h5", out_dir / "ds_true.h5"
    for path in (raw_path, true_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists.")
    ds_raw, ds_true = make_synthetic_dataset(TRUE_PARAMS, reps, shots, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_raw.to_netcdf(raw_path)
    ds_true.to_netcdf(true_path)
    print(f"wrote {raw_path} and {true_path}")


############
# Analysis functions
############


def truncate(family: Family, n):
    return Family(
        family.label,
        family.coords,
        family.n[:n],
        family.data[:n],
        family.errs[:n],
        family.shots,
    )


def subsample_dataset(ds_raw: xr.Dataset, shots_per_subsample: int) -> xr.Dataset:
    if shots_per_subsample > ds_raw.sizes["shot"]:
        raise ValueError(
            f"subsample={shots_per_subsample} exceeds number of shots={ds_raw.sizes['shot']}"
        )
    # Sample shots without replacement
    subsampled_shots = np.random.choice(
        ds_raw["shot"].values,
        size=shots_per_subsample,
        replace=False,
    )
    # Subsamples every variable with a `shots` dimension
    ds_raw = ds_raw.sel(shot=subsampled_shots)
    # Reset shots coordinate
    ds_raw = ds_raw.assign_coords(shot=np.arange(shots_per_subsample))

    return ds_raw


def convert_to_df(rows: dict) -> pd.DataFrame:
    records = []

    for results in rows.values():
        for result in results:
            record = {k: v for k, v in result.items() if k != "result"}

            opt = result["result"]
            record.update(
                {
                    "success": opt.success,
                    "status": opt.status,
                    "message": opt.message,
                    "nfev": opt.nfev,
                    "njev": opt.njev,
                    "optimality": opt.optimality,
                }
            )

            records.append(record)

    df = pd.DataFrame(records)
    return df


def load_true_params(truth_path: Path) -> dict[int, dict[str, float]]:
    """{db_set: {param: true value}} from a ds_true.h5 written by `generate`."""
    if not truth_path.exists():
        raise FileNotFoundError(f"{truth_path} does not exist.")
    with xr.open_dataset(truth_path) as ds_true:
        table = ds_true["true_value"].to_pandas()
    return {int(k): row.to_dict() for k, row in table.iterrows()}


def extract_raw_scaling_data(
    data_path: Path,
    truth_path: Path,
    seed: int,
    shots_per_subsample: int,
    num_subsamples: int,
) -> pd.DataFrame:
    true_params = load_true_params(truth_path)
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
    else:
        raise RuntimeError(f"Expected H5 file. Received {data_path.suffix}")

    families = iter_families(ds)
    rng = np.random.default_rng(seed)
    max_reps = len(families[0].n)
    iter_n = range(10, max_reps, 5)
    rows = {x: [] for x in iter_n}
    idxes = ds_raw["shot"].values.copy()
    rng.shuffle(idxes)
    for trial in range(num_subsamples):
        shots = idxes[trial * shots_per_subsample : (trial + 1) * shots_per_subsample]
        subds = ds_raw.sel(shot=shots)
        # Reset shots coordinate
        subds = subds.assign_coords(shot=np.arange(len(shots)))

        # subds = subsample_dataset(ds_raw, idxes)
        families = iter_families(prepare_dataset(subds))
        for idx, family in enumerate(families):
            if idx > 0:
                continue
            db_set = int(family.coords["db_set"])
            for n in iter_n:
                trunc_family = truncate(family, n)
                family_rows, _ = process_single_family(trunc_family, rng)
                final = family_rows[-1]
                final["trial"] = trial
                final["db_set"] = db_set
                final.update(
                    {f"true_{name}": true_params[db_set][name] for name in PARAM_NAMES}
                )
                rows[n].append(final)

    df = convert_to_df(rows)
    return df


def analyze_scaling_data(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE of every fitted parameter against its true value, per (family, repetitions).

    Long format, one row per (family, repetitions, param). Parameters held fixed in
    the fit have no estimate column and are skipped. `bias` and `std` split the RMSE
    as rmse^2 = bias^2 + std^2.
    """
    group_cols = ["family", "repetitions"]
    if not all(col in df.columns for col in group_cols):
        raise ValueError(f"Expected columns {group_cols} in the DataFrame")

    fitted = [
        name for name in PARAM_NAMES if name in df.columns and f"true_{name}" in df
    ]
    errors = pd.DataFrame(
        {name: df[name] - df[f"true_{name}"] for name in fitted}
    ).join(df[group_cols])
    errors = errors.melt(id_vars=group_cols, var_name="param", value_name="error")
    grouped = errors.groupby([*group_cols, "param"], sort=False)["error"]
    rmse_df = pd.DataFrame(
        {
            "rmse": grouped.apply(lambda e: np.sqrt(np.mean(e**2))),
            "bias": grouped.mean(),
            "std": grouped.std(ddof=0),
            "trials": grouped.size(),
        }
    ).reset_index()
    return rmse_df


# Spacing between neighbouring slope guides, in decades of RMSE at fixed repetitions.
GUIDE_SPACING = 0.25
# The slope fit uses repetitions >= this; shorter runs can land on the flipped branch.
MIN_SLOPE_REPS = 20


def _slope_guides(ax, slope: float, color: str, label: str) -> None:
    """Parallel n^slope lines, GUIDE_SPACING decades apart, across the current view."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    # log10 intercepts of the lines through the view's corners.
    corners = [np.log10(y) - slope * np.log10(x) for x in (x0, x1) for y in (y0, y1)]
    steps = np.arange(
        np.floor(min(corners) / GUIDE_SPACING),
        np.ceil(max(corners) / GUIDE_SPACING) + 1,
    )
    xs = np.array([x0, x1])
    for i, step in enumerate(steps):
        ax.loglog(
            xs,
            10 ** (step * GUIDE_SPACING) * xs**slope,
            "--",
            lw=0.8,
            color=color,
            zorder=0,
            label=label if i == 0 else None,
        )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def plot_scaling(rmse_df: pd.DataFrame, output_path: Path) -> None:
    """RMSE against repetitions on log-log axes, one page per family.

    Each panel carries the least-squares slope of log(RMSE) against log(repetitions)
    over repetitions >= MIN_SLOPE_REPS, drawn dashed navy, and regularly spaced
    n^-1 (dark green) and n^-1/2 (light green) guides.
    """
    with PdfPages(output_path) as pdf:
        for family, frame in rmse_df.groupby("family", sort=False):
            params = list(dict.fromkeys(frame["param"]))
            ncols = min(4, len(params))
            nrows = int(np.ceil(len(params) / ncols))
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False
            )
            for ax, name in zip(axes.flat, params):
                sub = frame[frame["param"] == name].sort_values("repetitions")
                reps = sub["repetitions"].to_numpy(dtype=float)
                rmse = sub["rmse"].to_numpy(dtype=float)
                ax.loglog(reps, rmse, "o-", ms=4, lw=1.5, label="RMSE")
                _slope_guides(ax, -1, "darkgreen", r"$n^{-1}$")
                _slope_guides(ax, -0.5, "lightgreen", r"$n^{-1/2}$")
                used = (rmse > 0) & (reps >= MIN_SLOPE_REPS)
                if used.sum() >= 2:
                    slope, intercept = np.polyfit(
                        np.log(reps[used]), np.log(rmse[used]), 1
                    )
                    ax.loglog(
                        reps[used],
                        np.exp(intercept) * reps[used] ** slope,
                        "--",
                        lw=1.2,
                        color="navy",
                        label="fit",
                    )
                    ax.set_title(f"{name} (slope {slope:+.2f})")
                else:
                    ax.set_title(name)
                ax.set_xlabel("repetitions")
            for ax in axes.flat[len(params) :]:
                ax.set_visible(False)
            for row in axes:
                row[0].set_ylabel("RMSE")
            axes.flat[0].legend(loc="best", fontsize=8)
            trials = int(frame["trials"].max())
            fig.suptitle(f"{family} ({trials} trials)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimator scaling with repetitions.")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser(
        "generate", help="Write synthetic shot data from TRUE_PARAMS."
    )
    generate.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for ds_raw.h5 and ds_true.h5.",
    )
    generate.add_argument("--reps", type=int, default=50, help="Largest repetition.")
    generate.add_argument("--shots", type=int, default=5000, help="Shots per step.")
    generate.add_argument("--seed", type=int, default=1, help="Seed")

    run = commands.add_parser("run", help="Fit subsampled, truncated families.")
    run.add_argument("--data", type=Path, required=True, help="Input file path (h5)")
    run.add_argument(
        "--truth",
        type=Path,
        default=None,
        help="ds_true.h5 with the true parameters (default: next to --data).",
    )
    run.add_argument("--seed", type=int, default=1, help="Seed")
    run.add_argument(
        "--output", type=Path, required=True, help="Output parquet file path."
    )
    args = parser.parse_args()

    if args.command == "generate":
        write_synthetic_dataset(args.out, args.reps, args.shots, args.seed)
    else:
        # if args.output.exists():
        #     raise FileExistsError(f"Output file {args.output} already exists.")
        # truth = args.truth or args.data.parent / "ds_true.h5"
        # df = extract_raw_scaling_data(args.data, truth, args.seed, 5000, 100)
        # df.to_parquet(args.output, index=False)

        # rmse_df = analyze_scaling_data(df)
        # rmse_df.to_csv(args.output.with_suffix(".rmse.csv"), index=False)
        rmse_df = pd.read_csv(args.data.parent / "scale.rmse.csv")
        plot_scaling(rmse_df, args.output.with_suffix(".pdf"))
