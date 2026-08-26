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

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from least_squares import fit_joint, get_fidelities

HERE = Path(__file__).resolve().parent

STATES = ["++", "+-", "-+", "--"]
CSV_COLUMN_FOR_STATE = {"++": "pp", "+-": "pm", "-+": "mp", "--": "mm"}
PARAM_NAMES = ["eta", "eps", "kap", "d1", "d2", "r1", "r2"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_series(csv_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    df = pd.read_csv(csv_path)
    # print(.shape)
    n = np.arange(len(df), dtype=float)
    # data = {
    #     state: df[CSV_COLUMN_FOR_STATE[state]].to_numpy(dtype=float) for state in STATES
    # }
    return (
        n,
        df.to_numpy().T,
    )  # ---------------------------------------------------------------------------


# Modes: the 13 complex per-step coefficients that get raised to the n-th power
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeGroup:
    key: str
    label: str
    amplitude: "callable"  # params -> float, in [0, 1]
    phase: "callable"  # params -> float (radians); point is amplitude * exp(+-1j*phase)
    paired: bool = True  # True: plot the +/- conjugate pair; False: single point


MODE_GROUPS: list[ModeGroup] = [
    ModeGroup(
        "eps_eta",
        r"$d_1^4$, $\phi=8(\epsilon+\eta)$",
        lambda p: p["d1"] ** 4,
        lambda p: 8 * (p["eps"] + p["eta"]),
    ),
    ModeGroup(
        "eta_kap",
        r"$d_2^4$, $\phi=8(\eta+\kappa)$",
        lambda p: p["d2"] ** 4,
        lambda p: 8 * (p["eta"] + p["kap"]),
    ),
    ModeGroup(
        "eps_m_kap",
        r"$d_1^4 d_2^4$, $\phi=8(\epsilon-\kappa)$",
        lambda p: (p["d1"] ** 4) * (p["d2"] ** 4),
        lambda p: 8 * (p["eps"] - p["kap"]),
    ),
    ModeGroup(
        "eps_p_kap",
        r"$d_1^4 d_2^4$, $\phi=8(\epsilon+\kappa)$",
        lambda p: (p["d1"] ** 4) * (p["d2"] ** 4),
        lambda p: 8 * (p["eps"] + p["kap"]),
    ),
    ModeGroup(
        "eta_m_kap",
        r"$r_1^4 d_2^4$, $\phi=8(\eta-\kappa)$",
        lambda p: (p["r1"] ** 4) * (p["d2"] ** 4),
        lambda p: 8 * (p["eta"] - p["kap"]),
    ),
    ModeGroup(
        "eps_m_eta",
        r"$d_1^4 r_2^4$, $\phi=8(\epsilon-\eta)$",
        lambda p: (p["d1"] ** 4) * (p["r2"] ** 4),
        lambda p: 8 * (p["eps"] - p["eta"]),
    ),
    ModeGroup(
        "const", "constant (DC) mode", lambda p: 1.0, lambda p: 0.0, paired=False
    ),
]


def mode_points(params: dict[str, float]) -> list[tuple[str, str, complex]]:
    """Returns a flat list of (group_key, group_label, complex_value) for all 13 modes."""
    points = []
    for group in MODE_GROUPS:
        amp = group.amplitude(params)
        phi = group.phase(params)
        if group.paired:
            points.append((group.key, group.label, amp * np.exp(-1j * phi)))
            points.append((group.key, group.label, amp * np.exp(1j * phi)))
        else:
            points.append((group.key, group.label, complex(amp, 0.0)))
    return points


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Categorical colors (fixed order), reference palette slots 1-7 (blue, orange, aqua,
# yellow, magenta, green, violet). Marker shape is assigned per group as a redundant,
# color-independent encoding; fill (solid vs hollow) distinguishes noiseless vs noisy.
GROUP_COLOR = {
    "eps_eta": "#2a78d6",
    "eta_kap": "#eb6834",
    "eps_m_kap": "#1baf7a",
    "eps_p_kap": "#eda100",
    "eta_m_kap": "#e87ba4",
    "eps_m_eta": "#008300",
    "const": "#4a3aa7",
}
GROUP_MARKER = {
    "eps_eta": "o",
    "eta_kap": "s",
    "eps_m_kap": "^",
    "eps_p_kap": "D",
    "eta_m_kap": "v",
    "eps_m_eta": "P",
    "const": "*",
}


def plot_modes(fitted: dict[str, dict[str, float]], out_path: Path) -> None:
    """fitted: {"noiseless": params_dict, "noisy": params_dict}"""
    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0, 1.08)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_rlabel_position(112.5)
    ax.tick_params(axis="y", labelsize=8, colors="#898781")
    ax.tick_params(axis="x", labelsize=9, colors="#52514e")
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.spines["polar"].set_color("#c3c2b7")

    seen_group_handles = {}
    dataset_names = ["noiseless", "noisy"]
    for dataset_name in fitted:
        params = fitted[dataset_name]
        points = mode_points(params)
        if dataset_name == "noiseless":
            marker = "o"
            color = "#2a78d6"
        else:
            marker = "*"
            color = "#008300"
        for key, label, val in points:
            r = abs(val)
            theta = np.angle(val)
            # color = GROUP_COLOR[key]
            # marker = GROUP_MARKER[key]
            face = color
            edge = "white"
            lw = 0.6
            ax.scatter(
                theta,
                r,
                s=80,
                marker=marker,
                facecolor=face,
                edgecolor=edge,
                linewidths=lw,
                zorder=3,
                clip_on=False,
            )
            if key not in seen_group_handles:
                seen_group_handles[key] = plt.Line2D(
                    [],
                    [],
                    marker=marker,
                    linestyle="none",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=9,
                    # label=label,
                )

    group_handles = [seen_group_handles[g.key] for g in MODE_GROUPS]
    group_legend = ax.legend(
        handles=group_handles,
        title="Mode",
        loc="upper left",
        bbox_to_anchor=(1.12, 1.05),
        fontsize=9,
        title_fontsize=10,
        frameon=False,
    )
    ax.add_artist(group_legend)

    style_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#2a78d6",
            markeredgecolor="white",
            markersize=9,
            label="Noiseless",
        ),
        plt.Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markerfacecolor="#008300",
            markeredgecolor="white",
            markersize=9,
            markeredgewidth=1.6,
            label="Noisy",
        ),
    ]
    ax.legend(
        handles=style_handles,
        title="Dataset",
        loc="lower left",
        bbox_to_anchor=(1.12, 0.0),
        fontsize=9,
        title_fontsize=10,
        frameon=False,
    )

    ax.set_title(
        "Fitted oscillation modes (per-step complex coefficients)\n"
        "radius = decay amplitude, angle = phase",
        fontsize=11,
        pad=24,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_data_and_fit(n_range, data: dict, params, out_path):
    plt.figure(figsize=(8.5, 7.5))
    colors = [
        "#2a78d6",  # blue
        "#e26f46",  # orange
        "#56a764",  # green
        "#ab40e8",  # purple
        "#edc149",  # yellow
        "#d23a3a",  # red
        "#1f979d",  # teal
        "#6f4c99",  # violet
        "#bca35c",  # gold
        "#7e909c",  # gray/blue-gray
    ]
    for i, key in enumerate(data):
        plt.plot(
            n_range,
            data[key],
            color=colors[i],
            marker="o",
            markersize=4,
            linewidth=1.7,
            markerfacecolor=colors[i],
            markeredgecolor="white",
            label=key,
        )

        fitted_data = get_fidelities(
            n_range,
            params["eta"],
            params["eps"],
            params["kap"],
            params["d1"],
            params["d2"],
            params["r1"],
            params["r2"],
        )
        plt.plot(
            n_range,
            fitted_data,
            color=colors[i],
            linestyle="--",
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=colors[i],
            linewidth=1.8,
            label=f"{key} fit",
        )
    plt.legend()
    plt.savefig(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def format_params(params: dict[str, float]) -> str:
    lines = []
    for name in PARAM_NAMES:
        lines.append(f"    {name:>4s} = {params[name]: .6f}")
    lines.append(f"    rmse = {params['rmse']:.3e}")
    return "\n".join(lines)


def main() -> None:
    datasets = {
        "noiseless": HERE / "data/decay_noiseless_data.csv",
        "noisy": HERE / "data/decay_noisy_data.csv",
    }

    fitted = {}

    # Global multi-start fit for the noiseless baseline.
    n0, data0 = load_series(datasets["noiseless"])
    params0 = fit_joint(n0, data0)
    fitted["noiseless"] = params0
    print(f"[noiseless] joint fit over states {STATES} (n=0..{int(n0[-1])}):")
    print(format_params(params0))
    print()

    # Warm-start the noisy fit from the noiseless solution so both fits land in the
    # same phase branch (see fit_joint docstring) and are directly comparable.
    n1, data1 = load_series(datasets["noisy"])
    params1 = fit_joint(n1, data1)
    fitted["noisy"] = params1
    print(
        f"[noisy] joint fit over states {STATES} (n=0..{int(n1[-1])}), warm-started from noiseless fit:"
    )
    print(format_params(params1))
    print()

    out_path = HERE / "output/mode_polar_chart.pdf"
    plot_modes(fitted, out_path)
    print(f"Saved polar mode chart to {out_path}")
    eps, eta, kap = np.pi / 180, 0.4 * np.pi / 180, 0.2 * np.pi / 180
    print("True eps, eta, kap: ", eps, eta, kap)


def ibm_analysis():
    import json

    with open(HERE / "counts.json", "r") as f:
        data = json.load(f)

    fitted = {}
    data = data["results"]
    n = len(data)
    n1 = np.zeros(n, dtype=np.int_)
    data1 = {
        "++": np.zeros(n, dtype=np.float32),
        "+-": np.zeros(n, dtype=np.float32),
        "-+": np.zeros(n, dtype=np.float32),
        "--": np.zeros(n, dtype=np.float32),
    }

    for i, dp in enumerate(data):
        n1[i] = dp["n_blocks"]
        num_shots = dp["total_shots"]
        data1["++"][i] = dp["counts"]["00"] / num_shots
        data1["+-"][i] = dp["counts"]["01"] / num_shots
        data1["-+"][i] = dp["counts"]["10"] / num_shots
        data1["--"][i] = dp["counts"]["11"] / num_shots

    # Warm-start the noisy fit from the noiseless solution so both fits land in the
    # same phase branch (see fit_joint docstring) and are directly comparable.
    params1 = fit_joint(n1, data1)
    fitted["noisy"] = params1
    print(f"[noisy] joint fit over states {STATES} (n=0..{int(n1[-1])})")
    print(format_params(params1))
    print()

    out_path = HERE / "output/ibm_polar_chart.pdf"
    plot_modes(fitted, out_path)
    plot_data_and_fit(n1, data1, params1, HERE / "ibm_fit.pdf")
    print(f"Saved polar mode chart to {out_path}")
    eps, eta, kap = np.pi / 180, 0.4 * np.pi / 180, 0.2 * np.pi / 180
    print("True eps, eta, kap: ", eps, eta, kap)


if __name__ == "__main__":
    main()
    # ibm_analysis()
