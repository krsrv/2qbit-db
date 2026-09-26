import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from scale import get_simulated_probabilities

TRUE_PARAMS = {
    "II": 0.000,
    "XX": 0.000,
    "XY": 0.000,
    "XZ": 0.000,
    "XI": 0.000,
    "YX": 0.000,
    "YY": 0.000,
    "YZ": 0.000,
    "YI": 0.000,
    "ZX": 0.000,
    "ZY": 0.000,
    "ZZ": 0.000,
    "ZI": 0.000,
    "IX": 0.000,
    "IY": 0.000,
    "IZ": 0.000,
    "d1": 0.04568,
    "d2": 0.03218,
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


def construct_params(error_idx, eps=0.05, remove_decay=False):
    keys = list(TRUE_PARAMS.keys())
    params = TRUE_PARAMS.copy()
    for i, key in enumerate(keys):
        if i < 16:
            params[key] = eps if i == error_idx else 0.0
        if i >= 16 and remove_decay:
            params[key] = 0
    return params


n = np.arange(50)
eps = 0.05
prob_result = np.zeros((6, 16, 4, 50))
for k in range(6):
    for error_idx in range(16):
        params = construct_params(error_idx, eps)
        prob = get_simulated_probabilities(k, params, n)
        prob_result[k, error_idx] = prob.T

eps_range = np.logspace(-3, -1, 9)
suppression_result = np.zeros((6, 16, len(eps_range)))

for error_idx in range(16):
    for k in range(6):
        params = construct_params(error_idx, 0.0, remove_decay=False)
        ideal_op = get_simulated_probabilities(k, params, [1], return_op=True)
        for eps_idx, eps_ in enumerate(eps_range):
            params = construct_params(error_idx, eps_, remove_decay=False)
            op = get_simulated_probabilities(k, params, [1], return_op=True)
            suppression_result[k, error_idx, eps_idx] = np.sum(np.abs(op - ideal_op))

# Each set's own Pauli triple (model.DB_SETS pauli_labels); the rest are "outside".
OWN = {
    0: ["ZZ", "ZI", "IZ"],
    1: ["ZZ", "ZI", "IZ"],
    2: ["YY", "YI", "IY"],
    3: ["XX", "XI", "IX"],
    4: ["ZX", "XY", "YZ"],
    5: ["XZ", "YX", "ZY"],
}
OWN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
OTHER_COLOR, STRONG_OTHER_COLOR, INK = "#c9c7c1", "#6e6c66", "#3d3c38"
N_LABELED = 2  # outside Paulis direct-labelled per panel: the ones that move it most
# Curves closer than this are indistinguishable on the plot and share one label.
SAME_CURVE_TOL = 5e-3

keys = list(TRUE_PARAMS.keys())[:16]
# Page 3's set colours, categorical order: blue, orange, aqua, yellow, magenta, green.
SET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def plot_panels(values, ylabel, ylim, title):
    """(6, 4) panels of `values[k, error_idx, idx]`; own triple coloured."""
    fig, axs = plt.subplots(6, 4, figsize=(16, 20), sharex=True, sharey=True)
    for k in range(6):
        own = OWN[k]
        for idx in range(4):
            ax = axs[k, idx]
            baseline = values[k, keys.index("II"), idx]
            others = [p for p in keys if p != "II" and p not in own]
            effect = {
                p: np.abs(values[k, keys.index(p), idx] - baseline).max()
                for p in others
            }
            # Outside Paulis with indistinguishable curves share one label ("IX = ZX").
            groups = []
            for p in sorted(others, key=effect.get, reverse=True):
                y = values[k, keys.index(p), idx]
                for group in groups:
                    first = values[k, keys.index(group[0]), idx]
                    if np.allclose(y, first, atol=SAME_CURVE_TOL):
                        group.append(p)
                        break
                else:
                    groups.append([p])
            strong = [g for g in groups[:N_LABELED] if effect[g[0]] > 1e-3]
            for p in others:
                labelled = any(p in g for g in strong)
                ax.plot(
                    n,
                    values[k, keys.index(p), idx],
                    lw=0.8,
                    zorder=1,
                    color=STRONG_OTHER_COLOR if labelled else OTHER_COLOR,
                )
            if strong:
                # Listed in a corner: the curves often peak together.
                text = "\n".join(f"{', '.join(g)}: {effect[g[0]]:.2f}" for g in strong)
                ax.text(
                    0.97,
                    0.96,
                    "strongest outside, max |dP|\n" + text,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    color=INK,
                    bbox=dict(fc="white", ec="none", alpha=0.85),
                )
            for p, color in zip(own, OWN_COLORS):
                ax.plot(
                    n,
                    values[k, keys.index(p), idx],
                    lw=2,
                    color=color,
                    zorder=3,
                    label=p,
                )
            # On top, so a curve that coincides (a cancelled error) shows dashed.
            ax.plot(n, baseline, "--", lw=1.2, color="black", zorder=4)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.2)
            if idx == 0:
                ax.set_ylabel(f"Set {k}\n{ylabel}")
                handles = ax.get_legend_handles_labels()[0] + [
                    plt.Line2D([], [], ls="--", lw=1.5, color="black"),
                    plt.Line2D([], [], lw=0.8, color=OTHER_COLOR),
                ]
                ax.legend(
                    handles,
                    own + ["no error", "other Paulis"],
                    fontsize=7,
                    loc="center right",
                    title="0.05 error on",
                    title_fontsize=7,
                )
            if k == 0:
                ax.set_title(f"P_{['00', '01', '10', '11'][idx]}")
            if k == 5:
                ax.set_xlabel("# Repetitions")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


with PdfPages("output/dd_echo.pdf") as pdf:
    # Page 1: the probabilities.
    pdf.savefig(
        plot_panels(
            prob_result,
            "probability",
            (0, 1),
            f"Single lab-frame Pauli error ({eps}) on every CZ: own triple in colour,"
            " other Paulis grey (strongest two dark grey, listed)",
        ),
        bbox_inches="tight",
    )

    # Page 2: the same, minus the no-error curve, so small effects are visible.
    deviation = prob_result - prob_result[:, [keys.index("II")]]
    lim = 1.05 * np.abs(deviation).max()
    pdf.savefig(
        plot_panels(
            deviation,
            "P - P(no error)",
            (-lim, lim),
            f"Deviation from the no-error curve for a single {eps} Pauli error on every CZ",
        ),
        bbox_inches="tight",
    )

    # Page 3: how much each set suppresses each Pauli error, from its eps scaling.
    # Distance of one repetition's superoperator from the ideal one, without decay.
    # Slope 1 on log-log axes: the error survives at first order; on the floor: cancels.
    floor = 1e-6  # an exactly cancelled error is drawn here instead of at log(0)
    fig, axs = plt.subplots(4, 4, figsize=(16, 14), sharex=True, sharey=True)
    for error_idx, ax in enumerate(axs.flat):
        if keys[error_idx] == "II":
            continue  # its panel holds the legend
        for k, color in zip(range(6), SET_COLORS):
            own = keys[error_idx] in OWN[k]
            ax.loglog(
                eps_range,
                # An exact cancellation is round-off (~1e-14): draw it at the floor.
                np.maximum(suppression_result[k, error_idx], floor),
                "o-",
                color=color,
                ms=3,
                lw=2.5 if own else 1,
                label=f"Set {k}",
            )
        for power, style in ((1, ":"), (2, "-.")):
            ax.loglog(
                eps_range,
                30 * (eps_range / eps_range[-1]) ** power,
                style,
                color=INK,
                lw=0.8,
                label=rf"$\propto \epsilon^{power}$",
            )
        ax.set_title(keys[error_idx])
        # Each set's eps scaling of this error, from the slope over the two smallest
        # eps: ~1 survives at first order, ~2 is suppressed, on the floor cancels.
        first, second = (
            suppression_result[:, error_idx, 0],
            suppression_result[:, error_idx, 1],
        )
        slope = np.log(second / first) / np.log(eps_range[1] / eps_range[0])
        order = [
            "exact" if first[k] < floor else "eps" if slope[k] < 1.5 else "eps^2"
            for k in range(6)
        ]
        at_small_eps = suppression_result[:, error_idx, 0]
        survives = [k for k in range(6) if at_small_eps[k] >= 0.1 * at_small_eps.max()]
        cancels = [
            (
                f"{k}"
                # if at_small_eps[k] < floor
                # else f"{k} (/{at_small_eps.max() / at_small_eps[k]:.0f})"
            )
            for k in range(6)
            if k not in survives
        ]
        ax.text(
            0.03,
            0.97,
            "\n".join(
                f"{name}: sets "
                + (", ".join(str(k) for k in range(6) if order[k] == name) or "-")
                for name in ("eps", "eps^2", "exact")
            )
            + f"\n cancels: sets {', '.join(cancels)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=INK,
            bbox=dict(fc="white", ec="none", alpha=0.85),
        )
        ax.set_ylim(floor / 2, 1e2)
        ax.grid(alpha=0.2, which="both")
    ax_legend = axs.flat[keys.index("II")]
    ax_legend.legend(
        *axs.flat[1].get_legend_handles_labels(),
        loc="center",
        title="thick: the Pauli is in the set's own triple",
    )
    ax_legend.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel(r"error size $\epsilon$")
    for ax in axs[:, 0]:
        ax.set_ylabel("sum |op - ideal op|, one repetition")
    fig.suptitle(
        "One repetition's deviation from ideal for a single lab-frame Pauli error of"
        " size eps on every CZ, per set (no decay)",
        y=1.0,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
plt.close("all")
