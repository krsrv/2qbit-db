import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import least_squares

from experiment_fit import iter_families, prepare_dataset
from matrix_pencil import _fit_weights, _hankel_pair


def get_spectrum(G0, G1, shots, N=None):
    U, s, Vh = np.linalg.svd(G0, full_matrices=False)
    if N is None:
        ul = np.sqrt(1 / shots)
        idx = np.where(s < ul)[0]
        N = idx[0] if len(idx) != 0 else 13
    U_N, s_N, V_N = U[:, :N], s[:N], Vh[:N].conj().T
    # U1, s1, V1h = np.linalg.svd(G1, full_matrices=False)
    # G1_f = (U1[:, :N] * s1[:N]) @ V1h[:N]
    A = U_N.conj().T @ G1 @ V_N / s_N[None, :]

    eigenvalues = np.linalg.eigvals(A)
    return eigenvalues


def to_unit(theta):
    # (theta, phi) -> unit vector (a, b, c); fixes ||p|| = 1 so the noise floor is scale-free
    t, f = theta
    return np.array([np.cos(t), np.sin(t) * np.cos(f), np.sin(t) * np.sin(f)])


def _get_weights(evals, data):
    weight_residual = [_fit_weights(data[i : i + 1], evals) for i in range(4)]
    weights = [weight_residual[i][0] for i in range(4)]
    return weights


def get_model_data(params, data, shots, count=None, N=None):
    M = data.shape[1]
    L = M // 2
    G0 = np.stack([_hankel_pair(data[i : i + 1], L)[0] for i in range(4)], axis=0)
    G1 = np.stack([_hankel_pair(data[i : i + 1], L)[1] for i in range(4)], axis=0)

    p = to_unit(params)
    G0 = np.sum(p[:, None, None] * G0[:3], axis=0)
    G1 = np.sum(p[:, None, None] * G1[:3], axis=0)
    eigenvalues = get_spectrum(G0, G1, shots, N=None).astype(
        complex
    )  # real negatives -> NaN otherwise

    weights = _get_weights(eigenvalues, data)
    M = data[:1].shape[1]
    if count is None:
        V = eigenvalues[None, :] ** np.arange(0, M)[:, None]
    else:
        V = eigenvalues[None, :] ** np.linspace(0, M - 1, count)[:, None]
    model_data = np.array([(V @ weights[i].T).real for i in range(4)])
    return eigenvalues, model_data


def residuals(params, data, shots, N=None):
    _, model_data = get_model_data(params, data, shots, N=N)
    # least_squares squares and sums these itself
    return (model_data[:, :, 0] - data).reshape(-1)


def debug_params(params):
    print("Params (theta, phi):", params)
    print("Params parameters (a, b, c):", to_unit(params))


def convert_osc_to_str(osc):
    return "\n".join(
        [
            f"{np.log(osc[i]).real:.5e}".ljust(30)
            + f"+ {np.log(osc[i]).imag:.5e}j".ljust(30)
            for i in range(osc.shape[0])
        ]
    )


def debug_plot(params, data):
    M = data.shape[1]
    _, model_data = get_model_data(params, data, 100)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for idx in range(4):
        ax = axs[idx // 2, idx % 2]
        ax.plot(
            np.arange(data[idx].shape[0]), data[idx], "o", markersize=4, label="Data"
        )
        ax.plot(np.linspace(0, M - 1, 100), model_data[idx][:, 0], label="Model")
        ax.set_title(f"Series {idx}")
        ax.set_ylim(0, 1)
        ax.legend()
    plt.tight_layout()
    plt.show()


def _extract_params(data: np.ndarray, shots=5000, N=None):
    "data is a (4 x T) array"
    # Initial guess for (theta, phi): (a, b, c) = (1, 1, 1) / sqrt(3)
    initial_params = np.array([np.arccos(1 / np.sqrt(3)), np.pi / 4])

    # Repeat least squares optimization 10 times
    results = []
    res = least_squares(
        residuals,
        initial_params,
        args=(data, shots, N),
    )
    results.append(res)
    for i in range(10):
        # p and -p give the same pencil, so the a >= 0 hemisphere covers everything
        random_init = np.random.uniform([0.0, 0.0], [np.pi / 2, 2 * np.pi])
        res = least_squares(
            residuals,
            random_init,
            args=(data, shots, N),
        )
        results.append(res)
    result = min(
        results, key=lambda r: r.cost
    )  # Select the best result (lowest sum of squared residuals)

    # Extract optimized parameters
    opt_params = result.x
    return opt_params


def get_model_fit(data: np.ndarray, shots=5000, count=None, N=None):
    params = _extract_params(data, shots=shots, N=N)
    evals, model_data = get_model_data(params, data, shots, count=count, N=N)
    omegas = np.log(evals)
    sort_idx = np.argsort(np.abs(omegas.imag))
    omegas = omegas[sort_idx]
    return params, omegas, model_data


def analyze_instance():
    file = "data/63_XI,ZI,IZ_10-15deg/#6512_40b_2Q_DB_general_inj_err_125648/ds_raw.h5"
    ds_raw = xr.open_dataset(file)
    ds = prepare_dataset(ds_raw)
    families = iter_families(ds)

    state_label = {0: "00", 1: "01", 2: "10", 3: "11"}
    fig, axs = plt.subplots(6, 4, figsize=(20, 12))
    for family_idx in range(6):
        data = families[family_idx].data.T
        params, omegas, model_data = get_model_fit(
            data, families[family_idx].shots, count=100, N=13
        )
        for idx in range(4):
            ax = axs[family_idx, idx]
            ax.plot(
                np.arange(data[idx].shape[0]),
                data[idx],
                "o",
                markersize=4,
                label=f"{idx}",
            )
            ax.plot(
                np.linspace(0, data[idx].shape[0], 100),
                model_data[idx][:, 0],
                label="Model",
            )
            ax.set_title(f"{state_label[idx]}")
            if idx == 0:
                ax.set_ylabel(f"Set {family_idx}")
            ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def analyze_errors():
    import glob

    h5_files = sorted(glob.glob("data/errors/XI,ZI,IZ_10-15deg/*/ds_raw.h5"))

    fig, axs = plt.subplots(len(h5_files), 5 * 6, figsize=(20 * 6, 24))
    state_label = {0: "00", 1: "01", 2: "10", 3: "11"}
    family_label = {
        0: "No DD",
        1: "ZZ,IZ,ZI",
        2: "YY,IY,YI",
        3: "XX,XI,IX",
        4: "ZX,XY,YZ",
        5: "XZ,YX,ZY",
    }
    for file_idx, file in enumerate(h5_files):
        file_label = file.split("#")[1].split("_")[0]
        ds_raw = xr.open_dataset(file)
        ds = prepare_dataset(ds_raw)
        families = iter_families(ds)
        for family_idx in range(6):
            data = families[family_idx].data.T
            params, omegas, model_data = get_model_fit(
                data, families[family_idx].shots, 100
            )

            for idx in range(4):
                ax = axs[file_idx, 5 * family_idx + idx]
                ax.plot(
                    np.arange(data[idx].shape[0]),
                    data[idx],
                    "o",
                    markersize=4,
                    label="Data",
                )
                ax.plot(
                    np.linspace(0, data[idx].shape[0], 100),
                    model_data[idx][:, 0],
                    label="Model",
                )
                if idx == 0:
                    ax.set_ylabel(f"{file_label}")
                if idx == 3:
                    param_str = "\n".join(
                        f"{name} = {p:.4f}" for name, p in zip(("θ", "φ"), params)
                    )
                    ax.text(
                        0.97,
                        0.75,
                        param_str,
                        transform=ax.transAxes,
                        va="top",
                        ha="right",
                        fontsize=10,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                    )
                if file_idx == 0:
                    ax.set_title(f"{family_label[family_idx]}: {state_label[idx]}")
                ax.set_ylim(0, 1)
                ax.legend(loc="upper right")

            ax = axs[file_idx, 5 * family_idx + 4]
            ax.plot(omegas.imag, np.ones_like(omegas.real), "o", markersize=4)
            ax.axhline(0, color="gray", lw=0.5)
            ax.axvline(0, color="gray", lw=0.5)
            ax.set_xlabel(r"Im $\omega$")
            ax.set_ylabel(r"Re $\omega$")
            if file_idx == 0:
                ax.set_title(f"{family_label[family_idx]}: $\\omega$")
    plt.tight_layout()
    plt.savefig("output/multi_family_data_analysis.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # analyze_errors()
    analyze_instance()
