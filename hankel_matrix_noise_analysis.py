from pathlib import Path

import numpy as np
from matrix_pencil import _hankel_pair
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent


def get_fidelity(state, n, eta, eps, kap, d1, d2, r1, r2):
    if state == "++":
        return (
            1
            / 16
            * (
                4
                + (np.exp(-8j * (eps + eta)) * (d1**4)) ** n
                + (np.exp(8j * (eps + eta)) * (d1**4)) ** n
                + (np.exp(-8j * (eta + kap)) * (d2**4)) ** n
                + (np.exp(8j * (eta + kap)) * (d2**4)) ** n
                + (np.exp(-8j * (eps - kap)) * (d1**4) * (d2**4)) ** n
                + (np.exp(8j * (eps - kap)) * (d1**4) * (d2**4)) ** n
                + (np.exp(-8j * (eps + kap)) * (d1**4) * (d2**4)) ** n
                + (np.exp(8j * (eps + kap)) * (d1**4) * (d2**4)) ** n
                + (np.exp(-8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                + (np.exp(8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                + (np.exp(-8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                + (np.exp(8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                + (r1 - 1)
                * (
                    (np.exp(8j * (eta + kap)) * (d2**4)) ** n
                    - (np.exp(-8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                )
                / (np.exp(4j * eta) + r1)
                + (r1 - 1)
                * (
                    (np.exp(-8j * (eta + kap)) * (d2**4)) ** n
                    - (np.exp(8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                )
                / (np.exp(-4j * eta) + r1)
                + (r2 - 1)
                * (
                    (np.exp(-8j * (eps + eta)) * (d1**4)) ** n
                    - (np.exp(-8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                )
                / (np.exp(-4j * eta) + r2)
                + (r2 - 1)
                * (
                    (np.exp(8j * (eps + eta)) * (d1**4)) ** n
                    - (np.exp(8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                )
                / (np.exp(4j * eta) + r2)
            )
        )
    elif state == "+-":
        return (
            1
            / 16
            * (
                4
                + (np.exp(-8j * (eps + eta)) * (d1**4)) ** n
                + (np.exp(8j * (eps + eta)) * (d1**4)) ** n
                - (np.exp(-8j * (eta + kap)) * (d2**4)) ** n
                - (np.exp(8j * (eta + kap)) * (d2**4)) ** n
                - (np.exp(-8j * (eps - kap)) * (d1**4) * (d2**4)) ** n
                - (np.exp(8j * (eps - kap)) * (d1**4) * (d2**4)) ** n
                - (np.exp(-8j * (eps + kap)) * (d1**4) * (d2**4)) ** n
                - (np.exp(8j * (eps + kap)) * (d1**4) * (d2**4)) ** n
                - (np.exp(-8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                - (np.exp(8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                + (np.exp(-8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                + (np.exp(8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                - (r1 - 1)
                * (
                    (np.exp(8j * (eta + kap)) * (d2**4)) ** n
                    - (np.exp(-8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                )
                / (np.exp(4j * eta) + r1)
                - (r1 - 1)
                * (
                    (np.exp(-8j * (eta + kap)) * (d2**4)) ** n
                    - (np.exp(8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                )
                / (np.exp(-4j * eta) + r1)
                + (r2 - 1)
                * (
                    (np.exp(-8j * (eps + eta)) * (d1**4)) ** n
                    - (np.exp(-8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                )
                / (np.exp(-4j * eta) + r2)
                + (r2 - 1)
                * (
                    (np.exp(8j * (eps + eta)) * (d1**4)) ** n
                    - (np.exp(8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                )
                / (np.exp(4j * eta) + r2)
            )
        )
    elif state == "-+":
        return (
            1
            / 16
            * (
                4
                - (np.exp(-8j * (eps + eta)) * (d1**4)) ** n
                - (np.exp(8j * (eps + eta)) * (d1**4)) ** n
                + (np.exp(-8j * (eta + kap)) * (d2**4)) ** n
                + (np.exp(8j * (eta + kap)) * (d2**4)) ** n
                - (np.exp(-8j * (eps - kap)) * (d1**4) * (d2**4)) ** n
                - (np.exp(8j * (eps - kap)) * (d1**4) * (d2**4)) ** n
                - (np.exp(-8j * (eps + kap)) * (d1**4) * (d2**4)) ** n
                - (np.exp(8j * (eps + kap)) * (d1**4) * (d2**4)) ** n
                + (np.exp(-8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                + (np.exp(8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                - (np.exp(-8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                - (np.exp(8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                + (r1 - 1)
                * (
                    (np.exp(8j * (eta + kap)) * (d2**4)) ** n
                    - (np.exp(-8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                )
                / (np.exp(4j * eta) + r1)
                + (r1 - 1)
                * (
                    (np.exp(-8j * (eta + kap)) * (d2**4)) ** n
                    - (np.exp(8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                )
                / (np.exp(-4j * eta) + r1)
                - (r2 - 1)
                * (
                    (np.exp(-8j * (eps + eta)) * (d1**4)) ** n
                    - (np.exp(-8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                )
                / (np.exp(-4j * eta) + r2)
                - (r2 - 1)
                * (
                    (np.exp(8j * (eps + eta)) * (d1**4)) ** n
                    - (np.exp(8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                )
                / (np.exp(4j * eta) + r2)
            )
        )
    elif state == "--":
        return (
            1
            / 16
            * (
                4
                - (np.exp(-8j * (eps + eta)) * (d1**4)) ** n
                - (np.exp(8j * (eps + eta)) * (d1**4)) ** n
                - (np.exp(-8j * (eta + kap)) * (d2**4)) ** n
                - (np.exp(8j * (eta + kap)) * (d2**4)) ** n
                + (np.exp(-8j * (eps - kap)) * (d1**4) * (d2**4)) ** n
                + (np.exp(8j * (eps - kap)) * (d1**4) * (d2**4)) ** n
                + (np.exp(-8j * (eps + kap)) * (d1**4) * (d2**4)) ** n
                + (np.exp(8j * (eps + kap)) * (d1**4) * (d2**4)) ** n
                - (np.exp(-8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                - (np.exp(8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                - (np.exp(-8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                - (np.exp(8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                - (r1 - 1)
                * (
                    (np.exp(8j * (eta + kap)) * (d2**4)) ** n
                    - (np.exp(-8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                )
                / (np.exp(4j * eta) + r1)
                - (r1 - 1)
                * (
                    (np.exp(-8j * (eta + kap)) * (d2**4)) ** n
                    - (np.exp(8j * (eta - kap)) * (r1**4) * (d2**4)) ** n
                )
                / (np.exp(-4j * eta) + r1)
                - (r2 - 1)
                * (
                    (np.exp(-8j * (eps + eta)) * (d1**4)) ** n
                    - (np.exp(-8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                )
                / (np.exp(-4j * eta) + r2)
                - (r2 - 1)
                * (
                    (np.exp(8j * (eps + eta)) * (d1**4)) ** n
                    - (np.exp(8j * (eps - eta)) * (d1**4) * (r2**4)) ** n
                )
                / (np.exp(4j * eta) + r2)
            )
        )
    raise NotImplementedError("Other states not supported yet")


def load(path):
    return np.genfromtxt(path, delimiter=",", skip_header=1).T  # (4, M)


def check_and_abs(arr: np.ndarray):
    assert np.max(np.abs(np.imag(arr))) < 1e-5
    return np.abs(arr)


def construct_data(data, sigma=0.0):
    # Additive white noise
    noise = np.random.normal(0, 1, len(data))
    # Apply a threshold such that values below 0 are set to 0 and values above 1 are set to 1
    noisy_data = data + sigma * noise
    # return np.clip(noisy_data, 0, 1)
    return noisy_data


eps, eta, kap = np.pi / 180, 0.4 * np.pi / 180, 0.2 * np.pi / 180
N = 60
sigma_err = 0.0001

tg = 50
T = 30000
decay_pp_data = check_and_abs(
    [
        get_fidelity(
            "++",
            i,
            eta,
            eps,
            kap,
            np.exp(-tg / 30000),
            np.exp(-tg / 30000),
            np.exp(-tg / 30000),
            np.exp(-tg / 30000),
        )
        for i in range(N)
    ]
)
decay_pm_data = check_and_abs(
    [
        get_fidelity(
            "+-",
            i,
            eta,
            eps,
            kap,
            np.exp(-tg / 30000),
            np.exp(-tg / 30000),
            np.exp(-tg / 30000),
            np.exp(-tg / 30000),
        )
        for i in range(N)
    ]
)
decay_mp_data = check_and_abs(
    [
        get_fidelity(
            "-+",
            i,
            eta,
            eps,
            kap,
            np.exp(-tg / 30000),
            np.exp(-tg / 30000),
            np.exp(-tg / 30000),
            np.exp(-tg / 30000),
        )
        for i in range(N)
    ]
)
decay_mm_data = check_and_abs(
    [
        get_fidelity(
            "--",
            i,
            eta,
            eps,
            kap,
            np.exp(-tg / 30000),
            np.exp(-tg / 30000),
            np.exp(-tg / 30000),
            np.exp(-tg / 30000),
        )
        for i in range(N)
    ]
)


noiseless = load(HERE / "data/decay_noiseless_data.csv")[:, ::1]
R, M = noiseless.shape
L = M // 2
G0_noiseless, G1_noiseless = _hankel_pair(noiseless, L)

plt.semilogy(np.linalg.svdvals(G0_noiseless), label="Noiseless", marker="o")

for sigma_err in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    decay_pp_noisy_data = construct_data(decay_pp_data, sigma=sigma_err)
    decay_pm_noisy_data = construct_data(decay_pm_data, sigma=sigma_err)
    decay_mp_noisy_data = construct_data(decay_mp_data, sigma=sigma_err)
    decay_mm_noisy_data = construct_data(decay_mm_data, sigma=sigma_err)
    noisy = np.stack(
        [
            decay_pp_noisy_data,
            decay_pm_noisy_data,
            decay_mp_noisy_data,
            decay_mm_noisy_data,
        ]
    )
    R, M = noisy.shape
    L = M // 2
    G0_noisy, G1_noisy = _hankel_pair(noisy, L)
    plt.semilogy(np.linalg.svdvals(G0_noisy), label=f"Noisy {sigma_err}", marker="o")

plt.ylabel("Magnitude")
plt.xlabel("#")
plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
plt.title("Hankel matrix H_0 singular value spectrum under noise")
plt.legend()
plt.savefig(HERE / "output/hankel_svd.pdf", bbox_inches="tight")
plt.show()
