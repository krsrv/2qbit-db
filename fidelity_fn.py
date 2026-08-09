import numpy as np


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
            )
        )
    raise ValueError(f"unknown state {state!r}")
