# -*- coding: utf-8 -*-
"""Use precomputed asym-CHSH data to reproduce get_data_achsh outputs."""
from pathlib import Path

import numpy as np
import qutip as qtp
from scipy.optimize import fsolve


def entropy(pp):
    """Entropy of a distribution pp."""
    ent = 0.0
    for p in pp:
        if 0.0 < p < 1.0:
            ent += -p * np.log2(p)
    return ent


def cond_entropy(pabs, pbs):
    return entropy(pabs) - entropy(pbs)


def correl_2qubit(theta, axs, bys, eta, nv):
    """Correlations <AxBy>, <Ax>, <By> in order [A0B0, A0B1, A1B0, A1B1, A0, A1, B0, B1]."""
    psi = np.cos(theta) * qtp.ket('00') + np.sin(theta) * qtp.ket('11')
    dm = nv * qtp.ket2dm(psi) + (1.0 - nv) * qtp.qeye([2, 2]) / 4.0

    ma, mb = [], []
    for _a in axs:
        ma.append(eta * (np.cos(_a) * qtp.sigmaz() + np.sin(_a) * qtp.sigmax())
                  - (1.0 - eta) * qtp.qeye(2))
    for _b in bys:
        mb.append(eta * (np.cos(_b) * qtp.sigmaz() + np.sin(_b) * qtp.sigmax())
                  - (1.0 - eta) * qtp.qeye(2))

    mean_vals = []
    for _ma in ma:
        for _mb in mb:
            mean_vals.append((dm * qtp.tensor(_ma, _mb)).tr().real)
    for _ma in ma:
        mean_vals.append((dm * qtp.tensor(_ma, qtp.qeye(2))).tr().real)
    for _mb in mb:
        mean_vals.append((dm * qtp.tensor(qtp.qeye(2), _mb)).tr().real)

    return np.array(mean_vals)


def load_asym_chsh_dat(path):
    data = np.genfromtxt(path, delimiter=',')
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expected at least two columns: S, H(A|E).")
    s_vals = data[:, 0]
    hages = data[:, 1]
    return s_vals, hages


def load_asym_chsh_csv_cols(path, s_col=0, h_col=2, skip_header=1):
    data = np.genfromtxt(path, delimiter=',', skip_header=skip_header)
    if data.ndim != 2 or data.shape[1] <= max(s_col, h_col):
        raise ValueError("CSV does not have required columns.")
    return data[:, s_col], data[:, h_col]


def get_data_achsh_from_hages(alpha, vals, hages):
    nus, etas, hagbs_nu, hagbs_eta = [], [], [], []
    mu = np.arctan(1 / alpha)

    def _fun_nu(nu, val):
        correl = correl_2qubit(np.pi / 4, [0.0, np.pi / 2], [mu, -mu], 1.0, nu[0])
        return (alpha / (2 + 2 * alpha)) * (correl[0] + correl[1]) + (1 / (2 + 2 * alpha)) * (
            correl[2] - correl[3]) - val

    def _fun_eta(eta, val):
        correl = correl_2qubit(np.pi / 4, [0.0, np.pi / 2], [mu, -mu], eta[0], 1.0)
        return (alpha / (2 + 2 * alpha)) * (correl[0] + correl[1]) + (1 / (2 + 2 * alpha)) * (
            correl[2] - correl[3]) - val

    for _val in vals:
        nu_best = fsolve(_fun_nu, x0=np.array([1.0]), args=(_val,), xtol=1e-6)[0]
        nus.append(nu_best)
        pabs_nu = (np.ones(4) + np.array([nu_best, -nu_best, -nu_best, nu_best])) / 4
        hagbs_nu.append(cond_entropy(
            pabs_nu, [pabs_nu[0] + pabs_nu[2], pabs_nu[1] + pabs_nu[3]]
        ))

        eta_best = fsolve(_fun_eta, x0=np.array([1.0]), args=(_val,), xtol=1e-6)[0]
        etas.append(eta_best)
        pabs_eta = eta_best ** 2 * np.array([0.5, 0.0, 0.0, 0.5]) \
            + eta_best * (1 - eta_best) * np.array([0.5, 0.0, 0.5, 0.0]) \
            + (1 - eta_best) * eta_best * np.array([0.5, 0.5, 0.0, 0.0]) \
            + (1 - eta_best) ** 2 * np.array([1.0, 0.0, 0.0, 0.0])
        hagbs_eta.append(cond_entropy(
            pabs_eta, [pabs_eta[0] + pabs_eta[2], pabs_eta[1] + pabs_eta[3]]
        ))

    keys_nu = np.array(hages) - np.array(hagbs_nu)
    keys_nu[keys_nu < 0] = 0.0
    keys_eta = np.array(hages) - np.array(hagbs_eta)
    keys_eta[keys_eta < 0] = 0.0

    data = np.vstack((vals, hages, nus, hagbs_nu, keys_nu, etas, hagbs_eta, keys_eta)).T
    return data


def get_data_achsh_from_dat(alpha, dat_path):
    s_vals, hages = load_asym_chsh_dat(dat_path)
    vals = s_vals / (2 * (1 + alpha))
    return get_data_achsh_from_hages(alpha, vals, hages)


def main():
    alpha = 0.95
    csv_path = Path(__file__).resolve().parent.parent / 'asym_chsh_alpha0.95_m8.csv'
    s_vals, hages = load_asym_chsh_csv_cols(csv_path, s_col=0, h_col=1, skip_header=1)
    vals = s_vals / (2 * (1 + alpha))
    data = get_data_achsh_from_hages(alpha, vals, hages)
    out_path = Path(__file__).with_name('noise_alpha0.95_from_csv.csv')
    header = "val,HAgE,nu,HAgB_nu,key_nu,eta,HAgB_eta,key_eta"
    np.savetxt(out_path, data, delimiter=",", header=header, comments="")
    print(f"Saved noise data to {out_path}")


if __name__ == '__main__':
    main()
