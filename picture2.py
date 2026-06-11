import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import chaospy
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
from sympy import S

def binary_entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def bar_f(s, alpha):
    sqrt_term = np.sqrt(s ** 2 / 4 - alpha ** 2)
    x = 0.5 * (1 + sqrt_term)
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return 1 - binary_entropy(x)

def f_bd(s, alpha):
    sqrt_term = np.sqrt(s ** 2 / 4 - 1.0)
    x = 0.5 + (1.0 / (2.0 * alpha)) * sqrt_term
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return 1 - binary_entropy(x)


def bar_f_prime(s, alpha):
    sqrt_term = np.sqrt(s ** 2 / 4 - alpha ** 2)
    x = 0.5 * (1 + sqrt_term)
    x = np.clip(x, 1e-12, 1 - 1e-12)
    dx_ds = s / (8 * sqrt_term)
    return -np.log2((1 - x) / x) * dx_ds


def find_s_star(alpha, s_min, s_max, num_points=2000):
    s_grid = np.linspace(s_min + 1e-6, s_max - 1e-6, num_points)
    f_vals = bar_f(s_grid, alpha)
    f_prime = bar_f_prime(s_grid, alpha)
    g_vals = f_vals / (s_grid - s_min) - f_prime
    sign_changes = np.where(np.sign(g_vals[:-1]) != np.sign(g_vals[1:]))[0]
    if sign_changes.size == 0:
        return None
    left_idx = sign_changes[0]
    left = s_grid[left_idx]
    right = s_grid[left_idx + 1]
    for _ in range(60):
        mid = 0.5 * (left + right)
        f_mid = bar_f(mid, alpha)
        f_prime_mid = bar_f_prime(mid, alpha)
        g_mid = f_mid / (mid - s_min) - f_prime_mid
        if np.sign(g_mid) == np.sign(g_vals[left_idx]):
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


def convex_hull_curve(s_vals, alpha):
    s_min = 2.0
    s_max = 2 * np.sqrt(1 + alpha ** 2)
    s_star = find_s_star(alpha, s_min, s_max)
    f_vals = bar_f(s_vals, alpha)
    if s_star is None:
        return f_vals
    k = bar_f_prime(s_star, alpha)
    return np.where(s_vals < s_star, k * (s_vals - s_min), f_vals)


class ACHSHSDP:
    def __init__(self, m, alpha, verbose=False, parallel=False):
        t_nodes, w_nodes = chaospy.quadrature.radau(m - 1, chaospy.Uniform(0, 1), 1)
        self.t_nodes = t_nodes[0]
        self.weights = w_nodes
        self.alpha = alpha

        self.A = [Ai for Ai in ncp.generate_measurements([2, 2], 'A')]
        self.B = [Bj for Bj in ncp.generate_measurements([2, 2], 'B')]
        self.Z = ncp.generate_operators('Z', 2, hermitian=False)

        self.sdp = ncp.SdpRelaxation(
            ncp.flatten([self.A, self.B, self.Z]),
            verbose=verbose,
            normalized=True,
            parallel=parallel,
        )
        self.solver_opts = {
            "intpnt_co_tol_rel_gap": 1e-9,
            "intpnt_tol_pfeas": 1e-9,
            "intpnt_tol_dfeas": 1e-9,
        }

    def get_subs(self):
        subs = {}
        subs.update(ncp.projective_measurement_constraints(self.A, self.B))
        for a in ncp.flatten([self.A, self.B]):
            for z in ncp.flatten(self.Z):
                subs.update({z * a: a * z, Dagger(z) * a: a * Dagger(z)})
        return subs

    def local_monomials_abz(self):
        a_ops = [S.One] + list(ncp.flatten(self.A))
        b_ops = [S.One] + list(ncp.flatten(self.B))
        z_flat = list(ncp.flatten(self.Z))
        z_ops = [S.One] + z_flat + [Dagger(z) for z in z_flat]
        monomials = []
        for a in a_ops:
            for b in b_ops:
                for z in z_ops:
                    monomials.append(a * b * z)
        a0_outcomes = [self.A[0][0], 1 - self.A[0][0]]
        for a0 in a0_outcomes:
            for z in z_flat:
                monomials.append(a0 * Dagger(z) * z)
                monomials.append(a0 * z * Dagger(z))
        return monomials

    def obj_j(self, t_j):
        ma = [self.A[0][0], 1 - self.A[0][0]]
        return sum(
            m_a * (m_z + Dagger(m_z) + (1 - t_j) * Dagger(m_z) * m_z)
            + t_j * m_z * Dagger(m_z)
            for m_a, m_z in zip(ma, self.Z)
        )

    def constr_ineq(self, val):
        ma0 = 2 * self.A[0][0] - 1
        ma1 = 2 * self.A[1][0] - 1
        mb0 = 2 * self.B[0][0] - 1
        mb1 = 2 * self.B[1][0] - 1
        return [
            self.alpha * ma0 * (mb0 + mb1) / (2 * (self.alpha + 1))
            + ma1 * (mb0 - mb1) / (2 * (self.alpha + 1))
            - val
        ]

    def init(self):
        local_monos = self.local_monomials_abz()
        self.sdp.get_relaxation(
            level=2,
            equalities=[],
            inequalities=[],
            substitutions=self.get_subs(),
            momentinequalities=self.constr_ineq(0.0),
            objective=self.obj_j(self.t_nodes[0]),
            extramonomials=local_monos,
        )

    def get_hage(self, val):
        self.sdp.process_constraints(
            equalities=[],
            inequalities=[],
            momentequalities=[],
            momentinequalities=self.constr_ineq(val),
        )

        hage = 0.0
        num_terms = len(self.t_nodes) - 1
        for j in range(num_terms):
            t_j = self.t_nodes[j]
            w_j = self.weights[j]
            self.sdp.set_objective(self.obj_j(t_j))
            self.sdp.solve(solver='mosek', solverparameters=self.solver_opts)
            if self.sdp.status not in ('optimal', 'optimal_inaccurate'):
                return np.nan
            hage += (w_j / (t_j * np.log(2))) * (1 + self.sdp.dual)
        return hage


def compute_curve(alpha, m=8, n_points=30):
    s_min = 2.0
    s_max = 2 * np.sqrt(1 + alpha ** 2)
    s_vals = np.linspace(s_min, s_max, n_points)
    val_vals = s_vals / (2 * (1 + alpha))

    model = ACHSHSDP(m=m, alpha=alpha, verbose=False, parallel=True)
    model.init()

    h_vals = []
    for val in val_vals:
        h_vals.append(model.get_hage(val))
    return s_vals, np.array(h_vals)


def achsh(ax):
    s_10, h_10 = compute_curve(alpha=1.0, m=8)
    s_09, h_09 = compute_curve(alpha=0.9, m=8)
    s_08, h_08 = compute_curve(alpha=0.8, m=8)

    np.savetxt('1-1.0.csv', np.column_stack([s_10, h_10]), delimiter=',', fmt='%.7f')
    np.savetxt('1-0.9.csv', np.column_stack([s_09, h_09]), delimiter=',', fmt='%.7f')
    np.savetxt('1-0.8.csv', np.column_stack([s_08, h_08]), delimiter=',', fmt='%.7f')

    ax.plot(s_10, h_10, label=r'NPA $\alpha=1.0$',
            linewidth=2, color='green', linestyle='solid', alpha=0.9)
    ax.plot(s_09, h_09, label=r'NPA $\alpha=0.9$',
            linewidth=2, color='pink', linestyle='solid', alpha=0.9)
    ax.plot(s_08, h_08, label=r'NPA $\alpha=0.8$',
            linewidth=2, color='red', linestyle='solid', alpha=0.9)

    s_vals_10 = np.linspace(2.0, 2 * np.sqrt(1 + 1.0 ** 2), 200)
    s_vals_09 = np.linspace(2.0, 2 * np.sqrt(1 + 0.9 ** 2), 200)
    s_vals_08 = np.linspace(2.0, 2 * np.sqrt(1 + 0.8 ** 2), 200)
    ax.plot(s_vals_10, convex_hull_curve(s_vals_10, 1.0),
            linewidth=1.5, color='green', linestyle='--', alpha=0.9,
            label=r'Convex hull $\alpha=1.0$')
    ax.plot(s_vals_09, convex_hull_curve(s_vals_09, 0.9),
            linewidth=1.5, color='pink', linestyle='--', alpha=0.9,
            label=r'Convex hull $\alpha=0.9$')
    ax.plot(s_vals_08, convex_hull_curve(s_vals_08, 0.8),
            linewidth=1.5, color='red', linestyle='--', alpha=0.9,
            label=r'Convex hull $\alpha=0.8$')

    s_bd = np.linspace(2.0, 2.7, 200)
    # ax.plot(s_bd, f_bd(s_bd, 1.0), linewidth=1.2, color='green', linestyle='-.', alpha=0.8,
    #         label=r'$F_{\mathrm{BD}}$ $\alpha=1.0$')
    # ax.plot(s_bd, f_bd(s_bd, 0.9), linewidth=1.2, color='pink', linestyle='-.', alpha=0.8,
    #         label=r'$F_{\mathrm{BD}}$ $\alpha=0.9$')
    # ax.plot(s_bd, f_bd(s_bd, 0.8), linewidth=1.2, color='red', linestyle='-.', alpha=0.8,
    #         label=r'$F_{\mathrm{BD}}$ $\alpha=0.8$')

    s_star_10 = find_s_star(1.0, 2.0, 2 * np.sqrt(1 + 1.0 ** 2))
    s_star_09 = find_s_star(0.9, 2.0, 2 * np.sqrt(1 + 0.9 ** 2))
    s_star_08 = find_s_star(0.8, 2.0, 2 * np.sqrt(1 + 0.8 ** 2))
    if s_star_10 is not None:
        ax.scatter([s_star_10], [bar_f(s_star_10, 1.0)], color='green', s=30, zorder=5,
                   label=r'$S^*$ $\alpha=1.0$')
    if s_star_09 is not None:
        ax.scatter([s_star_09], [bar_f(s_star_09, 0.9)], color='pink', s=30, zorder=5,
                   label=r'$S^*$ $\alpha=0.9$')
    if s_star_08 is not None:
        ax.scatter([s_star_08], [bar_f(s_star_08, 0.8)], color='red', s=30, zorder=5,
                   label=r'$S^*$ $\alpha=0.8$')
    hds, lbs = ax.get_legend_handles_labels()
    ax.legend(hds[::-1], lbs[::-1], fontsize=10, fancybox=True)
    ax.set_xlabel(r'Bell-inequality violation', fontsize=15)
    ax.set_ylabel(r'Bits ', fontsize=15)
    ax.set_xlim([2.0, 3.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks(np.arange(2.0, 3.0 + 0.001, 0.1))
    ax.set_yticks(np.arange(0.0, 1.0 + 0.001, 0.1))
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(7, 5))  # 创建单独的一个子图
    achsh(ax)  # 将 ax 传递给 achsh 函数
    plt.tight_layout()  # 自动调整布局
    plt.savefig('Code_DIQKD_MerminPeresGame-master/figure/picture2.png', dpi=300)
    plt.show()
