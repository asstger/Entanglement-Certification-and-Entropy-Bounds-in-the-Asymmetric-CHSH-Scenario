# -*- coding: utf-8 -*-
"""
Standalone NPA computation for alpha=0.9 boundary of H(A|E).
Outputs aCHSHalpha-0.9.csv with columns:
  val = S / (2*(1+alpha)),  H(A|E) lower bound
"""
import numpy as np
import ncpol2sdpa as ncp
import chaospy
from sympy.physics.quantum.dagger import Dagger


def binary_entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


class ACHSHSDP:
    def __init__(self, m, verbose=False, parallel=False):
        # Gauss-Radau quadrature nodes/weights on [0,1] with endpoint 1
        t_nodes, w_nodes = chaospy.quadrature.radau(m, chaospy.Uniform(0, 1), 1)
        t_nodes = t_nodes[0]
        self.t = t_nodes[: 2 * m - 1]
        self.ck = [w_nodes[i] / (t_nodes[i] * np.log(2)) for i in range(2 * m - 1)]

        # Measurement operators
        self.A = [Ai for Ai in ncp.generate_measurements([2, 2], 'A')]
        self.B = [Bj for Bj in ncp.generate_measurements([2, 2], 'B')]
        self.Z = ncp.generate_operators('Z', 2, hermitian=False)

        self.sdp = ncp.SdpRelaxation(
            ncp.flatten([self.A, self.B, self.Z]),
            verbose=verbose,
            normalized=True,
            parallel=parallel,
        )

    def get_subs(self):
        subs = {}
        subs.update(ncp.projective_measurement_constraints(self.A, self.B))
        for a in ncp.flatten([self.A, self.B]):
            for z in ncp.flatten(self.Z):
                subs.update({z * a: a * z, Dagger(z) * a: a * Dagger(z)})
        return subs

    def extra_monomials(self):
        monos = []
        z_all = ncp.flatten(self.Z)
        for a in self.A[0]:
            for z in z_all:
                monos += [a * Dagger(z) * z]
        return monos

    def oper_ineq(self, j):
        op_ineq = []
        alpha_j = max(1 / self.t[j], 1 / (1 - self.t[j])) * 3 / 2
        for z in self.Z:
            op_ineq += [alpha_j - z * Dagger(z), alpha_j - Dagger(z) * z]
        return op_ineq

    def obj_j(self, j):
        ma = [self.A[0][0], 1 - self.A[0][0]]
        return sum(
            m_a * (m_z + Dagger(m_z) + (1 - self.t[j]) * Dagger(m_z) * m_z)
            + self.t[j] * m_z * Dagger(m_z)
            for m_a, m_z in zip(ma, self.Z)
        )

    def constr_ineq(self, val, alpha):
        ma0 = 2 * self.A[0][0] - 1
        ma1 = 2 * self.A[1][0] - 1
        mb0 = 2 * self.B[0][0] - 1
        mb1 = 2 * self.B[1][0] - 1
        # val = S / (2*(1+alpha))
        return [
            alpha * ma0 * (mb0 + mb1) / (2 * (alpha + 1))
            + ma1 * (mb0 - mb1) / (2 * (alpha + 1))
            - val
        ]

    def init(self):
        self.sdp.get_relaxation(
            level=2,
            equalities=[],
            inequalities=self.oper_ineq(0),
            substitutions=self.get_subs(),
            extramonomials=self.extra_monomials(),
        )

    def get_hage(self, val, alpha):
        # Apply Bell constraint
        self.sdp.process_constraints(
            equalities=[],
            inequalities=[],
            momentequalities=[],
            momentinequalities=self.constr_ineq(val, alpha),
        )

        # Objective: sum over Gauss-Radau terms
        obj = 0
        for j in range(len(self.t)):
            self.sdp.set_objective(self.obj_j(j))
            self.sdp.solve(solver='mosek')
            if self.sdp.status not in ('optimal', 'optimal_inaccurate'):
                return np.nan, self.sdp.status
            obj += self.ck[j] * self.sdp.primal

        # add constant term c_m
        c_m = sum(self.ck[:-1])
        return c_m + obj, 'optimal'


def main():
    alpha = 0.9
    # val = S / (2*(1+alpha)) in [sqrt(1+alpha^2)/(1+alpha), alpha/(1+alpha)]
    vals = np.linspace(
        (np.sqrt(alpha ** 2 + 1)) / (alpha + 1), alpha / (alpha + 1), 50
    )

    model = ACHSHSDP(m=8, verbose=False, parallel=True)
    model.init()

    hages = []
    statuses = []
    for val in vals:
        hage, status = model.get_hage(val, alpha)
        hages.append(hage)
        statuses.append(status)

    data = np.vstack((vals, hages)).T
    output_path = '0.9.csv'
    np.savetxt(output_path, data, delimiter=',')
    print(f"Saved NPA boundary data to {output_path} (alpha={alpha})")
    if any(s != 'optimal' for s in statuses):
        bad = sum(s != 'optimal' for s in statuses)
        print(f"Warning: {bad} points did not solve optimally.")


if __name__ == '__main__':
    main()
