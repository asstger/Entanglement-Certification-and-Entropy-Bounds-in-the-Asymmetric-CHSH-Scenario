

import ncpol2sdpa as ncp
import pandas as pd
from sympy.physics.quantum.dagger import Dagger
import chaospy
from Mod_Func import *
import numpy as np
from scipy.optimize import fsolve
from itertools import product

class aCHSH_SDP:
    def __init__(self, m, verbose=1, parallel=False):
        tt, ww = chaospy.quad_gauss_radau(m, chaospy.Uniform(0, 1), 1)
        self.M = 2 * m
        self.T = tt[0][:m * 2 - 1]
        self.CK = [ww[i] / (tt[0][i] * np.log(2)) for i in range(m * 2 - 1)]

        self.A = [Ai for Ai in ncp.generate_measurements([2, 2], 'A')]
        self.B = [Bj for Bj in ncp.generate_measurements([2, 2], 'B')]

        self.Z = ncp.generate_operators('Z', 2, hermitian=False)

        self.SDP = ncp.SdpRelaxation(ncp.flatten([self.A, self.B, self.Z]),
                                     verbose=verbose, normalized=True, parallel=parallel)

    def obj_j(self, j):
        mas = [self.A[0][0], 1 - self.A[0][0]]

        return sum(ma * (mz + Dagger(mz) + (1 - self.T[j]) * Dagger(mz) * mz)
                   + self.T[j] * mz * Dagger(mz)
                   for ma, mz in zip(mas, self.Z)
                   )

    def get_subs(self):
        subs = {}
        subs.update(ncp.projective_measurement_constraints(self.A, self.B))
        for a, z in product(ncp.flatten([self.A, self.B]), ncp.flatten(self.Z)):
            subs.update({z * a: a * z, Dagger(z) * a: a * Dagger(z)})

        return subs

    def extra_monomials(self):
        monos = []
        z_all = ncp.flatten(self.Z)
        for a, z in product(self.A[0], z_all):
            monos += [a * Dagger(z) * z]
        """
        a_all = ncp.flatten(self.A)
        b_all = ncp.flatten(self.B)
        for a, b, z in product(a_all, b_all, z_all):
            monos += [a * b * z, a * b * Dagger(z)]"""

        return monos

    def oper_ineq(self, j):
        op_ineq = []
        alpj = max(1 / self.T[j], 1 / (1 - self.T[j])) * 3 / 2
        for z in self.Z:
            op_ineq += [alpj - z * Dagger(z), alpj - Dagger(z) * z]

        return op_ineq




    def constr_ineq(self, val, alpha):
        ma0 = 2 * self.A[0][0] - 1
        ma1 = 2 * self.A[1][0] - 1
        mb0 = 2 * self.B[0][0] - 1
        mb1 = 2 * self.B[1][0] - 1
        #return [(1 - eps) * ma0 * (mb0 + mb1) / 2 + eps * ma1 * (mb0 - mb1) / 2 - val]
        return [alpha * ma0 * (mb0 + mb1) / (2 * (alpha + 1)) + ma1 * (mb0 - mb1) / (2 * (alpha + 1)) - val]


    def init(self):
        self.SDP.get_relaxation(level=2,
                                equalities=[],
                                inequalities=[],
                                momentequalities=[],
                                momentinequalities=self.constr_ineq(0, 0.9),
                                objective=self.obj_j(0),
                                substitutions=self.get_subs(),
                                extramonomials=self.extra_monomials())
        print(f'SDP is initialized.')

    def get_hage(self):
        hage = sum(self.CK)
        print('hage',hage)
        for j in range(self.M - 1):
            self.SDP.set_objective(self.obj_j(j))
            self.SDP.solve(solver='mosek')
            if self.SDP.status == 'optimal':
                hage += self.CK[j] * self.SDP.dual
                # print('ckj',self.CK[j])
                # print(j, self.SDP.status, self.SDP.primal, self.SDP.dual, hage)
                # print('obj_j', self.SDP.__getitem__(self.obj_j(j)))
                # print("SDP x_mat:", self.SDP.x_mat)  # The primal solution matrix.
                # print("mon_set",self.SDP.monomial_sets)
                # print("mon_index",self.SDP.monomial_index)
                # print("constraints",self.SDP.constraints)
                # print('t',self.T)
                data_s = self.SDP.x_mat

                # # 只保留第一个矩阵
                # if isinstance(data_s, (list, tuple)):
                #     data_s = data_s[0]
                #
                # # 分析数据结构
                # print(f"x_mat类型: {type(data_s)}")
                # if isinstance(data_s, (list, tuple, np.ndarray)):
                #     print(f"x_mat形状: {np.array(data_s).shape}")
                #
                # # 尝试保存数据
                # try:
                #     # 如果是稀疏矩阵，转换为密集矩阵
                #     if hasattr(data_s, 'todense'):
                #         data_array = data_s.todense()
                #     else:
                #         # 尝试直接转换为numpy数组
                #         data_array = np.array(data_s)
                #
                #     # 保存数据
                #     if len(data_array.shape) == 2:
                #         np.savetxt(f'x_mat_j{j}_91.csv', data_array, delimiter=',', fmt='%.6f')
                #     else:
                #         # 如果是一维数组，保存为单列
                #         np.savetxt(f'x_mat_j{j}_91.csv', data_array.reshape(-1, 1), delimiter=',', fmt='%.6f')
                #
                #     print(f'已保存x_mat到x_mat_j{j}_91.csv')
                # except Exception as e:
                #     print(f"保存x_mat时出错: {e}")
                #     print(f"尝试保存原始数据...")
                #     # 如果转换失败，尝试保存原始数据
                #     with open(f'x_mat_j{j}_raw.txt', 'w') as f:
                #         f.write(str(data_s))
                #     print(f'已将原始数据保存到x_mat_j{j}_1_raw.txt')


            else:
                hage = 0.
                print(j, self.SDP.status, self.SDP.primal, self.SDP.dual, hage)
                break

        return hage


def get_hages(alpha, model):
    #vals = np.linspace(np.sqrt(1 - 2 * eps + 2 * eps ** 2), 1 - eps, 10)
    vals = np.linspace((np.sqrt(alpha ** 2 + 1)) / (alpha+1),alpha/ (alpha+1), 50)
    # vals = np.linspace((np.sqrt(alpha ** 2 + 1)) / 2, 1 / 2,1)
    hages = []
    for val in vals:
        model.SDP.process_constraints(equalities=[],
                                      inequalities=[],
                                      momentequalities=[],
                                      momentinequalities=model.constr_ineq(val, alpha))
        hages += [model.get_hage()]


    return hages, vals


def get_data_achsh(alpha, model):
    hages, vals = get_hages(alpha=alpha, model=model)


    data = np.vstack((vals, hages)).T

    return data


if __name__ == '__main__':
    Model = aCHSH_SDP(8, verbose=False, parallel=True)
    Model.init()

    Data5 = get_data_achsh(1.1, Model)
    np.savetxt('aCHSHalpha-1.1.csv', Data5, delimiter=',')




