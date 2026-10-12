import numpy as np
from scipy.linalg import eigh


def solve_orbitals(F, S):
    """直接求解广义本征值问题 F*C = S*C*epsilon"""
    orbital_energies, C = eigh(F, S)
    return orbital_energies, C


def build_density(C, nocc):
    """由空间轨道构造密度矩阵"""
    C_occ = C[:, :nocc]
    P = 2.0 * C_occ @ C_occ.T
    return P


def build_fock(h, eri, P):
    """由密度矩阵构造 RHF Fock 矩阵"""
    # J_nm = sum_pq (nm|pq) P_pq
    J = np.einsum("nmpq,pq->nm", eri, P)
    # K_nm = sum_pq (np|mq) P_pq
    K = np.einsum("npmq,pq->nm", eri, P)
    F = h + J - 0.5 * K
    return F


def compute_energy(P, h, F, E_nuc):
    """计算当前密度对应的电子能量和总能量"""
    E_elec = 0.5 * np.einsum("nm,nm->", P, h + F)
    E_tot = E_elec + E_nuc
    return E_elec, E_tot


def run_scf(S, h, eri, E_nuc, nelec, max_iter=100,
            energy_tol=1e-10, density_tol=1e-8, verbose=True):
    """求解闭壳层 RHF"""
    nao = h.shape[0]
    if h.shape != (nao, nao) or S.shape != (nao, nao):
        raise ValueError("h 和 S 必须是大小相同的方阵喵")
    if eri.shape != (nao, nao, nao, nao):
        raise ValueError("双电子积分形状必须为 (nao, nao, nao, nao)")
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("闭壳层 RHF 需要正偶数个电子！")
    nocc = nelec // 2
    if nocc > nao:
        raise ValueError("占据轨道数不能超过基函数数喵喵！")
    if max_iter < 1 or min(energy_tol, density_tol) <= 0:
        raise ValueError("迭代次数和收敛阈值必须为正！！")
    if np.linalg.eigvalsh(S).min() <= 1e-10:
        raise ValueError("重叠矩阵存在过小或非正本征值")

    # 初猜：用单电子哈密顿量 h 求轨道，再构造密度和 Fock 矩阵
    _, C = solve_orbitals(h, S)
    P = build_density(C, nocc)
    F = build_fock(h, eri, P)
    _, E_old = compute_energy(P, h, F, E_nuc)
    converged = False

    if verbose:
        print("初猜密度矩阵 P：\n", P)
        print("初猜总能量：", E_old, "Hartree")
        print("\n迭代       总能量 / Hartree         |dE|        ||dP||")

    for iteration in range(1, max_iter + 1):
        _, C = solve_orbitals(F, S)
        P_new = build_density(C, nocc)
        F_new = build_fock(h, eri, P_new)
        E_elec, E_tot = compute_energy(P_new, h, F_new, E_nuc)

        dE = abs(E_tot - E_old)
        dP = np.linalg.norm(P_new - P)
        if verbose:
            print(f"{iteration:4d}  {E_tot:22.12f}  {dE:11.3e}  {dP:11.3e}")

        P, F = P_new, F_new
        if dE < energy_tol and dP < density_tol:
            converged = True
            break
        E_old = E_tot

    if verbose:
        if converged:
            print(f"SCF 在 {iteration} 轮后收敛")
        else:
            print(f"SCF 在 {max_iter} 轮内未收敛，以下仅为最后一轮结果")

    orbital_energies = np.diag(C.T @ F @ C)
    return {
        "converged": converged,
        "iterations": iteration,
        "E_elec": E_elec,
        "E_tot": E_tot,
        "orbital_energies": orbital_energies,
        "C": C,
        "P": P,
        "F": F,
    }
