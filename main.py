import argparse
from pathlib import Path
import tomllib

import numpy as np
from pyscf import gto

from hf import run_scf

DEFAULT_INPUT = Path(__file__).with_name("input.toml")


def load_input(path):
    """读取分子和 SCF 参数；检查拼写，避免没有设置"""
    with Path(path).open("rb") as file:
        settings = tomllib.load(file)

    allowed = {
        "molecule": {"atom", "unit", "basis", "charge", "spin"},
        "scf": {"max_iter", "energy_tol", "density_tol"},
    }
    for section, values in settings.items():
        if section not in allowed or not isinstance(values, dict):
            raise ValueError(f"未知或格式不正确的输入分区：{section}")
        unknown = values.keys() - allowed[section]
        if unknown:
            raise ValueError(f"[{section}] 中有未知参数：{', '.join(sorted(unknown))}")
    missing = allowed["molecule"] - settings.get("molecule", {}).keys()
    if missing:
        raise ValueError(f"[molecule] 缺少参数：{', '.join(sorted(missing))}")
    return settings


def build_molecule(settings):
    """用输入文件中的 [molecule] 参数构造分子"""
    return gto.M(
        atom=settings["atom"],
        unit=settings["unit"],
        basis=settings["basis"],
        charge=settings["charge"],
        spin=settings["spin"],
        verbose=0,
    )


def get_integrals(mol):
    """准备积分"""
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    h = T + V
    eri = mol.intor("int2e", aosym="s1")
    E_nuc = mol.energy_nuc()
    return S, h, eri, E_nuc


def main(input_path=DEFAULT_INPUT):
    settings = load_input(input_path)
    mol = build_molecule(settings["molecule"])
    if mol.spin != 0:
        raise ValueError("本程序仅处理闭壳层 RHF，请设置 spin=0")
    S, h, eri, E_nuc = get_integrals(mol)

    print("输入文件：", Path(input_path).resolve())
    print("电子数：", mol.nelectron)
    print("基函数数：", mol.nao_nr())
    print("重叠矩阵 S：\n", S)
    print("单电子矩阵 h：\n", h)
    print("双电子积分数组形状：", eri.shape)
    print("核排斥能：", E_nuc, "Hartree")

    result = run_scf(S, h, eri, E_nuc, mol.nelectron, **settings.get("scf", {}))

    if result["converged"]:
        print("\n收敛结果")
        print("轨道能量：", result["orbital_energies"], "Hartree")
    else:
        print("\n最后一轮结果（未收敛）")
        print("当前轨道中 F 的对角元：", result["orbital_energies"], "Hartree")
    print("电子能：", result["E_elec"], "Hartree")
    print("核排斥能：", E_nuc, "Hartree")
    print("总能量：", result["E_tot"], "Hartree")
    print("轨道系数 C：\n", result["C"])
    print("密度矩阵 P：\n", result["P"])
    print("Fock 矩阵 F：\n", result["F"])
    print("电子数 Tr(PS)：", np.trace(result["P"] @ S))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 TOML 输入文件运行闭壳层 RHF")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                        help="输入文件路径，默认使用 main.py 同目录下的 input.toml")
    args = parser.parse_args()
    try:
        main(args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))
