# Hartree–Fock 
通过此程序可逐步实现限制性 Hartree–Fock（RHF）方程求解与自洽场（SCF）迭代。
本项目参考 [hf-tutorial](https://github.com/yangjunjie0320/hf-tutorial) 
的思路，用 NumPy 和 SciPy 实现闭壳层 RHF 的 SCF 迭代，用 PySCF 构造分子并计算原子轨道积分。正式计算中的密度矩阵、Fock 矩阵、能量和迭代循环由本项目代码计算，PySCF 只负责提供积分和参考值。

当前默认体系是 **HeH⁺ / STO-3G**：He–H 距离为 **1.5 Bohr**，2 个电子，总电荷为+1。程序计算给定核坐标下的单点能，不进行几何优化。

## 环境配置


使用 Python 3.11 或更新版本（输入文件通过标准库 `tomllib` 读取），在项目目录中运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.in` 列出直接依赖，`requirements.txt` 记录本次验证环境的精确版本。

*本项目的开发环境与依赖配置由 OpenAl Codex辅助完成。*



## 运行

激活项目环境后，在项目目录运行默认的 HeH⁺ 输入：

```bash
python main.py
```

不指定文件时，读取 `main.py` 同目录下的 `input.toml`。也可以指定另一个输入文件，例如三核的水分子：

```bash
python main.py examples/h2o.toml
```

另外两个输入示例：

```bash
python main.py examples/h3_plus.toml
python main.py examples/h4.toml
```

| 输入文件 | 体系 | 电荷 | 电子数 | 基组 |
| --- | --- | --- | --- | --- |
| `examples/h3_plus.toml` | H₃⁺，边长 1.65 Bohr 的等边三角形 | +1 | 2 | STO-3G |
| `examples/h4.toml` | H₄，相邻间距 1.4 Bohr 的线性氢链 | 0 | 4 | STO-3G |

两个示例均设置 `spin=0`，进行闭壳层 RHF 单点计算。坐标是指定的示例几何，未经过优化；H₄ 用作氢链模型，不表示稳定的平衡结构。

显式指定的相对路径以终端当前目录为准。程序会打印所读取文件的完整路径。

输出包含电子数、基函数数、积分矩阵、初猜密度、每轮总能量及变化量，以及最终的轨道系数、密度矩阵和 Fock 矩阵。能量单位为 Hartree。

## 不同体系的计算设置

修改或复制 `input.toml` 即可，无需修改 Python 源码。输入采用 TOML 格式：`[molecule]` 是分子设置，`[scf]` 是迭代设置，`#` 后可以写注释。

`[molecule]` 中的五个参数都必须提供。`[scf]` 可以省略，也可以只填写想修改的参数；未填写时使用 `run_scf()` 的默认值：最多 100 轮，能量阈值 `1e-10` Hartree，密度变化范数阈值 `1e-8`。未知参数会报错，方便发现拼写问题。

默认 `input.toml` 为 HeH⁺。程序根据输入自动取得电子数和基函数数，SCF 函数不需要随原子数量改写。

| 输入 | 含义 | 更换时需要注意 |
| --- | --- | --- |
| `atom` | 每个原子的元素与三维坐标 | 三个原子写三行；更多原子继续添加 |
| `unit` | 输入坐标的长度单位 | 默认例子中是 `Bohr`，水的例子是 `Angstrom`，不能只换坐标而忽略单位 |
| `basis` | 每个原子采用的基组 | 更大的基组会增加基函数数、积分数量和计算开销 |
| `charge` | 整个体系的总电荷 | 全电子计算中，电子数为所有核电荷之和减去总电荷 |
| `spin` | PySCF 中的 $N_\alpha-N_\beta$ | 本程序要求为 0；它不是自旋多重度 $2S+1$ |

输入约定可参阅 [PySCF 分子定义文档](https://pyscf.org/user/gto.html)。当前实现要求正偶数个电子并采用双占据轨道；仅有偶数电子和 `spin=0`，并不能保证一个实际体系适合闭壳层描述。


## 计算原理与步骤

本程序采用闭壳层 RHF：固定原子核，每个占据空间轨道放两个自旋相反的电子。先给出一个电子分布，再求它对应的轨道和新电子分布，反复计算直到自洽，这个过程叫 SCF。

### 1. 确定分子和基组

从输入文件读取原子坐标、电荷、自旋和基组，得到电子数 $N$ 和基函数数。闭壳层占据轨道数为 $N/2$。

### 2. 准备积分

用 PySCF 计算重叠矩阵 $S$、动能矩阵 $T$、电子–核吸引矩阵 $V$、双电子积分 `eri` 和核排斥能 $E_{\mathrm{nuc}}$。其中单电子矩阵为 $h=T+V$。

### 3. 构造初猜密度

先只用 $h$ 求解轨道，得到轨道系数 $C$。选取能量最低的 $N/2$ 个轨道，构造初猜密度矩阵：

$$
P=2C_{\mathrm{occ}}C_{\mathrm{occ}}^{\mathrm T}.
$$

$C_{\mathrm{occ}}$ 是占据轨道的系数，系数 2 表示每个轨道有两个电子。

### 4. 构造 Fock 矩阵

用密度矩阵和双电子积分计算库仑矩阵 $J$、交换矩阵 $K$，再加上单电子部分：

$$
F=h+J-\frac12K.
$$

$J$ 描述电子间的库仑作用，$K$ 对应交换作用。

### 5. 求解新的分子轨道

求解 Roothaan方程：

$$
FC=SC\varepsilon.
$$

得到新的轨道系数 $C$ 和轨道能量 $\varepsilon$。因为基函数一般不正交，方程中有重叠矩阵 $S$。程序用 `eigh(F, S)` 完成这一步。

### 6. 更新密度并计算能量

用新的占据轨道重新计算 $P$，再用新密度重建 $F$。用相互对应的 $P$ 和 $F$ 计算总能量：

$$
E_{\mathrm{tot}}=\frac12\sum_{nm}P_{nm}(h_{nm}+F_{nm})+E_{\mathrm{nuc}}.
$$

$n,m$ 是基函数下标。这个表达式避免重复计算电子间的相互作用，并加上核排斥能。初猜能量也按同一公式计算。

### 7. 判断是否收敛

比较本轮与上一轮的总能量、密度矩阵。默认要求能量变化小于 $10^{-10}$ Hartree，密度变化小于 $10^{-8}$，两项同时满足任务收敛。

未收敛则用更新后的 $F$ 回到第 5 步继续求解，最多计算 100 轮。

### 8. 输出结果

收敛后输出总能量、轨道能量、轨道系数、密度矩阵和 Fock 矩阵。如果达到迭代上限仍未收敛，程序会明确提示，输出仅代表最后一轮结果。

## 程序框架

```text
Hartree-Fock/
├── input.toml          默认 HeH⁺ 分子与 SCF 参数
├── examples/
│   └── h2o.toml        水分子输入示例
├── main.py             读取输入、积分准备、调用求解器、打印结果
├── hf.py               RHF 的矩阵运算与 SCF 迭代
├── requirements.in     直接依赖
├── requirements.txt    已记录的依赖版本
├── .vscode/            编辑器配置
├── .gitignore          排除虚拟环境、缓存等本地文件
└── README.md           使用方法、理论与框架
```

关于 SCF、初猜和收敛方法，可进一步阅读 [PySCF SCF 文档](https://pyscf.org/user/scf.html)。

## 学习参考
- [Szabo & Ostlund 教材](https://github.com/nominhanggai/szaboqc)Szabo and Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory, Dover Publications, New York, 1996
- [szabo汉化版](https://github.com/nominhanggai/szaboqc)
- [hf-tutorial](https://github.com/yangjunjie0320/hf-tutorial)：大头！:)
- [PySCF 文档](https://pyscf.org/user/gto.html)
- [SciPy 广义本征值求解](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html)
- [Hartree-Fock理论：从行列式到代码计算](https://www.bilibili.com/video/BV1VYhX6JE58/?share_source=copy_web&vd_source=18fded4d54c675e3ac39cda991f179f0) 一个很好的介绍视频，同系列其他视频也很好看。
## 写在后面
Hartree-Fock作为理论计算的hello world，拖拖沓沓今天才开始写qwq。大约一年前就想着要写，但对Szabo望而却步。如今对各家的旗舰模型“许愿”一句“给我写一个非限制性Hartree-Fock程序”，都能得到不错的结果。为什么还要写？我想肯定和之前不是一个写法，这个程序的环境依赖，检查改正和整理，以及输入文件的生成都有ai的参与。当然纯许愿就没意思了。作为非cs专业的学生，至少学起来、写起来有个即时的辅助工具可以使用会轻松不少。
最后，感谢理论计算化学自学答疑群内外给予我各种帮助和指导的各位老师、同学(^_^)。特别感谢[步允霆学长](https://github.com/Yunting-Bu)和[前前群主大头](https://github.com/yangjunjie0320)。

***hello world, hello quantum chemistry( ^ω^ )***
