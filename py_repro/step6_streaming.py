# -*- coding: utf-8 -*-
r"""
step6_streaming.py —— Step6：流式传输实验（子采样 κ + 质量阈值 ν）
==================================================================
★★★ 本步输入：无文件依赖，直接调流水线生成（内部走 Step0~5 同款逻辑）★★★
★★★ 本步输出：experiments/fig12_streaming.png + streaming_scan.csv ★★★

论文依据
--------
§6.4 拥塞时的降载策略（对应论文 Figure 12）：
  远程传输带宽不够时，接收端必须丢掉一部分事件。丢法有两种：

  ① 均匀随机子采样 (uniform subsampling)：每个事件以概率 κ 保留
        κ ∈ [1.0, 0.8, 0.6, 0.4, 0.2]
     白话：**闭眼随便扔**。扔掉的可能是粒子，也可能是噪声。

  ② 质量分数过滤 (quality filtering)：s = 式(2) 匹配分数（越小越好），
        只保留 s <= ν 的点，ν ∈ [0, 1]
     白话：**挑匹配得好的留、可疑的扔**。ν 越大保留越多（2026-10-21 起与论文同口径）。

论文结论（Fig.12 的故事）：**两种手段用途不同，不是谁替代谁**。
  ① κ 均匀子采样（Fig.11）：论文原话是 raw 点的 Cov/Prec 在 κ 很小时**仍然很高**，
     只有 trajectory 点的 Cov 会降，Prec 全程不变 → 跟踪质量保住，延迟降几个数量级。
  ② ν 质量过滤（Fig.12）：性能随 ν **先陡升后走平**（论文操作点 s≤0.5），作用是减少噪声匹配。

⚠️⚠️ 与论文的口径差异（写报告必须说明，否则会被误判成复现失败）
------------------------------------------------------------------
  ✅ 2026-10-21 起 差异A（横轴方向相反）已消除（审查发现A，销坑#5）：
    pipeline 的 ν 滤波现在直接用式(2) 匹配分数 s（match_and_triangulate 返回），
    保留 s ≤ ν → ν 越大保留越多，与论文 Fig.12 同轴同向。
    （旧版用局部密度代理、方向相反，已废弃。）

  差异B（Cov 掉得比论文凶）：
    论文评的是三角测量出的 **raw 点本身**；我们评的是 **聚类后的簇心**，
    每帧要 ≥2 点才成簇（MIN_POINTS=2）。事件砍掉 80% 后大量帧凑不到 2 点 → 簇直接消失，
    所以本图 κ=0.2 时 Cov 掉到 0.467，比论文的"仍然很高"要惨。
    ⇒ 这是评估对象不同导致的，不是算法写错。

  实测（3P，本仓库）：κ=0.6 → Cov 0.982；ν 保留 60% → Cov 0.903。
  ⇒ 同保留率下 **κ 保召回更好、ν 保准确更好**（ν 全程 Prec=1.000）。

两种调用方式
------------
  1) python step6_streaming.py
  2) from step6_streaming import run_step;  run_step(out_dir)
"""
import argparse
import os

import numpy as np

import _core as K

KAPPAS = [1.0, 0.8, 0.6, 0.4, 0.2]
EXP_PARTICLES = 3       # 用 3 粒子场景（够密、又不至于跑太久）


def run_step(out_dir=None, verbose=True):
    K.banner("Step6", "流式传输：子采样 κ 与质量阈值 ν", "§6.4 降载策略（对应 Fig.12）")

    out_dir = K.ensure_dir(out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))

    cov_k, prec_k, cov_n, prec_n = [], [], [], []
    for kp in KAPPAS:
        m = K.ra.pipeline(EXP_PARTICLES, kappa=kp, seed_offset=EXP_PARTICLES)
        cov_k.append(m["coverage"]); prec_k.append(m["precision"])
        if verbose:
            print(f"[κ={kp:.1f}] Cov={m['coverage']:.3f}  Prec={m['precision']:.3f}")

    for nu in K.C.NU_GRID:
        m = K.ra.pipeline(EXP_PARTICLES, nu=float(nu), seed_offset=EXP_PARTICLES)
        cov_n.append(m["coverage"]); prec_n.append(m["precision"])
        if verbose:
            print(f"[ν={nu:.1f}] Cov={m['coverage']:.3f}  Prec={m['precision']:.3f}")

    with open(os.path.join(out_dir, "streaming_scan.csv"), "w", encoding="utf-8-sig") as f:
        f.write("scheme,param,coverage,precision\n")
        for kp, c, p in zip(KAPPAS, cov_k, prec_k):
            f.write(f"kappa,{kp},{c:.6f},{p:.6f}\n")
        for nu, c, p in zip(K.C.NU_GRID, cov_n, prec_n):
            f.write(f"nu,{nu},{c:.6f},{p:.6f}\n")

    _plot(out_dir, cov_k, prec_k, cov_n, prec_n)
    print(f"[out ] 已保存：{out_dir}\\fig12_streaming.png / streaming_scan.csv")
    return {"kappas": KAPPAS, "cov_k": cov_k, "prec_k": prec_k,
            "nu": list(K.C.NU_GRID), "cov_n": cov_n, "prec_n": prec_n}


def _plot(out_dir, cov_k, prec_k, cov_n, prec_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(KAPPAS, cov_k, "o-", label="Coverage")
    ax[0].plot(KAPPAS, prec_k, "s-", label="Precision")
    ax[0].set_xlabel("subsample fraction κ"); ax[0].set_ylim(0, 1.05)
    ax[0].legend(); ax[0].set_title("Uniform random subsampling (Fig.12 analog)")

    ax[1].plot(K.C.NU_GRID, cov_n, "o-", label="Coverage")
    ax[1].plot(K.C.NU_GRID, prec_n, "s-", label="Precision")
    ax[1].set_xlabel("quality threshold ν"); ax[1].set_ylim(0, 1.05)
    ax[1].legend(); ax[1].set_title("Congestion quality filtering (Fig.12 analog)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig12_streaming.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Step6 流式传输实验")
    ap.add_argument("--out", default=None, help="输出目录（默认 py_repro/experiments）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_step(args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
