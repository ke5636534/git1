# -*- coding: utf-8 -*-
r"""
step7_experiments.py —— Step7：主实验（粒子数扫描，论文 Figure 6 复刻）
==================================================================
★★★ 本步输入：无文件依赖，内部跑完整流水线（Step0~5 同款逻辑）★★★
★★★ 本步输出：experiments/fig6_particle_count.png + particle_count.csv ★★★

论文依据
--------
§5 主实验（对应论文 Figure 6）：把盘上的粒子数从 1 加到 10，看精度怎么变。
  粒子数 ∈ PARTICLE_COUNTS = [1, 2, 3, 5, 10]
  每个粒子数跑完整链路：生成 -> 去噪 -> 匹配三角测量 -> 聚类 -> 评估

论文想证明的核心结论：
  **粒子变多（事件密度暴涨）时，事件驱动的重建精度基本不掉** ——
  这是"事件相机 + 事件级匹配"相对"帧相机 + 图像处理"的根本优势：
  帧方案的算力随粒子数线性涨（每帧都要处理整幅图），事件方案只处理"变化的像素"。

读图方法：
  左图两根柱子（Cov / Prec）应该几乎齐平在 0.98+；
  右图 Chamfer 应该在 5~9 mm 这个量级小幅波动（不是随粒子数单调爆炸）。
  如果 Chamfer 随粒子数飙到几十厘米，说明匹配关卡太松、串了别的粒子。

两种调用方式
------------
  1) python step7_experiments.py
  2) from step7_experiments import run_step;  run_step(out_dir)
"""
import argparse
import os

import numpy as np

import _core as K


def run_step(out_dir=None, verbose=True):
    K.banner("Step7", "主实验：粒子数扫描 1/2/3/5/10P", "§5 主实验（对应 Fig.6）")

    out_dir = K.ensure_dir(out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))

    rows = []
    for n in K.C.PARTICLE_COUNTS:
        m = K.ra.pipeline(n, seed_offset=n)
        rows.append((n, m["coverage"], m["precision"], m["chamfer"], m["n_recon"]))
        if verbose:
            print(f"[{n:>2}P] Cov={m['coverage']:.4f}  Prec={m['precision']:.4f}  "
                  f"Chamfer={m['chamfer'] * 1000:5.2f} mm  recon={m['n_recon']}")

    with open(os.path.join(out_dir, "particle_count.csv"), "w", encoding="utf-8-sig") as f:
        f.write("particles,coverage,precision,chamfer_m,n_recon\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]}\n")

    _plot(rows, out_dir)
    print(f"[out ] 已保存：{out_dir}\\fig6_particle_count.png / particle_count.csv")
    return rows


def _plot(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = [r[0] for r in rows]
    cov = [r[1] for r in rows]
    prec = [r[2] for r in rows]
    cham = [r[3] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(counts))
    ax[0].bar(x - 0.2, cov, 0.4, label="Coverage")
    ax[0].bar(x + 0.2, prec, 0.4, label="Precision")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{c}P" for c in counts])
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("ratio"); ax[0].legend()
    ax[0].set_title("Coverage / Precision vs particle count (Fig.6 analog)")

    ax[1].plot(counts, cham, "o-", color="#d1495b")
    ax[1].set_xlabel("particle count"); ax[1].set_ylabel("Chamfer (m)")
    ax[1].set_title("Chamfer distance vs particle count")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_particle_count.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Step7 粒子数扫描主实验")
    ap.add_argument("--out", default=None, help="输出目录（默认 py_repro/experiments）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_step(args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
