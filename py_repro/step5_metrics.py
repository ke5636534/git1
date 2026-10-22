# -*- coding: utf-8 -*-
r"""
step5_metrics.py —— Step5：精度评估（Cov / Prec / Chamfer，论文式3/4/7）
==================================================================
★★★ 本步输入：data/<名>/cluster_centers.npz（Step3）+ gt.npz（Step0）★★★
★★★ 本步输出：metrics.json / metrics.csv + fig_cov_prec.png ★★★

论文依据
--------
§6.1 定义三个指标（O=真值点集，R=重建点集，ρ=5 cm 容差球）：

  式(3) 覆盖率 Coverage： Cov(ρ) = |{o∈O : d(o,R) <= ρ}| / |O|
         白话：**真值里有多少粒子被找到了**（召回，越高越好）

  式(4) 精确率 Precision：Prec(ρ) = |{r∈R : d(r,O) <= ρ}| / |R|
         白话：**我报出来的点里有多少是真的**（准确，越高越好）

  式(5)(6) d_NN：真值点到重建点集的平均最近邻距离（论文里还有 d_C 版本）
         白话：平均"差多远"，越小越好

  式(7) 倒角距离 Chamfer： d_C = ½·( mean_{o∈O} d(o,R) + mean_{r∈R} d(r,O) )
         白话：**双向**平均距离，越小越好。★ 必须双向！
         只算单向（真值→重建）会漏掉"凭空多出来的点"，数值会虚低。

★ 评估对象的选择（与论文口径的差异，写进报告时要说明）：
  论文用"每帧重建点集"评 Cov/Prec；本项目用 Step3 的**簇心**（=每帧重建点集），
  不用 Step4 的轨迹点。原因见 step4_track.py 抬头说明：简化检测器太稀疏，
  连续漏检 12 步会把轨迹删光，用轨迹点评会让指标虚低甚至为 0。

两种调用方式
------------
  1) python step5_metrics.py --in "py_repro\data\sim1p"
  2) from step5_metrics import run_step;  run_step(dataset_dir)
"""
import argparse
import json
import os

import numpy as np

import _core as K


def run_step(dataset_dir, verbose=True):
    K.banner("Step5", "精度评估 Cov / Prec / Chamfer", "§6.1 式(3)覆盖率、式(4)精确率、式(7)倒角距离")

    cpath = os.path.join(dataset_dir, "cluster_centers.npz")
    if not os.path.exists(cpath):
        raise SystemExit(f"[×] 找不到 {cpath}\n    请先跑 Step3：python step3_cluster.py --in {dataset_dir}")
    centers = np.load(cpath)["centers"]
    recon = centers[:, 1:]                                   # 去掉时间列 -> (K,3)

    gt = K.load_gt(dataset_dir)
    if gt is None or len(gt) == 0:
        print("[!] 这个数据集没有真值（gt.npz 为空）——真实采集数据没有 OptiTrack 真值，")
        print("    只能看重建了多少点，算不了 Cov/Prec/Chamfer。")
        print(f"[out ] 重建点 {len(recon)} 个；跳过指标。")
        return {"coverage": None, "precision": None, "chamfer": None,
                "d_nn_gt2recon": None, "d_nn_recon2gt": None,
                "n_recon": len(recon), "n_gt": 0}

    m = K.metrics.evaluate(recon, gt[:, 1:])

    with open(os.path.join(dataset_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({**m, "rho": K.C.RHO}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(dataset_dir, "metrics.csv"), "w", encoding="utf-8-sig") as f:
        f.write("metric,value\n")
        f.write(f"coverage,{m['coverage']:.6f}\n")
        f.write(f"precision,{m['precision']:.6f}\n")
        f.write(f"chamfer_m,{m['chamfer']:.6f}\n")
        f.write(f"d_nn_gt2recon_m,{m['d_nn_gt2recon']:.6f}\n")
        f.write(f"d_nn_recon2gt_m,{m['d_nn_recon2gt']:.6f}\n")
        f.write(f"n_recon,{m['n_recon']}\n")
        f.write(f"n_gt,{m['n_gt']}\n")

    if verbose:
        print(f"[metrics] 容差半径 ρ = {K.C.RHO * 100:.0f} cm")
        print(f"[metrics] 重建点 {m['n_recon']} / 真值点 {m['n_gt']}")
        print(f"[metrics] Coverage  (Cov)  = {m['coverage']:.4f}   ← 真值里被找到的比例（越高越好）")
        print(f"[metrics] Precision (Prec) = {m['precision']:.4f}   ← 报出来的点里真的比例（越高越好）")
        print(f"[metrics] Chamfer          = {m['chamfer']:.6f} m ({m['chamfer'] * 1000:.2f} mm)  ← 双向平均距离（越小越好）")
        print(f"[metrics] d_NN(O->R) 式(5) = {m['d_nn_gt2recon']:.6f} m ({m['d_nn_gt2recon'] * 1000:.2f} mm)  ← 真值点->重建点集的平均最近邻（越小越好）")
        print(f"[metrics] d_NN(R->O) 式(6) = {m['d_nn_recon2gt']:.6f} m ({m['d_nn_recon2gt'] * 1000:.2f} mm)  ← 重建点->真值点集的平均最近邻（越小越好）")
        print(f"[metrics] 校验 ½(式5+式6)   = {0.5 * (m['d_nn_gt2recon'] + m['d_nn_recon2gt']):.6f} m  ← 应逐位等于 Chamfer（式7）")
        print(f"[out    ] 已保存：metrics.json / metrics.csv")

    _plot(recon, gt[:, 1:], m, dataset_dir)
    return m


def _plot(recon, gt, m, dataset_dir):
    """左：XZ 俯视叠图（真值 vs 重建）；右：三个指标数字牌。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(gt[:, 0], gt[:, 2], s=4, c="#c9c9c9", label=f"ground truth ({len(gt)})")
    ax[0].scatter(recon[:, 0], recon[:, 2], s=6, c="#d1495b", label=f"reconstructed ({len(recon)})")
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Z (m)")
    ax[0].set_title("XZ view: reconstruction vs ground truth")
    ax[0].legend(fontsize=8)

    ax[1].axis("off")
    txt = (f"Coverage  = {m['coverage']:.4f}\n"
           f"Precision = {m['precision']:.4f}\n"
           f"Chamfer   = {m['chamfer'] * 1000:.2f} mm\n"
           f"\nrho = {K.C.RHO * 100:.0f} cm\n"
           f"recon={m['n_recon']} / gt={m['n_gt']}")
    ax[1].text(0.5, 0.5, txt, ha="center", va="center", fontsize=15,
               family="monospace", bbox=dict(boxstyle="round,pad=0.7", fc="#eef4fb", ec="#2f6fb5"))
    ax[1].set_title("Metrics (§6.1)")
    fig.tight_layout()
    fig.savefig(os.path.join(dataset_dir, "fig_cov_prec.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Step5 精度评估")
    ap.add_argument("--in", dest="inp", default=None)
    args = ap.parse_args()
    d = K.resolve_dir(args.inp, "选择数据集文件夹")
    run_step(d)


if __name__ == "__main__":
    main()
