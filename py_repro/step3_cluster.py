# -*- coding: utf-8 -*-
r"""
step3_cluster.py —— Step3：3D 点云聚类（简化 DBSCAN）
==================================================================
★★★ 本步输入：data/<名>/match_points.npz（Step2 产物）★★★
★★★ 本步输出：cluster_centers.npz / .csv + fig_cluster.png ★★★

论文依据
--------
§4 接收端：匹配出的 3D 点里，**一个真实粒子在一帧内可能贡献好几个事件点**（突发 burst），
噪声点也会零星落进来。所以在进入跟踪器之前先聚类：

  ① 按时间步分帧（一个时间步 = DT_S = 10 ms，论文 §5.2 的帧长）
  ② 帧内做半径聚类：两点距离 <= CLUSTER_RADIUS(1 cm) 归为一簇（论文 DBSCAN 的 eps=1cm）
  ③ 每簇点数 < MIN_POINTS(2) 直接丢掉（单点/离群，论文 DBSCAN 的 minPts=2）
  ④ 簇内坐标取平均 = 这个粒子的 3D 位置估计

★ 你会看到一个"怪现象"：1207 个匹配点只聚出 148 簇（丢掉 88%）。
  这不是参数错，是**像素量化**的必然结果：Z = f·B/d 里 d 只有整像素精度，
  1 px 的 d 抖动会让 Z 跳约 1.2 cm，本身就超过 1 cm 的聚类半径，
  所以"同一粒子的两个采样点"经常凑不进一个 1 cm 球里，各自成了单点被丢掉。
  论文用真实 DAVIS346 + 更细的匹配（亚像素/多事件融合）才没这么惨。
  这条写进复现报告里，是"简化模型 vs 论文"最该说清楚的一条。

两种调用方式
------------
  1) python step3_cluster.py --in "py_repro\data\sim1p"
  2) from step3_cluster import run_step;  run_step(dataset_dir)
"""
import argparse
import os

import numpy as np

import _core as K


def run_step(dataset_dir, make_fig=True, verbose=True):
    K.banner("Step3", "3D 点云聚类（1cm 半径 + 每簇>=2 点）", "§4 DBSCAN（eps=1cm, minPts=2）")

    path = os.path.join(dataset_dir, "match_points.npz")
    if not os.path.exists(path):
        raise SystemExit(f"[×] 找不到 {path}\n    请先跑 Step2：python step2_match_triangulate.py --in {dataset_dir}")
    pts = np.load(path)["pts"]

    step_us = K.C.DT_S * 1e6
    n_frames = len(np.unique((pts[:, 0] // step_us).astype(int))) if len(pts) else 0
    centers = K.cl.cluster_timed(pts)                        # (K,4)=[t_us,X,Y,Z]

    np.savez_compressed(os.path.join(dataset_dir, "cluster_centers.npz"), centers=centers)
    with open(os.path.join(dataset_dir, "cluster_centers.csv"), "w", encoding="utf-8-sig") as f:
        f.write("t_us,X_m,Y_m,Z_m\n")
        for row in centers:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")

    if verbose:
        print(f"[cluster] 输入匹配点 : {len(pts)}")
        print(f"[cluster] 分帧       : {n_frames} 帧（步长 {step_us:.0f} µs = {K.C.DT_S * 1000:.0f} ms）")
        print(f"[cluster] 参数       : 半径 {K.C.CLUSTER_RADIUS * 100:.0f} cm，每簇至少 {K.C.MIN_POINTS} 点")
        print(f"[cluster] 输出簇心   : {len(centers)}")
        if len(pts):
            print(f"[cluster] 保留率     : {len(centers) / len(pts) * 100:.1f}%"
                  f"  （像素量化导致 Z 抖动 > 1cm，同源点常凑不成簇，属预期）")

    if make_fig and len(centers):
        _plot(centers, dataset_dir)

    print(f"[out  ] 已保存：cluster_centers.npz / cluster_centers.csv" +
          (" / fig_cluster.png" if make_fig and len(centers) else ""))
    return centers


def _plot(centers, dataset_dir):
    """簇心的 3D 分布（XZ 俯视 + 时间轴），看粒子是不是在 0.4~1.0 m 那个环里跑。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(centers[:, 1], centers[:, 3], c=centers[:, 0] / 1e6, s=10, cmap="viridis")
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Z (m)")
    ax[0].set_title(f"Cluster centers (XZ view)  K={len(centers)}")
    ax[1].hist(centers[:, 3], bins=40, color="#2f6fb5", alpha=0.85)
    ax[1].set_xlabel("Z (m)"); ax[1].set_ylabel("count")
    ax[1].set_title("Depth distribution of cluster centers")
    fig.tight_layout()
    fig.savefig(os.path.join(dataset_dir, "fig_cluster.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Step3 点云聚类")
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    d = K.resolve_dir(args.inp, "选择数据集文件夹")
    run_step(d, make_fig=not args.no_fig)


if __name__ == "__main__":
    main()
