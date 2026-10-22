# -*- coding: utf-8 -*-
r"""
step4_track.py —— Step4：逐帧跟踪（GNN + 卡尔曼 + 匈牙利）
==================================================================
★★★ 本步输入：data/<名>/cluster_centers.npz（Step3 产物）★★★
★★★ 本步输出：track_points.npz / track_ids.npz + fig_tracks.png ★★★

论文依据
--------
§4 接收端跟踪器 = MATLAB `trackerGNN`（全局最近邻），四个动作：

  ① 预测 (predict)：每条轨迹用**恒速卡尔曼**把位置外推到当前帧
       状态 x = [X,Y,Z, vx,vy,vz]，F 里塞 dt，P = F P Fᵀ + Q
  ② 分配 (assign)：预测点 vs 当前帧检测点做**代价矩阵**（欧氏距离），
       用 **匈牙利算法**（scipy.linear_sum_assignment）全局最优配对，
       超过门限 GATE=0.25 m 的不配
  ③ 确认 (confirm)：最近 3 步窗口内 >=1 次命中才算"真轨迹"（CONFIRM=(1,3)）
  ④ 删除 (delete)：最近 12 步窗口内 >=5 次漏检就注销（DELETE=(5,12)，不要求连续）

论文原文 = MATLAB trackerGNN 的确认阈值 (1,3)、删除阈值 (5,12)。
2026-10-21 起按论文滑窗语义实现（原为 CONFIRM_AFTER=1 / 连续漏检12步 的简化，
并修掉新轨迹出生帧白背一次 miss 的差一错误）。

★ 为什么指标评估"用聚类点、不用跟踪点"（重要）：
  简化版检测器太稀疏（135 帧有检测 / 991 帧），连续漏检 12 步就会把轨迹删光，
  跟踪输出可能只剩几十个点甚至 0 个点。而论文 6.1 节的 Cov/Prec/Chamfer
  评的是**每一帧的重建点集**，不是轨迹——所以 Step5 吃的是 Step3 的簇心。
  跟踪这一步是"论文 §4 系统里确实有、且必须演示"的环节，但它不是精度指标的输入。

两种调用方式
------------
  1) python step4_track.py --in "py_repro\data\sim1p"
  2) from step4_track import run_step;  run_step(dataset_dir)
"""
import argparse
import os

import numpy as np

import _core as K


def run_step(dataset_dir, make_fig=True, verbose=True):
    K.banner("Step4", "逐帧跟踪（GNN + 卡尔曼 + 匈牙利）", "§4 MATLAB trackerGNN：预测/分配/确认/删除")

    path = os.path.join(dataset_dir, "cluster_centers.npz")
    if not os.path.exists(path):
        raise SystemExit(f"[×] 找不到 {path}\n    请先跑 Step3：python step3_cluster.py --in {dataset_dir}")
    centers = np.load(path)["centers"]

    frames = K.tr.frames_from_centers(centers)                # 按时间步补空帧，保持真实时间轴
    n_det = sum(1 for f in frames if len(f))
    pts, ids, n_trk = K.tr.track_trajectories(frames, keep_deleted=True)

    np.savez_compressed(os.path.join(dataset_dir, "track_points.npz"), pts=pts)
    np.savez_compressed(os.path.join(dataset_dir, "track_ids.npz"), ids=ids)

    if verbose:
        print(f"[track] 输入簇心   : {len(centers)}   帧数 {len(frames)}（有检测 {n_det} 帧）")
        print(f"[track] 参数       : 确认 {K.C.CONFIRM}（滑窗）/ 删除 {K.C.DELETE}（滑窗）"
              f" / 门限 {K.tr.GATE:g} m")
        print(f"[track] 卡尔曼     : 恒速模型 [X,Y,Z,vx,vy,vz]，Q={K.C.PROCESS_NOISE:g} R={K.C.MEAS_NOISE:g}")
        print(f"[track] 输出轨迹   : {n_trk} 条，{len(pts)} 个轨迹点")
        if len(pts):
            print(f"[track] 轨迹点 Z   : {pts[:, 2].min():.3f} ~ {pts[:, 2].max():.3f} m")

    if make_fig and len(pts):
        _plot(pts, ids, dataset_dir)

    print(f"[out  ] 已保存：track_points.npz / track_ids.npz" +
          (" / fig_tracks.png" if make_fig and len(pts) else ""))
    return pts, ids, n_trk


def _plot(pts, ids, dataset_dir):
    """轨迹图：不同 id 不同颜色，XZ 俯视 + Z 随时间变化。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    uniq = np.unique(ids) if len(ids) else []
    cmap = plt.get_cmap("tab20")
    for k, tid in enumerate(uniq[:20]):
        m = ids == tid
        ax[0].plot(pts[m, 0], pts[m, 2], "-o", ms=3, lw=1, color=cmap(k % 20), label=f"id{tid}")
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Z (m)")
    ax[0].set_title(f"Tracks (XZ)  {len(uniq)} ids")
    if len(uniq) <= 8:
        ax[0].legend(fontsize=7, ncol=2)
    ax[1].scatter(np.arange(len(pts)), pts[:, 2], s=8, c="#2f6fb5")
    ax[1].set_xlabel("track point index (in time order)"); ax[1].set_ylabel("Z (m)")
    ax[1].set_title("Z along track points")
    fig.tight_layout()
    fig.savefig(os.path.join(dataset_dir, "fig_tracks.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Step4 跟踪")
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    d = K.resolve_dir(args.inp, "选择数据集文件夹")
    run_step(d, make_fig=not args.no_fig)


if __name__ == "__main__":
    main()
