# -*- coding: utf-8 -*-
r"""
step2_match_triangulate.py —— Step2：双目事件匹配（式2）+ 三角测量
==================================================================
★★★ 本步输入：data/<名>/events_left_f.npz + events_right_f.npz（Step1 产物）★★★
★★★ 本步输出：match_points.npz / .csv + fig_match_Z.png ★★★

论文依据
--------
§4 事件匹配三关卡（对应 config.py 四个阈值）+ 式(2) 评分：

  关卡① 时间窗 |t_L - t_R| <= T_MAX_US = 150 µs      —— 太远就不是同一次闪烁
  关卡② 纵向偏移 |y_L - y_R| <= Y_TOL_PX = 15 px     —— 双目并排同高，y 应该几乎一样
  关卡③ 视差 d = x_L - x_R ∈ [D_MIN_PX=5, D_MAX_PX=150] px
                                                       —— d<=0 违反几何，d 太小=无穷远/同点误配

  式(2) 加权评分（分越低越好，α=0.5 时间、β=0.4 纵向、γ=0.1 视差）：
        score = α·|Δt|/t_max + β·|Δy|/y_tol + γ·|d - d_mid|/((d_max-d_min)/2)
      三个候选里取分数最低那个作为匹配对（贪心一对多，简化版 Hungarian）。

  三角测量（标准立体视觉，论文 §4 的核心输出）：
        Z = f · B / d          f=300px, B=0.14m
        X = (x_L - W/2) · Z / f   （左相机光心为原点）
        Y = (y_L - H/2) · Z / f
      单位换算：Z 单位米；d 单位像素。**d 只有 1 px 的量化误差，就会让 Z 跳 1cm 量级**
      —— 这就是 Step3 里"1207 个点只聚出 148 簇"的根因，不是 bug。

两种调用方式
------------
  1) python step2_match_triangulate.py --in "py_repro\data\sim1p"
  2) from step2_match_triangulate import run_step;  run_step(dataset_dir)
"""
import argparse
import os

import numpy as np

import _core as K


def run_step(dataset_dir, make_fig=True, verbose=True):
    K.banner("Step2", "双目事件匹配（式2）+ 三角测量 Z=f·B/d", "§4 事件匹配 + 式(2) + 立体三角测量")

    left = K.load_events(dataset_dir, "left_f")
    right = K.load_events(dataset_dir, "right_f")

    matched = K.mt.match_and_triangulate(left, right)        # (M,4)=[t,X,Y,Z]

    np.savez_compressed(os.path.join(dataset_dir, "match_points.npz"), pts=matched)
    with open(os.path.join(dataset_dir, "match_points.csv"), "w", encoding="utf-8-sig") as f:
        f.write("t_us,X_m,Y_m,Z_m\n")
        for row in matched:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")

    if verbose:
        print(f"[match] 三关卡：|Δt|<={K.C.T_MAX_US:.0f}µs  |Δy|<={K.C.Y_TOL_PX:.0f}px  "
              f"d∈[{K.C.D_MIN_PX:.0f},{K.C.D_MAX_PX:.0f}]px")
        print(f"[match] 式(2) 权重：α={K.C.W_T} β={K.C.W_Y} γ={K.C.W_D}")
        print(f"[match] 左 {len(left)} / 右 {len(right)} -> 匹配 {len(matched)} 个 3D 点")
        if len(matched):
            z = matched[:, 3]
            print(f"[match] Z 深度：min={z.min():.3f}m  中位数={np.median(z):.3f}m  "
                  f"max={z.max():.3f}m  标准差={z.std():.3f}m")
            print(f"[match] 三角测量：f={K.C.F_PX:.0f}px  B={K.C.BASELINE:.2f}m")

    if make_fig and len(matched):
        _plot_z(matched[:, 3], dataset_dir)

    print(f"[out  ] 已保存：match_points.npz / match_points.csv" +
          (" / fig_match_Z.png" if make_fig and len(matched) else ""))
    return matched


def _plot_z(z, dataset_dir):
    """Z 深度直方图——一眼看出匹配结果健不健康（该是单峰 + 少量离群）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(z, bins=60, color="#2f6fb5", alpha=0.85)
    ax.set_xlabel("Z (m)")
    ax.set_ylabel("count")
    ax.set_title(f"Reconstructed depth  N={len(z)}  median={np.median(z):.2f} m")
    fig.tight_layout()
    path = os.path.join(dataset_dir, "fig_match_Z.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="Step2 事件匹配 + 三角测量")
    ap.add_argument("--in", dest="inp", default=None, help="数据集目录")
    ap.add_argument("--no-fig", action="store_true", help="不画图")
    args = ap.parse_args()
    d = K.resolve_dir(args.inp, "选择数据集文件夹")
    run_step(d, make_fig=not args.no_fig)


if __name__ == "__main__":
    main()
