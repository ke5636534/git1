# -*- coding: utf-8 -*-
"""一键复现入口。对应论文第5-6节实验：
  实验1：不同粒子数 {1,2,3,5,10}P 下的覆盖率/精确率/倒角距离（类比Fig.6/7）
  实验2：子采样κ 与 质量阈值ν 对精度的影响（类比Fig.12）
运行: python run_all.py
"""
import os
import time
# matplotlib 缓存/配置写到本目录，避免占用C盘用户目录
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", ".mpl"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import gen_events, filter as flt, match_triangulate as mt, cluster as cl, track as tr, metrics, streaming

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

LOG_PATH = os.path.join(OUT, "run_log.txt")
_log_fh = None


def log(msg=""):
    """同时打印到 stdout 和 results/run_log.txt，每行立即 flush（边跑边能看）。"""
    global _log_fh
    print(msg, flush=True)
    if _log_fh is None:
        _log_fh = open(LOG_PATH, "w", encoding="utf-8")
    _log_fh.write(msg + "\n")
    _log_fh.flush()


def close_log():
    global _log_fh
    if _log_fh is not None:
        _log_fh.close()
        _log_fh = None


def stage(tag, text):
    """阶段行：标签补齐到 13 列对齐。"""
    log(f"[{tag}]".ljust(13) + text)


def pipeline(n_particles, kappa=1.0, nu=-1.0, seed_offset=0):
    """完整流水线，返回 metrics 字典。"""
    t = time.perf_counter()
    left, right, gt = gen_events.generate(n_particles, seed_offset)
    stage("gen_events", f"{n_particles}粒子 -> 左{len(left)}/右{len(right)}事件, "
                        f"真值{len(gt)}点, 用时{time.perf_counter() - t:.2f}s")

    n_l, n_r = len(left), len(right)
    t = time.perf_counter()
    left, right = flt.filter_events(left), flt.filter_events(right)
    stage("filter", f"左{n_l}->{len(left)} 右{n_r}->{len(right)}, "
                    f"用时{time.perf_counter() - t:.2f}s")

    t = time.perf_counter()
    matched = mt.match_and_triangulate(left, right, return_scores=True)   # (t,X,Y,Z,s)
    if len(matched):
        zmin, zmax = float(matched[:, 3].min()), float(matched[:, 3].max())
        stage("match", f"匹配点{len(matched)}个, Z范围[{zmin:.3f},{zmax:.3f}]m, "
                       f"用时{time.perf_counter() - t:.2f}s")
    else:
        stage("match", f"匹配点0个, 用时{time.perf_counter() - t:.2f}s")
    if len(matched) == 0:
        stage("cluster", f"0点 -> 0簇, 用时{0.0:.2f}s")
        t = time.perf_counter()
        m = metrics.evaluate(np.zeros((0, 3)), gt[:, 1:])
        stage("metrics", f"Cov={m['coverage']:.3f}/Prec={m['precision']:.3f}/"
                         f"Chamfer={m['chamfer']:.4f}m, 用时{time.perf_counter() - t:.2f}s")
        return m
    # 传输层：子采样 + 质量滤波（作用在带时间戳与式(2)分数的匹配行上）
    # ν 滤波用论文口径：保留 s <= ν（s 是式(2) 匹配分数，越小越好）——审查发现A，销坑#5
    keep = np.ones(len(matched), dtype=bool)
    if kappa < 1.0:
        keep &= np.random.default_rng(C.SEED).random(len(matched)) < kappa
    if nu >= 0:
        keep &= matched[:, 4] <= nu
    matched = matched[keep][:, :4]               # 后续只用 (t,X,Y,Z) 四列
    # 按时间步分帧 -> 逐帧聚类（跟踪器 track.py 作为独立模块演示，
    # 指标评估用聚类后的重建点，避免简化跟踪器过度过滤）
    t = time.perf_counter()
    step_us = C.DT_S * 1e6
    buckets = {}
    for row in matched:
        k = int(row[0] / step_us)
        buckets.setdefault(k, []).append(row[1:])
    frames = [cl.cluster_points(np.array(buckets[k]))
              for k in sorted(buckets) if len(buckets[k])]
    recon = np.vstack(frames) if frames else np.zeros((0, 3))
    stage("cluster", f"{len(matched)}点 -> {len(recon)}簇, "
                     f"用时{time.perf_counter() - t:.2f}s")
    # 跟踪器演示（1P 场景），仅打印存活轨迹数
    if n_particles == 1:
        tracks = tr.track_frames(frames)
        log(f"[tracker demo] 1P 跟踪输出点: {len(tracks)}")

    t = time.perf_counter()
    m = metrics.evaluate(recon, gt[:, 1:])
    stage("metrics", f"Cov={m['coverage']:.3f}/Prec={m['precision']:.3f}/"
                     f"Chamfer={m['chamfer']:.4f}m, 用时{time.perf_counter() - t:.2f}s")
    return m


def main():
    # ---- 实验1：粒子数扫描 ----
    rows = []
    for n in C.PARTICLE_COUNTS:
        log(f"===== 开始 {n} 粒子 =====")
        m = pipeline(n, seed_offset=n)
        rows.append((n, m["coverage"], m["precision"], m["chamfer"], m["n_recon"]))
        log(f"{n}P  Cov={m['coverage']:.3f} Prec={m['precision']:.3f} "
            f"Chamfer={m['chamfer']:.4f}m  recon={m['n_recon']}")

    arr = np.array([(r[1], r[2], r[3]) for r in rows])
    counts = [r[0] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(counts))
    ax[0].bar(x - 0.2, arr[:, 0], 0.4, label="Coverage")
    ax[0].bar(x + 0.2, arr[:, 1], 0.4, label="Precision")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{c}P" for c in counts])
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("ratio"); ax[0].legend()
    ax[0].set_title("Coverage / Precision vs particle count (Fig.6 analog)")
    ax[1].plot(counts, arr[:, 2], "o-", color="red")
    ax[1].set_xlabel("particle count"); ax[1].set_ylabel("Chamfer (m)")
    ax[1].set_title("Chamfer distance vs particle count")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_particle_count.png"), dpi=150)
    plt.close(fig)

    # ---- 实验2：子采样κ 与 质量阈值ν ----
    kappas = [1.0, 0.8, 0.6, 0.4, 0.2]
    cov_k, prec_k = [], []
    for kp in kappas:
        log(f"----- 子采样 kappa={kp:.2f} -----")
        m = pipeline(3, kappa=kp, seed_offset=3)
        cov_k.append(m["coverage"]); prec_k.append(m["precision"])

    cov_n, prec_n = [], []
    for nu in C.NU_GRID:
        log(f"----- 质量阈值 nu={nu:.2f} -----")
        m = pipeline(3, nu=nu, seed_offset=3)
        cov_n.append(m["coverage"]); prec_n.append(m["precision"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(kappas, cov_k, "o-", label="Coverage")
    ax[0].plot(kappas, prec_k, "s-", label="Precision")
    ax[0].set_xlabel("subsample fraction kappa"); ax[0].set_ylim(0, 1.05); ax[0].legend()
    ax[0].set_title("Uniform random subsampling (Fig.12 analog)")
    ax[1].plot(C.NU_GRID, cov_n, "o-", label="Coverage")
    ax[1].plot(C.NU_GRID, prec_n, "s-", label="Precision")
    ax[1].set_xlabel("quality threshold nu"); ax[1].set_ylim(0, 1.05); ax[1].legend()
    ax[1].set_title("Congestion quality filtering (Fig.12 analog)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig12_streaming.png"), dpi=150)
    plt.close(fig)

    # ---- 保存CSV ----
    with open(os.path.join(OUT, "results_particle_count.csv"), "w", encoding="utf-8-sig") as f:
        f.write("particles,coverage,precision,chamfer_m,n_recon\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    log("图与CSV已保存到: " + OUT)
    log("运行日志已保存到: " + LOG_PATH)
    close_log()


if __name__ == "__main__":
    main()
