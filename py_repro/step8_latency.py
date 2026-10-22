# -*- coding: utf-8 -*-
r"""
step8_latency.py —— Step8：延迟分解实验（论文 §6.2 / Figure 8 复刻）
==================================================================
★★★ 本步输入：无文件依赖，内部流式跑流水线 ★★★
★★★ 本步输出：experiments/fig8_latency.png + latency.csv ★★★

论文依据（§3 + §6.2 + Fig.4/Fig.8）
-----------------------------------
两个延迟的定义（原文口径）：
  L_proc,i  = 第 i 个事件被捕获（事件自身时间戳）→ 其三角测量结果被放进发送缓冲区
              （缓冲区最大等待 2 ms）
  L_track,i = 放进发送缓冲区 → 轨迹输出（含 trackerGNN 计算 + 最多 2 ms 缓冲等待）
  L_lat,i   = L_proc,i + L_track,i

论文实测结论（§6.2）：
  · 1P / 2P 的 L_proc 中位数 < 3 ms
  · 10P 的 L_proc 涨 3 个数量级
  · 诊断：瓶颈是**事件匹配循环**（在更大规模的事件里找最优匹配，含噪声事件），
          不是事件相机本身
  · 噪声滤波后每相机事件率：1P ≈ 5×10³ ev/s，10P ≈ 27×10³ ev/s
  · L_track 的中位数在各粒子数下都较低、分布相近（Fig.8c）

★ 本步为什么要"流式"测
  批量跑一次再除总时间，测的是"吞吐"不是"延迟"。延迟必须按到达顺序逐窗口推进：
  每 10 ms 收一批事件（与本项目 DT_S 聚类帧步一致）→ 立刻匹配三角测量 → 立刻算轨迹，逐个窗口记时间。
  这才是论文 Fig.8 那种 per-event 的口径。

★ 与论文的差异（写报告必须说明）
  1) 无网络：L_track 里没有传输时间、也没有 2 ms 发送缓冲等待，只有 trackerGNN 计算。
  2) 匹配实现不同：本项目用 searchsorted 建时间索引（O(log n + k)），
     论文的匹配循环是逐事件线性扫描找最优（更接近 O(n²)）。
     ⇒ 我们的 L_proc 随粒子数增长会明显**比论文温和**（论文 3 个数量级 ≈ 1000×）。
     这恰好从反面印证论文的诊断：瓶颈在匹配循环的写法，不在传感器。
  3) 事件率低 7~15 倍：仿真每粒子每步只发 1 个事件（≈100 ev/s·粒子，1P 时 746 ev/s·相机），
     真实 DAVIS346 里一个粒子扫过多个像素，单粒子就能产生数千 ev/s（论文 1P 为 5×10³ ev/s）。

★ 实测结果（2026-10-20，opencv_env，窗口 10 ms，按输出点加权）
  ┌────┬──────────┬────────┬──────────┬──────────┬──────────┐
  │ N  │ ev/s·相机 │ 重建点 │ L_proc中位│ L_track  │ L_lat    │
  ├────┼──────────┼────────┼──────────┼──────────┼──────────┤
  │ 1P │    746   │  140   │ 0.357 ms │ 0.297 ms │ 0.660 ms │
  │ 2P │    864   │  215   │ 0.541    │ 0.354    │ 0.886    │
  │ 3P │    980   │  245   │ 0.728    │ 0.383    │ 1.116    │
  │ 5P │   1221   │  355   │ 1.050    │ 0.497    │ 1.571    │
  │10P │   1797   │  519   │ 1.817    │ 0.714    │ 2.548    │
  └────┴──────────┴────────┴──────────┴──────────┴──────────┘
  · L_proc 中位 1P→10P 只涨 ×5.1（论文约三个数量级）——见差异 2)、3)。
  · L_track 全程 < L_proc 且随粒子数变化平缓，与论文 Fig.8c 一致。
  · 自检：流式累计匹配 1184/1184、2376/2376、3550/3550、5959/5959、11734/11734 = 100.0%。

★ 坑#8：窗口必须带 ±T_MAX_US 回看
  第一版按 [a,b) 硬切窗口，1P 匹配数从 1184 掉到 725（−39%）：仿真事件聚在步长网格上
  （t ≈ k·10 ms），而窗口网格起点 t0 = −14.2 µs，两者错开 14 µs ⇒ 每个突发正好骑在边界上，
  左事件进窗口 k、右事件进 k+1，配对被切开。
  修复后自检 100%。TC11 就是这条的回归测试。真实接收端本来就持有最近 150 µs 事件做匹配，
  回看在物理上是正确语义，不是"为了凑数把窗口放大"。

两种调用方式
------------
  1) python step8_latency.py
  2) from step8_latency import run_step;  run_step(out_dir)
"""
import argparse
import os
import time

import numpy as np

import _core as K

WINDOW_US = K.C.DT_S * 1e6   # 流式窗口：10 ms（与本项目 DT_S / 聚类帧步一致）
COUNTS = K.C.PARTICLE_COUNTS

# P14（坑#11 稳健化）：每个 (粒子数, κ) 重复测量趟数。
# 事件数据在固定 seed 下是确定性的，墙钟抖动只来自 perf_counter；
# 重复 N 趟并把 l_proc 样本拼接取中位，可把中位方差压约 √N。
N_REPEAT = 5


# ------------------------------------------------------------------
# 一、流式跟踪的一步（复用 tr.Track，避免 O(n^2) 地重跑整段历史）
# ------------------------------------------------------------------

def _tracker_step(tracks, dets, dt):
    """推进一帧：预测 -> 代价矩阵 -> 匈牙利 -> 更新/新建/漏检计数。
    逻辑与 track._gnn 的单帧体一致，只是把 tracks 状态保留在外面，便于流式调用。
    """
    dets = np.atleast_2d(dets) if len(dets) else np.zeros((0, 3))
    for tr in tracks:
        tr.predict()

    n, m = len(tracks), len(dets)
    assigned_t, assigned_d = set(), set()
    if n and m:
        cost = np.full((n, m), 1e9)
        for i, tr in enumerate(tracks):
            for j, z in enumerate(dets):
                d = float(np.linalg.norm(tr.pos - z))
                if d <= K.tr.GATE:
                    cost[i, j] = d
        ri, ci = K.tr.linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            if cost[i, j] < 1e8:
                tracks[i].update(dets[j])
                assigned_t.add(i)
                assigned_d.add(j)

    for j in range(m):
        if j not in assigned_d:
            tracks.append(K.tr.Track(dets[j], dt, _next_id(tracks)))

    for i, tr in enumerate(tracks):
        if i not in assigned_t:
            tr.miss_step()

    tracks[:] = [tr for tr in tracks if not tr._should_delete()]
    return len(dets)


def _next_id(tracks):
    return (max((t.tid for t in tracks), default=-1) + 1)


# ------------------------------------------------------------------
# 二、单个粒子数下的流式延迟测量
# ------------------------------------------------------------------

def _measure_once(n_particles, window_us=WINDOW_US, seed_offset=None, kappa=1.0):
    """kappa：事件子采样率，作用在【原始事件】上（filter 之后、match 之前）。

    默认值 1.0 = 不抽样，调用方不传时与历史行为逐位一致（TC11 依赖这一点）。
    ★ 为什么必须在 match 之前：论文 Fig.11 要证明的是"事件少了 → 匹配循环快了"，
      若在匹配之后抽样，被计时的匹配循环规模不变，L_proc 根本不会降，实验失效。
    """
    seed_offset = n_particles if seed_offset is None else seed_offset
    left, right, gt = K.ge.generate(n_particles, seed_offset)
    left, right = K.flt.filter_events(left), K.flt.filter_events(right)

    if kappa < 1.0:
        left = left[np.random.default_rng(K.C.SEED).random(len(left)) < kappa]
        right = right[np.random.default_rng(K.C.SEED).random(len(right)) < kappa]

    t0_all = min(left[:, 0].min(), right[:, 0].min())
    t1_all = max(left[:, 0].max(), right[:, 0].max())
    edges = np.arange(t0_all, t1_all + window_us, window_us)

    tracks = []
    l_proc, l_track, l_lat = [], [], []
    n_points = 0
    n_active_win = 0
    total_proc = total_track = 0.0
    dt = window_us / 1e6

    # ★ 坑#8：窗口必须带 ±T_MAX_US 的"回看/前瞻"，否则边界会把同一对事件切开。
    #   本项目的仿真事件全部聚在步长网格（t ≈ k·10ms）上，而窗口网格起点是
    #   t0_all = min(事件时间) = −14.2 µs，两者错开 14 µs ⇒ 每个突发正好骑在窗口边界上，
    #   左事件进窗口 k、右事件进窗口 k+1，匹配直接丢掉 39%（实测 1184 → 725）。
    #   真实系统里接收端本来就持有最近 150 µs 的事件缓冲，回看是物理上正确的语义。
    look = K.C.T_MAX_US
    n_matched_total = 0

    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        lw = left[(left[:, 0] >= a - look) & (left[:, 0] < b + look)]
        rw = right[(right[:, 0] >= a - look) & (right[:, 0] < b + look)]
        if len(lw) == 0 and len(rw) == 0:
            continue
        n_active_win += 1

        ta = time.perf_counter()
        matched = K.mt.match_and_triangulate(lw, rw) if (len(lw) and len(rw)) \
            else np.zeros((0, 4))
        if len(matched):                        # 只保留落在本窗口的输出点，避免重复计数
            matched = matched[(matched[:, 0] >= a) & (matched[:, 0] < b)]
        tb = time.perf_counter()
        n_matched_total += len(matched)

        centers = K.cl.cluster_points(matched[:, 1:]) if len(matched) else np.zeros((0, 3))
        _tracker_step(tracks, centers, dt)
        tc = time.perf_counter()

        total_proc += (tb - ta)
        total_track += (tc - tb)

        # ★ 关键：按"输出点"加权。没有输出点的窗口不产生延迟样本——
        #   因为 L_proc 定义里被计时的对象是"被放进缓冲区的三角测量结果"，
        #   没产出结果就没有可计时的对象。
        #   同一窗口产出的 k 个点是一起被释放的，**每个点的延迟都等于该窗口的处理耗时**，
        #   所以把 T 重复记 k 次（不是 T/k，那会把批处理延迟摊薄成假的低值）。
        k_pts = len(centers)
        if k_pts:
            n_points += k_pts
            l_proc.extend([(tb - ta) * 1e3] * k_pts)
            l_track.extend([(tc - tb) * 1e3] * k_pts)
            l_lat.extend([(tc - ta) * 1e3] * k_pts)

    dur = (t1_all - t0_all) / 1e6
    return {
        "n": n_particles,
        "n_matched": n_matched_total,
        "n_matched_global": len(K.mt.match_and_triangulate(left, right)),   # 自检基准
        "l_proc": np.array(l_proc),
        "l_track": np.array(l_track),
        "l_lat": np.array(l_lat),
        "n_points": n_points,
        "n_win": n_active_win,
        "ev_per_sec": len(left) / dur,          # 每相机（滤波后）
        "dur_s": dur,
        # 聚合口径：整段总耗时 / 输出点数 = 系统级"每点平均延迟"
        "agg_proc": total_proc / max(n_points, 1) * 1e3,
        "agg_track": total_track / max(n_points, 1) * 1e3,
    }


def measure(n_particles, window_us=WINDOW_US, seed_offset=None, kappa=1.0, n_repeat=1):
    """单次或多次流式测量（P14 / 坑#11 稳健化）。

    n_repeat<=1：与历史逐位一致（返回单次结果，rep_medians=[该趟中位]）。
    n_repeat>1 ：重复跑 n_repeat 趟——事件数据在固定 seed 下是确定性的，
                 只有墙钟计时在抖。把每趟的 l_proc / l_track / l_lat 样本直接拼接成一维
                 数组再取中位，相当于对同一场景多测几次取合并样本，把中位方差压约 √n_repeat。
                 · 数据派生量（n_matched / n_points / ev_per_sec / dur_s）各趟完全一致，
                   取首趟即可，锚点（11734/519）不受重复影响。
                 · rep_medians 是每趟单独的 L_proc 中位，供 TC12 做"内部波动 ≤25%"断言。
                 · n_points 仍报单趟重建点数（重复测量不产生 5 倍重建点，只是把同一批点
                   多测了几遍），l_proc 数组长 5× 只用于更稳的中位估计。
    """
    if n_repeat <= 1:
        r = _measure_once(n_particles, window_us, seed_offset, kappa)
        r["rep_medians"] = [float(np.median(r["l_proc"]))]
        return r

    reps = [_measure_once(n_particles, window_us, seed_offset, kappa)
            for _ in range(n_repeat)]
    base = reps[0]
    l_proc = np.concatenate([r["l_proc"] for r in reps])
    l_track = np.concatenate([r["l_track"] for r in reps])
    l_lat = np.concatenate([r["l_lat"] for r in reps])
    agg_proc = float(np.mean([r["agg_proc"] for r in reps]))
    agg_track = float(np.mean([r["agg_track"] for r in reps]))
    rep_medians = [float(np.median(r["l_proc"])) for r in reps]
    return {
        "n": base["n"],
        "n_matched": base["n_matched"],
        "n_matched_global": base["n_matched_global"],
        "l_proc": l_proc,
        "l_track": l_track,
        "l_lat": l_lat,
        "n_points": base["n_points"],
        "n_win": base["n_win"],
        "ev_per_sec": base["ev_per_sec"],
        "dur_s": base["dur_s"],
        "agg_proc": agg_proc,
        "agg_track": agg_track,
        "rep_medians": rep_medians,
    }


def run_step(out_dir=None, verbose=True):
    K.banner("Step8", "延迟分解 L_proc / L_track / L_lat", "§6.2 与 Figure 8（网络追踪延迟评估）")

    out_dir = K.ensure_dir(out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
    rows, results = [], []

    for n in COUNTS:
        r = measure(n, n_repeat=N_REPEAT)
        results.append(r)
        p50 = float(np.median(r["l_proc"]))
        p95 = float(np.percentile(r["l_proc"], 95))
        t50 = float(np.median(r["l_track"]))
        a50 = float(np.median(r["l_lat"]))
        rows.append((n, r["ev_per_sec"], r["n_points"], p50, p95, t50, a50))
        if verbose:
            print(f"[{n:>2}P] 事件率 {r['ev_per_sec']:>6.0f} ev/s·相机 | 重建点 {r['n_points']:>4} | "
                  f"L_proc 中位 {p50:6.3f} ms (p95 {p95:6.3f}) | "
                  f"L_track 中位 {t50:6.3f} ms | 每点摊薄 {r['agg_proc']:6.3f} ms")
            print(f"      自检：流式累计匹配 {r['n_matched']} / 全局一次匹配 "
                  f"{r['n_matched_global']}"
                  f"（{r['n_matched'] / max(r['n_matched_global'], 1) * 100:.1f}%，应≈100%）")

    with open(os.path.join(out_dir, "latency.csv"), "w", encoding="utf-8-sig") as f:
        f.write("particles,events_per_sec_per_camera,recon_points,"
                "L_proc_median_ms,L_proc_p95_ms,L_track_median_ms,L_lat_median_ms\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]:.1f},{row[2]},{row[3]:.6f},{row[4]:.6f},"
                    f"{row[5]:.6f},{row[6]:.6f}\n")

    _plot(results, out_dir)

    if verbose:
        base = rows[0][3] if rows[0][3] > 0 else 1e-9
        last = rows[-1][3]
        print(f"[lat ] L_proc 中位数 1P→10P：{base:.3f} ms → {last:.3f} ms"
              f"（×{last / base:.1f}）")
        print(f"[note] 论文称 10P 涨 3 个数量级(≈1000×)；本项目用 searchsorted 时间索引，"
              f"增长温和得多——见脚本抬头「与论文的差异 2」")
        print(f"[out ] 已保存：{out_dir}\\fig8_latency.png / latency.csv")
    return rows, results


def _plot(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) 箱线图：L_proc 随粒子数
    data = [r["l_proc"] for r in results]
    labels = [f"{r['n']}P" for r in results]
    bp = ax[0].boxplot(data, positions=range(1, len(data) + 1),
                       showfliers=False, patch_artist=True)
    ax[0].set_xticks(range(1, len(labels) + 1))
    ax[0].set_xticklabels(labels)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe3f7")
        patch.set_edgecolor("#185FA5")
    ax[0].set_ylabel("L_proc (ms)")
    ax[0].set_title("(a) Processing latency per particle count")

    # (b) CCDF：L_proc
    for r, c in zip(results, plt.get_cmap("viridis")(np.linspace(0, .85, len(results)))):
        x = np.sort(r["l_proc"])
        if len(x) == 0:
            continue
        y = 1.0 - np.arange(len(x)) / len(x)
        ax[1].plot(x, y, lw=1.6, color=c, label=f"{r['n']}P")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("processing latency (ms)")
    ax[1].set_ylabel("CCDF")
    ax[1].set_title("(b) CCDF of processing latency")
    ax[1].legend(fontsize=9, ncol=2)
    ax[1].grid(alpha=.25, which="both")

    # (c) CDF：L_track
    for r, c in zip(results, plt.get_cmap("plasma")(np.linspace(0, .8, len(results)))):
        x = np.sort(r["l_track"])
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax[2].plot(x, y, lw=1.6, color=c, label=f"{r['n']}P")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("remote tracking latency (ms)")
    ax[2].set_ylabel("CDF")
    ax[2].set_title("(c) CDF of remote tracking latency")
    ax[2].legend(fontsize=9, ncol=2)
    ax[2].grid(alpha=.25, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig8_latency.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Step8 延迟分解")
    ap.add_argument("--out", default=None, help="输出目录（默认 py_repro/experiments）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_step(args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
