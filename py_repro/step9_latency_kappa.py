# -*- coding: utf-8 -*-
r"""
step9_latency_kappa.py —— Step9：Fig.11（L_proc vs κ）+ Fig.13（重建点/秒 vs ν）
==================================================================================
★★★ 本步输入：无文件依赖，内部流式跑流水线（复用 step8 的窗口循环）★★★
★★★ 本步输出：experiments/fig11_latency_vs_kappa.png + latency_kappa.csv ★★★
                experiments/fig13_events_per_sec_vs_nu.png + events_per_sec_nu.csv ★★★

论文依据（§6.4，page 9，原文口径不改）
--------------------------------------
  Fig.11  "the mean of the processing, i.e. triangulation, latency is reduced by
           orders of magnitude for decreasing κ"          → L_proc 随 κ 下降
  Fig.13  "Average number of reconstructed events/sec for different filter
           thresholds ν given 1P"                          → 1P 下重建点/秒 vs ν（纵轴 log10）

★ 关键口径：κ 必须作用在【原始事件】上，即 filter 之后、match 之前
  论文要证明的是"事件率降了 → 匹配循环变快了"。
  若在匹配之后抽样（像 step6 / run_all.pipeline 那样），被计时的匹配循环规模不变，
  L_proc 纹丝不动，这个实验就失效了。本步通过 step8.measure(kappa=...) 把抽样点放在
  filter 与 match 之间，左右相机各用 np.random.default_rng(C.SEED) 独立抽取。

★ 锚点：κ=1.0 @10P 必须与 step8 的 10P 行逐位一致（匹配 11734 / 重建点 519）
  为此 seed_offset 固定 10（与 step8 的 10P 同源）。TC12 是这条的回归测试。

★ 与论文的差异（写报告必须说明）
  1) 事件率低 7~15 倍：仿真每粒子每步只发 1 个事件，真实 DAVIS346 单粒子就数千 ev/s。
  2) 匹配实现不同：本项目用 searchsorted 时间索引（O(log n + k)），论文的匹配循环更接近
     O(n²)。⇒ 我们的 L_proc 随 κ 下降是"近线性"的，达不到论文的 orders of magnitude。
     本项目只要 κ 1.0→0.1 降 3 倍以上，即认为"降载确实压低处理延迟"这一方向成立。

两种调用方式
------------
  1) python step9_latency_kappa.py
  2) from step9_latency_kappa import run_step;  run_step(out_dir)
"""
import argparse
import os

import numpy as np

import _core as K
import step8_latency as s8

KAPPAS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
FIG11_PARTICLES = 10      # 10 粒子固定场景
FIG11_SEED = 10           # ★ 与 step8 的 10P 同源，才能做锚点比对
FIG13_PARTICLES = 1       # Fig.13 原文：given 1P
FIG13_SEED = 1

# 锚点（step8 的 10P 实测值，TC12 直接引用；2026-10-21 P-A 修正后重测：
#   互斥匹配 + N_PER_STEP=3，匹配点翻倍、簇心 ×10）
ANCHOR_MATCHED = 23284
ANCHOR_POINTS = 5297
ANCHOR_L_PROC_MEDIAN_MS = 4.424


# ==================================================================
#  【A】Fig.11：L_proc vs κ（10 粒子）
# ==================================================================

def measure_kappa(kappa, n_particles=FIG11_PARTICLES, seed_offset=FIG11_SEED,
                  window_us=s8.WINDOW_US, n_repeat=s8.N_REPEAT):
    """单个 κ 的一次（或重复）流式测量。返回一行统计字典（含 step8 原始结果 raw）。

    n_repeat>1 时 step8.measure 把多趟 l_proc 样本拼接取中位，p50 更稳；
    rep_medians 是各趟单独中位，供 TC12 做"内部波动 ≤25%"断言（P14 / 坑#11）。
    """
    r = s8.measure(n_particles, window_us=window_us, seed_offset=seed_offset,
                   kappa=kappa, n_repeat=n_repeat)
    lp = r["l_proc"]
    n = len(lp)
    return {
        "kappa": float(kappa),
        "ev_per_sec": float(r["ev_per_sec"]),          # 每相机（滤波 + 抽样后）
        "n_matched": int(r["n_matched"]),              # 流式累计匹配点
        "n_points": int(r["n_points"]),                # 重建点（簇心，按输出点计）
        "p50": float(np.median(lp)) if n else float("nan"),
        "mean": float(np.mean(lp)) if n else float("nan"),
        "p95": float(np.percentile(lp, 95)) if n else float("nan"),
        "n_samples": n,                                # 延迟样本数（= 重建点数 × n_repeat）
        "dur_s": float(r["dur_s"]),
        "pts_per_sec": float(r["n_points"] / r["dur_s"]) if r["dur_s"] > 0 else 0.0,
        "rep_medians": r["rep_medians"],
        "raw": r,
    }


def sweep_kappa(kappas=KAPPAS, verbose=False):
    """κ 扫描，返回每 κ 一行。"""
    rows = []
    for kp in kappas:
        row = measure_kappa(kp)
        rows.append(row)
        if verbose:
            print(f"[κ={kp:.1f}] 事件率 {row['ev_per_sec']:>6.0f} ev/s·相机 | "
                  f"匹配 {row['n_matched']:>5} | 重建点 {row['n_points']:>4} | "
                  f"L_proc 中位 {row['p50']:6.3f} ms (均值 {row['mean']:6.3f} / "
                  f"p95 {row['p95']:6.3f}) | 重建点/秒 {row['pts_per_sec']:6.1f} | "
                  f"样本 {row['n_samples']}")
    return rows


# ==================================================================
#  【B】Fig.13：重建点/秒 vs ν（1 粒子）
# ==================================================================

def nu_scan(n_particles=FIG13_PARTICLES, seed_offset=FIG13_SEED, verbose=False):
    """1P 场景下按质量分数阈值 ν 过滤，统计存活点数与重建点/秒。
    ★ 2026-10-21 起用论文口径：s = 式(2) 匹配分数（越小越好），保留 s <= ν（审查发现A，销坑#5）。

    返回 (rows, n_matched, dur_s)。rows 每项 {nu, n_survive, pts_per_sec}。
    ν=0.0 时不过滤，存活点数 = 该场景匹配点数（1P/seed=1 时为 1184，锚点 c）。
    """
    left, right, gt = K.ge.generate(n_particles, seed_offset)
    left, right = K.flt.filter_events(left), K.flt.filter_events(right)
    matched = K.mt.match_and_triangulate(left, right, return_scores=True)   # (M,5)

    s = matched[:, 4] if len(matched) else np.zeros(0)

    t0 = min(float(left[:, 0].min()), float(right[:, 0].min()))
    t1 = max(float(left[:, 0].max()), float(right[:, 0].max()))
    dur = (t1 - t0) / 1e6

    rows = []
    for nu in K.C.NU_GRID:
        nu = float(nu)
        n = int(np.sum(s <= nu)) if len(s) else 0
        rows.append({"nu": nu, "n_survive": n,
                     "pts_per_sec": (n / dur) if dur > 0 else 0.0})
        if verbose:
            print(f"[ν={nu:.1f}] 存活点 {n:>5} | 重建点/秒 {rows[-1]['pts_per_sec']:8.2f}")
    return rows, int(len(matched)), float(dur)


# ==================================================================
#  入口
# ==================================================================

def run_step(out_dir=None, verbose=True):
    K.banner("Step9", "延迟 vs κ（Fig.11）+ 重建点/秒 vs ν（Fig.13）",
             "§6.4：processing latency vs κ / reconstructed events per sec vs ν")

    out_dir = K.ensure_dir(out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "experiments"))

    # ---- A：Fig.11 ----
    print(f"[A] Fig.11：{FIG11_PARTICLES} 粒子, seed_offset={FIG11_SEED}"
          f"（与 step8 的 10P 同源）, κ={KAPPAS}")
    print("    κ 作用在原始事件上（filter 之后、match 之前）")
    rows = sweep_kappa(verbose=verbose)

    with open(os.path.join(out_dir, "latency_kappa.csv"), "w", encoding="utf-8-sig") as f:
        f.write("kappa,events_per_sec_per_camera,matched_points,recon_points,"
                "L_proc_median_ms,L_proc_mean_ms,L_proc_p95_ms,"
                "recon_points_per_sec,latency_samples\n")
        for r in rows:
            f.write(f"{r['kappa']},{r['ev_per_sec']:.1f},{r['n_matched']},{r['n_points']},"
                    f"{r['p50']:.6f},{r['mean']:.6f},{r['p95']:.6f},"
                    f"{r['pts_per_sec']:.3f},{r['n_samples']}\n")

    # ---- B：Fig.13 ----
    print(f"\n[B] Fig.13：{FIG13_PARTICLES} 粒子, seed_offset={FIG13_SEED}, "
          f"ν 扫 K.C.NU_GRID（{len(K.C.NU_GRID)} 档）")
    nu_rows, n_matched_1p, dur_1p = nu_scan(verbose=verbose)

    with open(os.path.join(out_dir, "events_per_sec_nu.csv"), "w", encoding="utf-8-sig") as f:
        f.write("nu,surviving_points,recon_points_per_sec\n")
        for r in nu_rows:
            f.write(f"{r['nu']},{r['n_survive']},{r['pts_per_sec']:.6f}\n")

    _plot_fig11(rows, out_dir)
    _plot_fig13(nu_rows, out_dir)

    if verbose:
        a = rows[0]
        last = rows[-1]
        print(f"\n[锚点] κ=1.0 @10P：匹配 {a['n_matched']}（应 {ANCHOR_MATCHED}）"
              f" / 重建点 {a['n_points']}（应 {ANCHOR_POINTS}）"
              f" / L_proc 中位 {a['p50']:.3f} ms（应≈{ANCHOR_L_PROC_MEDIAN_MS}）")
        print(f"[下降] L_proc 中位 κ 1.0→{last['kappa']:.1f}：{a['p50']:.3f} → "
              f"{last['p50']:.3f} ms（×{a['p50'] / last['p50']:.1f}）")
        print(f"[锚点] Fig.13 ν=0.0 存活点 {nu_rows[0]['n_survive']}"
              f"（应等于 1P 匹配点数 {n_matched_1p}）")
        print(f"[out ] 已保存：{out_dir}\\fig11_latency_vs_kappa.png / latency_kappa.csv")
        print(f"               {out_dir}\\fig13_events_per_sec_vs_nu.png / events_per_sec_nu.csv")
    return {"kappa_rows": rows, "nu_rows": nu_rows,
            "n_matched_1p": n_matched_1p, "dur_1p": dur_1p}


# ==================================================================
#  出图
# ==================================================================

def _plot_fig11(rows, out_dir):
    """左：L_proc 中位 vs κ（纵轴对数）；右：重建点/秒 vs κ。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["kappa"] for r in rows]
    p50 = [r["p50"] for r in rows]
    pps = [r["pts_per_sec"] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax[0].plot(ks, p50, "o-", color="#185FA5", lw=1.8)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("subsampling fraction κ (on raw events)")
    ax[0].set_ylabel("median processing latency L_proc (ms)")
    ax[0].set_title("(a) L_proc vs κ, 10P  (Fig.11 analog)")
    ax[0].invert_xaxis()                     # κ 从 1.0 往左降，与论文"降载方向"一致
    ax[0].grid(alpha=.25, which="both")
    for x, y in zip(ks, p50):
        if np.isfinite(y):
            ax[0].annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                           xytext=(4, -10), fontsize=8)

    ax[1].plot(ks, pps, "s-", color="#d1495b", lw=1.8)
    ax[1].set_xlabel("subsampling fraction κ (on raw events)")
    ax[1].set_ylabel("reconstructed points / sec")
    ax[1].set_title("(b) reconstruction rate vs κ, 10P")
    ax[1].invert_xaxis()
    ax[1].grid(alpha=.25)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig11_latency_vs_kappa.png"), dpi=150)
    plt.close(fig)


def _plot_fig13(nu_rows, out_dir):
    """1P 下重建点/秒 vs ν，纵轴 log10（与论文原图一致）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nu = [r["nu"] for r in nu_rows]
    pps = [r["pts_per_sec"] if r["pts_per_sec"] > 0 else np.nan for r in nu_rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(nu, pps, "o-", color="#2f7a4f", lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("quality threshold ν")
    ax.set_ylabel("reconstructed events / sec (log10)")
    ax.set_title("Reconstruction rate vs ν, 1P  (Fig.13 analog)")
    ax.grid(alpha=.25, which="both")
    for x, y, r in zip(nu, pps, nu_rows):
        if np.isfinite(y):
            ax.annotate(f"{r['n_survive']}", (x, y), textcoords="offset points",
                        xytext=(4, -10), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig13_events_per_sec_vs_nu.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Step9 延迟 vs κ（Fig.11）+ 点/秒 vs ν（Fig.13）")
    ap.add_argument("--out", default=None, help="输出目录（默认 py_repro/experiments）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_step(args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
