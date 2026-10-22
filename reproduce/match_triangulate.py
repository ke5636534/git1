# -*- coding: utf-8 -*-
"""立体匹配 + 三角测量（对应论文4节：事件匹配评分式(2)，OpenCV标准三角测量）。
式(2): s_{k,i} = α|t_k-t_i|/tmax + β|y_k-y_i|/ytol + γ|d_k-d_i|/(dmax-dmin)
"""
import numpy as np
import config as C

CX, CY = C.IMG_W / 2, C.IMG_H / 2
DMID = (C.D_MIN_PX + C.D_MAX_PX) / 2


def _score(dt, dy, disp):
    """论文式(2)：α|Δt|/tmax + β|Δy|/ytol + γ|d-DMID|/(½(dmax-dmin))，第三项分母为半距。"""
    return (C.W_T * abs(dt) / C.T_MAX_US
            + C.W_Y * abs(dy) / C.Y_TOL_PX
            + C.W_D * abs(disp - DMID) / C.D_SCORE_DEN)


def match_and_triangulate(left, right, return_scores=False):
    """返回 3D 点数组 (M,4)=[t,X,Y,Z]；return_scores=True 时返回 (M,5)=[t,X,Y,Z,s]。

    s = 式(2) 匹配分数（越小越好）。2026-10-21 起 ν 滤波用论文口径「保留 s ≤ ν」，
    所以这里必须把分数带出来（审查发现A：原先算完就丢，导致下游全用局部密度代理）。
    互斥：同一右事件只能被一个左事件认领（used 集合，审查发现F，销坑#13）。
    """
    rt = right[:, 0]
    order = np.argsort(rt)
    rt_sorted = rt[order]
    out = []
    used = set()                               # 已被认领的右事件原始下标（互斥）
    for tl, ul, vl, _ in left:
        lo = np.searchsorted(rt_sorted, tl - C.T_MAX_US, side="left")
        hi = np.searchsorted(rt_sorted, tl + C.T_MAX_US, side="right")
        win = order[lo:hi]                     # 窗口内右事件的【原始下标】
        cand = right[win]
        if len(cand) == 0:
            continue
        dy = np.abs(cand[:, 2] - vl)
        cand, win, disp = cand[dy <= C.Y_TOL_PX], win[dy <= C.Y_TOL_PX], ul - cand[dy <= C.Y_TOL_PX, 1]
        if len(cand) == 0:
            continue
        ok = (disp >= C.D_MIN_PX) & (disp <= C.D_MAX_PX)
        cand, win, disp = cand[ok], win[ok], disp[ok]
        if len(cand) == 0:
            continue
        scores = (C.W_T * np.abs(tl - cand[:, 0]) / C.T_MAX_US
                  + C.W_Y * np.abs(vl - cand[:, 2]) / C.Y_TOL_PX
                  + C.W_D * np.abs(disp - DMID) / C.D_SCORE_DEN)
        # 贪心 argmin + 互斥：按分数从小到大找第一个还没被认领的右事件（发现F）。
        # 注：三关卡过滤后总分恒 <=1.0（0.5+0.4+0.1），不存在 ">1.0 拒绝" 分支（发现C 死代码已删）；
        #     论文真正的低分拒绝在 §6.4 的 ν 滤波（阈值 0.5/0.25/0.1）。
        j = -1
        for jj in np.argsort(scores):
            orig = int(win[jj])
            if orig not in used:
                j = int(jj)
                used.add(orig)
                break
        if j < 0:                              # 候选全被前面的左事件认领光
            continue
        tr, ur, vr, _ = cand[j]
        # 标准立体三角测量：Z = f*b/d
        Z = C.F_PX * C.BASELINE / disp[j]
        X = (ul - CX) / C.F_PX * Z - C.BASELINE / 2
        Y = (vl - CY) / C.F_PX * Z
        out.append((tr, X, Y, Z, float(scores[j])))
    arr = np.array(out) if out else np.zeros((0, 5))
    return arr if return_scores else arr[:, :4]


if __name__ == "__main__":
    import os
    os.environ.setdefault("MPLCONFIGDIR",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", ".mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import gen_events, filter as flt

    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(HERE, "output")
    os.makedirs(OUT, exist_ok=True)

    f_l = os.path.join(OUT, "left_filtered.npy")
    f_r = os.path.join(OUT, "right_filtered.npy")
    if os.path.exists(f_l) and os.path.exists(f_r):
        l = np.load(f_l); r = np.load(f_r)          # 左右分开的 .npy（无统一 npz）
        print("loaded from output/: left", l.shape, " right", r.shape)
    else:
        l, r, _ = gen_events.generate(1)
        l, r = flt.filter_events(l), flt.filter_events(r)
        print("(output/*.npy 不存在，改用 generate+filter 现算) left", l.shape, " right", r.shape)

    # ---- 时间戳对齐检查：两相机若不同步，用左时间戳对齐右 ----
    lt0, lt1 = float(l[:, 0].min()), float(l[:, 0].max())
    rt0, rt1 = float(r[:, 0].min()), float(r[:, 0].max())
    print(f"left  t range : [{lt0:.1f}, {lt1:.1f}] us")
    print(f"right t range : [{rt0:.1f}, {rt1:.1f}] us")
    delta = lt0 - rt0
    if abs(delta) > C.T_MAX_US:
        print(f"两相机时间戳未对齐 (delta={delta:.1f} us > tmax)，将右时间戳整体平移 {delta:.1f} us 对齐左")
        r = r.copy(); r[:, 0] += delta
    else:
        print(f"两相机时间戳基本对齐 (delta={delta:.1f} us <= tmax={C.T_MAX_US} us)，无需平移")

    # ---- 每帧（按左时间戳）事件数 ----
    step_us = C.DT_S * 1e6
    idx = (l[:, 0] // step_us).astype(int)
    keys, cnts = np.unique(idx, return_counts=True)
    print(f"\n[每帧左相机事件数] 共 {len(keys)} 帧 (frame = 左时间戳 // {step_us:.0f}us)")
    line = []
    for k, c in zip(keys, cnts):
        line.append(f"f{k}:{c}")
        if len(line) == 8:
            print("  " + "  ".join(line)); line = []
    if line:
        print("  " + "  ".join(line))

    # ---- 匹配 + 三角测量（论文4.1：±150us / |dy|<=15px / d in [5,150]px，searchsorted 取候选）----
    pts = match_and_triangulate(l, r)
    n_match = len(pts)
    print(f"\n匹配成功事件数: {n_match}")

    if n_match:
        np.save(os.path.join(OUT, "match_points.npy"), pts)
        with open(os.path.join(OUT, "match_points.csv"), "w", encoding="utf-8-sig") as f:
            f.write("t_us,X_m,Y_m,Z_m\n")
            for row in pts:
                f.write(",".join(f"{v:.6f}" for v in row) + "\n")

        Z = pts[:, 3]
        print(f"Z深度(m): min={Z.min():.4f}  max={Z.max():.4f}  "
              f"median={np.median(Z):.4f}  std={Z.std():.4f}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(Z, bins=30, color="steelblue", edgecolor="black")
        ax.axvline(np.median(Z), color="red", ls="--", label=f"median={np.median(Z):.3f} m")
        ax.set_xlabel("Z depth (m)"); ax.set_ylabel("count")
        ax.set_title(f"Z depth distribution (N={n_match})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "match_Z.png"), dpi=130)
        plt.close(fig)
        print("已保存: output/match_points.npy, output/match_points.csv, output/match_Z.png")

        print(f"\n>>> 成功匹配点数={n_match}  Z范围=[{Z.min():.4f}, {Z.max():.4f}] m")
    else:
        print("\n>>> 成功匹配点数=0  (无匹配点)")
