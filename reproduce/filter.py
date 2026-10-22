# -*- coding: utf-8 -*-
"""噪声滤波（对应论文4节：DVS噪声滤波器[19] + 热像素抑制）。
两步：1) 热像素删除（触发频率异常高的固定像素）；2) 不应期（同一像素最小间隔）。
"""
import numpy as np
import config as C

HOT_RATIO = 0.02        # 单像素事件占比超过该阈值视为热像素
REFRACT_US = 1000.0     # 同像素最小间隔1ms（简化）


def filter_events(ev):
    if len(ev) == 0:
        return ev
    t, x, y, pol = ev.T
    # 1) 热像素：统计每个像素触发次数
    # 键值用 65536 进制而非 x*IMG_H+y：真实数据（如 Gen4 720 行）y 超过 IMG_H=260 时
    # 旧键会把 (x,y) 和 (x,y+260) 并成一个像素。65536 对任何 y 都唯一，仿真结果不变。
    coords = x.astype(np.int64) * 65536 + y.astype(np.int64)
    keys, counts = np.unique(coords, return_counts=True)
    hot_set = set(keys[counts > len(ev) * HOT_RATIO].tolist()) if len(ev) else set()
    mask = np.array([c not in hot_set for c in coords])
    ev = ev[mask]

    # 2) 不应期：按像素分组，只保留间隔>=REFRACT_US的事件
    keep = np.ones(len(ev), dtype=bool)
    last = {}
    for i, (tt, xx, yy, _) in enumerate(ev):
        key = (int(xx), int(yy))
        if key in last and tt - last[key] < REFRACT_US:
            keep[i] = False
        else:
            last[key] = tt
    return ev[keep]


if __name__ == "__main__":
    import os
    os.environ.setdefault("MPLCONFIGDIR",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", ".mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import gen_events

    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(HERE, "output")
    os.makedirs(OUT, exist_ok=True)

    l, r, _ = gen_events.generate(1)
    fl = filter_events(l)
    fr = filter_events(r)

    print("滤波前左:", len(l), " 滤波后左:", len(fl))
    print("滤波前右:", len(r), " 滤波后右:", len(fr))

    np.save(os.path.join(OUT, "left_filtered.npy"), fl)
    np.save(os.path.join(OUT, "right_filtered.npy"), fr)

    for name, ev in (("left", fl), ("right", fr)):
        if len(ev) == 0:
            print(f"{name}: empty after filter")
            continue
        t0, t1 = float(ev[:, 0].min()), float(ev[:, 0].max())
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(ev[:, 1], ev[:, 2], c=ev[:, 0], s=2, cmap="viridis")
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
        ax.invert_yaxis()
        ax.set_title(f"{name} filtered  N={len(ev)}  t in [{t0:.1f}, {t1:.1f}] us")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"fig_{name}_filtered.png"), dpi=130)
        plt.close(fig)
        print(f"[{name}] timestamp range = [{t0:.1f}, {t1:.1f}] us  ({len(ev)} events)")
    print("saved to:", OUT)
