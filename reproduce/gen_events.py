# -*- coding: utf-8 -*-
"""合成双目事件流（替代真实 DAVIS346 采集）。
对应论文4节：场景中的反射粒子 -> 两台事件相机异步输出带时间戳的事件。
输出：左/右事件数组 (t_us, x, y, pol)，以及真值3D轨迹（用于第6节指标）。
"""
import numpy as np
import config as C


def _project(xyz, side):
    """针孔投影。side='l' 左相机在 -b/2，'r' 右相机在 +b/2。"""
    X, Y, Z = xyz
    cam_x = -C.BASELINE / 2 if side == "l" else C.BASELINE / 2
    u = C.F_PX * (X - cam_x) / Z + C.IMG_W / 2
    v = C.F_PX * Y / Z + C.IMG_H / 2
    return u, v


def _trajectory(t, pid):
    """平滑3D轨迹（类Lissajous），距离在0.4~1.0m，对应论文实验距离档。"""
    rng = np.random.default_rng(C.SEED + pid)
    ph = rng.uniform(0, 2 * np.pi, 3)
    w = rng.uniform(0.3, 0.8, 3)
    x = 0.15 * np.sin(w[0] * t + ph[0])
    y = 0.12 * np.cos(w[1] * t + ph[1])
    z = 0.7 + 0.3 * np.sin(w[2] * t + ph[2])
    return np.array([x, y, z])


def generate(n_particles=1, seed_offset=0):
    rng = np.random.default_rng(C.SEED + seed_offset)
    steps = int(C.DURATION_S / C.DT_S)
    ts_us = np.arange(steps) * C.DT_S * 1e6

    left, right, gt = [], [], []
    # 热像素位置（噪声源，论文dv噪声滤波要去掉）
    hot = np.column_stack([
        rng.integers(0, C.IMG_W, C.HOT_PIXELS),
        rng.integers(0, C.IMG_H, C.HOT_PIXELS),
    ])

    for pid in range(n_particles):
        for k, t in enumerate(np.arange(steps) * C.DT_S):
            xyz = _trajectory(t, pid)
            gt.append((ts_us[k], *xyz))
            for side, bucket in [("l", left), ("r", right)]:
                u, v = _project(xyz, side)
                # 模拟粒子扫过多个像素：每步发 N_PER_STEP 个邻近事件
                #（真实 DAVIS346 单粒子数千 ev/s；旧值 1 时互斥匹配后每帧只剩单点，
                #  凑不成 DBSCAN 的 minPts=2 簇 —— 2026-10-21 配套互斥修复）
                for _ in range(C.N_PER_STEP):
                    uq = int(np.clip(round(u + rng.normal(0, 0.6)), 0, C.IMG_W - 1))
                    vq = int(np.clip(round(v + rng.normal(0, 0.6)), 0, C.IMG_H - 1))
                    bucket.append((ts_us[k] + rng.uniform(-30, 30), uq, vq, 1))
                # 突发性(burstiness)：论文 Fig.10，偶尔多吐一簇邻近事件
                if rng.random() < 0.05:
                    for _ in range(rng.integers(2, 6)):
                        bucket.append((
                            ts_us[k] + rng.uniform(0, 20),
                            int(np.clip(uq + rng.integers(-3, 4), 0, C.IMG_W - 1)),
                            int(np.clip(vq + rng.integers(-3, 4), 0, C.IMG_H - 1)),
                            1,
                        ))

    # 随机噪声事件 + 热像素事件
    for bucket in (left, right):
        for _ in range(steps * C.NOISE_RATE):
            bucket.append((
                rng.uniform(0, C.DURATION_S * 1e6),
                rng.integers(0, C.IMG_W), rng.integers(0, C.IMG_H),
                rng.choice([-1, 1]),
            ))
        for k in range(steps // 3):
            h = hot[rng.integers(0, C.HOT_PIXELS)]
            bucket.append((rng.uniform(0, C.DURATION_S * 1e6), h[0], h[1], 1))

    left = np.array(left, dtype=float)
    right = np.array(right, dtype=float)
    gt = np.array(gt, dtype=float)
    # 按时间排序
    left = left[np.argsort(left[:, 0])]
    right = right[np.argsort(right[:, 0])]
    return left, right, gt


if __name__ == "__main__":
    l, r, g = generate(1)
    print("左事件:", l.shape, "右事件:", r.shape, "真值点:", g.shape)
