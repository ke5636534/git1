# -*- coding: utf-8 -*-
"""点云聚类（对应论文4节 DBSCAN：同簇点距约1cm，每簇至少2点，单点/离群点丢弃）。
这里用简化的半径贪心聚类，代替 sklearn DBSCAN。
"""
import numpy as np
import config as C


def cluster_points(pts):
    if len(pts) == 0:
        return pts
    n = len(pts)
    visited = np.zeros(n, dtype=bool)
    centers = []
    for i in range(n):
        if visited[i]:
            continue
        d = np.linalg.norm(pts - pts[i], axis=1)
        members = np.where(d <= C.CLUSTER_RADIUS)[0]
        visited[members] = True
        if len(members) >= C.MIN_POINTS:
            centers.append(pts[members].mean(axis=0))
    return np.array(centers) if centers else np.zeros((0, 3))


def cluster_timed(pts, step_us=None):
    """输入 (M,4)=[t_us,X,Y,Z]：按时间步分帧 -> 逐帧聚类 -> 输出簇心数组。
    返回 (K,4)=[t_us,X,Y,Z]，t_us 取该帧时间戳均值（保证下游按同一时间步回分帧不错位）。
    """
    step_us = C.DT_S * 1e6 if step_us is None else step_us
    pts = np.asarray(pts, dtype=float).reshape(-1, 4)
    if len(pts) == 0:
        return np.zeros((0, 4))
    buckets = {}
    for k, row in zip((pts[:, 0] // step_us).astype(int), pts):
        buckets.setdefault(int(k), []).append(row)
    out = []
    for k in sorted(buckets):
        arr = np.asarray(buckets[k], dtype=float)
        ctrs = cluster_points(arr[:, 1:])
        if len(ctrs) == 0:
            continue
        tk = float(arr[:, 0].mean())
        for c in ctrs:
            out.append([tk, *c])
    return np.array(out) if out else np.zeros((0, 4))


if __name__ == "__main__":
    import os

    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(HERE, "output")
    os.makedirs(OUT, exist_ok=True)

    src = os.path.join(OUT, "match_points.npy")
    if not os.path.exists(src):
        raise SystemExit("缺少 output/match_points.npy，请先运行 filter.py + match_triangulate.py")
    pts = np.load(src)
    n_in = len(pts)
    n_frames = len(np.unique((pts[:, 0] // (C.DT_S * 1e6)).astype(int))) if n_in else 0

    centers = cluster_timed(pts)

    print(f"输入匹配点 : {n_in}")
    print(f"时间步分帧 : {n_frames} 帧 (step = {C.DT_S * 1e6:.0f} us)")
    print(f"聚类参数   : 半径 {C.CLUSTER_RADIUS * 100:.0f} cm, 每簇至少 {C.MIN_POINTS} 点"
          f"（单点/离群丢弃）")
    print(f"输出簇心数 : {len(centers)}")

    np.save(os.path.join(OUT, "cluster_centers.npy"), centers)
    with open(os.path.join(OUT, "cluster_centers.csv"), "w", encoding="utf-8-sig") as f:
        f.write("t_us,X_m,Y_m,Z_m\n")
        for row in centers:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")
    print("已保存: output/cluster_centers.npy, output/cluster_centers.csv")
