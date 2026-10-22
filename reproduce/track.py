# -*- coding: utf-8 -*-
"""全局最近邻跟踪 GNN + 卡尔曼滤波（对应论文4节 MATLAB trackerGNN：预测/分配/确认/删除）。
确认 (1,3)：最近 3 步窗口内 >=1 次检测即确认；
删除 (5,12)：最近 12 步窗口内 >=5 次漏检即删除（滑动窗口计数，不要求连续）。
2026-10-21 修正两处：① 新轨迹出生帧不再背漏检（差一错误）；② 删除/确认改滑窗判定（原为连续漏检12步）。
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import config as C

GATE = 0.25  # m，匹配门限（论文未给值，此为假设值，见台账）


class Track:
    def __init__(self, point, dt, tid):
        self.tid = tid
        self.x = np.array([*point, 0, 0, 0], dtype=float)
        self.P = np.eye(6) * 0.05
        self.dt = dt
        self.hits = 1
        self.hit_hist = [1]     # 每步是否命中（1/0），供滑窗确认判定
        self.miss_hist = [0]    # 每步是否漏检（1/0），供滑窗删除判定
        # CONFIRM=(1,3)：最近 3 步窗口内 >=1 次检测即确认（新轨迹出生帧计 1 次命中）
        self.confirmed = sum(self.hit_hist[-C.CONFIRM[1]:]) >= C.CONFIRM[0]
        self.history = [point.copy()]

    def predict(self):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = self.dt
        self.x = F @ self.x
        Q = np.eye(6) * C.PROCESS_NOISE
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
        R = np.eye(3) * C.MEAS_NOISE
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P
        self.hits += 1
        self.hit_hist.append(1)
        self.miss_hist.append(0)
        if sum(self.hit_hist[-C.CONFIRM[1]:]) >= C.CONFIRM[0]:
            self.confirmed = True
        self.history.append(self.x[:3].copy())

    def miss_step(self):
        """漏检一步：只记历史，不做删除判定（判定统一在 _should_delete）。"""
        self.hit_hist.append(0)
        self.miss_hist.append(1)

    def _should_delete(self):
        """DELETE=(5,12)：最近 12 步窗口内漏检 >=5 次即删（滑窗计数，不要求连续）。"""
        return sum(self.miss_hist[-C.DELETE[1]:]) >= C.DELETE[0]

    @property
    def pos(self):
        return self.x[:3]


def _gnn(frames, dt=C.DT_S, keep_deleted=False):
    """核心 GNN 循环，返回 (存活轨迹, 已删除轨迹)。
    keep_deleted=True 时保留被删除轨迹，否则按 trackerGNN 语义丢弃。
    """
    tracks = []
    lost = []
    next_id = 0
    for dets in frames:
        dets = np.atleast_2d(dets)
        for tr in tracks:
            tr.predict()
        # 代价矩阵
        n, m = len(tracks), len(dets)
        cost = np.full((n, m), 1e9)
        for i, tr in enumerate(tracks):
            for j, z in enumerate(dets):
                d = np.linalg.norm(tr.pos - z)
                if d <= GATE:
                    cost[i, j] = d
        assigned_t, assigned_d = set(), set()
        if n and m:
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e8:
                    tracks[i].update(dets[j])
                    assigned_t.add(i); assigned_d.add(j)
        # 未匹配轨迹 -> 漏检（★ 必须在新轨迹 append 之前做：
        #   否则 enumerate(tracks) 会把刚出生的新轨迹也遍历到，
        #   每条新轨迹出生帧就白背一次 miss —— 差一错误，审查发现D）
        for i, tr in enumerate(tracks):
            if i not in assigned_t:
                tr.miss_step()
        # 删除判定（滑窗 (5,12)）
        alive = []
        for tr in tracks:
            if not tr._should_delete():
                alive.append(tr)
            elif keep_deleted and tr.confirmed:
                lost.append(tr)
        tracks = alive
        # 未匹配检测 -> 新轨迹（出生帧自带 1 次命中，不背 miss）
        for j in range(m):
            if j not in assigned_d:
                tracks.append(Track(dets[j], dt, next_id)); next_id += 1
    return tracks, lost


def track_frames(frames, dt=C.DT_S, keep_deleted=False):
    """frames: list，每个元素是该帧检测点 (M,3)。返回确认轨迹点集合 (N,3)。"""
    alive, lost = _gnn(frames, dt, keep_deleted)
    pool = alive + lost
    pts = []
    for tr in pool:
        if tr.confirmed:
            pts.extend(tr.history)
    return np.array(pts) if pts else np.zeros((0, 3))


def track_trajectories(frames, dt=C.DT_S, keep_deleted=True):
    """同上，额外返回每条点所属的轨迹 id。返回 (pts (N,3), ids (N,), 轨迹条数)。"""
    alive, lost = _gnn(frames, dt, keep_deleted)
    pool = alive + lost
    pts, ids = [], []
    for tr in pool:
        if tr.confirmed:
            for p in tr.history:
                pts.append(p)
                ids.append(tr.tid)
    arr = np.array(pts) if pts else np.zeros((0, 3))
    idarr = np.array(ids, dtype=int) if ids else np.zeros(0, dtype=int)
    return arr, idarr, len(pool)


def frames_from_centers(centers, step_us=None):
    """簇心数组 (M,4)=[t_us,X,Y,Z] -> 帧列表；按时间步补空帧，保持真实时间轴。"""
    step_us = C.DT_S * 1e6 if step_us is None else step_us
    centers = np.asarray(centers, dtype=float).reshape(-1, 4)
    if len(centers) == 0:
        return []
    idx = (centers[:, 0] // step_us).astype(int)
    buckets = {k: [] for k in range(int(idx.min()), int(idx.max()) + 1)}
    for k, row in zip(idx, centers[:, 1:]):
        buckets[int(k)].append(row)
    return [np.asarray(buckets[k], dtype=float).reshape(-1, 3) for k in sorted(buckets)]


if __name__ == "__main__":
    import os

    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(HERE, "output")
    os.makedirs(OUT, exist_ok=True)

    src = os.path.join(OUT, "cluster_centers.npy")
    if not os.path.exists(src):
        raise SystemExit("缺少 output/cluster_centers.npy，请先运行 cluster.py")
    centers = np.load(src)

    frames = frames_from_centers(centers)
    n_det = sum(1 for f in frames if len(f))
    pts, ids, n_trk = track_trajectories(frames, keep_deleted=True)

    print(f"输入簇心   : {len(centers)}  帧数: {len(frames)}（有检测 {n_det} 帧）")
    print(f"跟踪参数   : 确认 {C.CONFIRM}（最近{C.CONFIRM[1]}步>={C.CONFIRM[0]}次命中）, "
          f"删除 {C.DELETE}（最近{C.DELETE[1]}步>={C.DELETE[0]}次漏检, 滑窗）, "
          f"门限 {GATE} m")
    print(f"输出轨迹   : {n_trk} 条, {len(pts)} 个点")

    np.save(os.path.join(OUT, "track_points.npy"), pts)
    np.save(os.path.join(OUT, "track_ids.npy"), ids)
    print("已保存: output/track_points.npy, output/track_ids.npy")
