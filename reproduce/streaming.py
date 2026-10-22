# -*- coding: utf-8 -*-
"""传输层处理（对应论文6.4节：均匀随机子采样κ + 基于质量分数s的拥塞事件滤波）。
- 子采样：按比例κ均匀随机保留事件点（论文：不损失精度地加速）。
- 质量分数s：论文口径 = 式(2) 匹配分数（越小越好），过滤保留 s <= ν 的点。
  2026-10-21 修正（审查发现A，销坑#5）：原先用「局部密度当好度、保留 q >= ν」的代理，
  方向与论文相反；现在 match_and_triangulate 直接返回式(2) 分数列 s，
  filter_by_quality 改为论文口径「保留 s <= ν」→ ν 越大保留越多，与 Fig.12 同轴。
  quality_scores（局部密度代理）保留仅作教学对照，不再用于 ν 滤波。
"""
import numpy as np
import config as C


def quality_scores(pts, radius=0.02):
    """【已废弃于 ν 滤波，仅留教学对照】局部密度近似质量分，归一化到[0,1]。

    ⚠ 这不是论文的 s。论文的 s 是式(2) 匹配分数（越小越好）。
    """
    if len(pts) == 0:
        return pts
    q = np.zeros(len(pts))
    for i, p in enumerate(pts):
        q[i] = np.sum(np.linalg.norm(pts - p, axis=1) <= radius)
    if q.max() > q.min():
        q = (q - q.min()) / (q.max() - q.min())
    return q


def subsample(pts, kappa):
    n = len(pts)
    keep = np.random.default_rng(C.SEED).random(n) < kappa
    return pts[keep]


def filter_by_quality(scored_pts, nu):
    """论文口径 ν 滤波：scored_pts 为 (M,5)=[t,X,Y,Z,s]，保留 s <= nu（s 越小越好）。"""
    if len(scored_pts) == 0:
        return scored_pts
    return scored_pts[scored_pts[:, 4] <= nu]
