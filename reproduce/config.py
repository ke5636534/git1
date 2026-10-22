# -*- coding: utf-8 -*-
"""全局参数。数值对应论文第4-6节，做了简化，单位见注释。
论文: Remote Particle Trajectory Tracking using Event-Based Vision Streams (MMSys'26)
"""
import numpy as np

# ---- 双目相机（论文4节：两台 DAVIS346，并排同高，间距14cm）----
BASELINE = 0.14            # m，左右相机间距
F_PX = 300.0               # 简化焦距（像素）
IMG_W, IMG_H = 346, 260    # DAVIS346 分辨率

# ---- 立体匹配阈值（论文4节：tmax=150us, ytol=15px, dmin=5px, dmax=150px）----
T_MAX_US = 150.0
Y_TOL_PX = 15.0
D_MIN_PX = 5.0
D_MAX_PX = 150.0
# 匹配评分权重（论文式(2)原文：α=0.5 时间差、β=0.4 纵向偏移、γ=0.1 视差偏差）
W_T, W_Y, W_D = 0.5, 0.4, 0.1
# 式(2)第三项目分母：论文为 1/2(dmax-dmin)
D_SCORE_DEN = (D_MAX_PX - D_MIN_PX) / 2

# ---- 聚类（论文4节 DBSCAN：簇内点距约1cm，每簇至少2点）----
CLUSTER_RADIUS = 0.01
MIN_POINTS = 2

# ---- 跟踪器（论文4节 MATLAB trackerGNN：确认(1,3)，删除(5,12)）----
# 语义 = MATLAB trackerGNN 滑动窗口计数（2026-10-21 修正，原为"连续漏检12步"的简化）：
#   CONFIRM=(1,3)：最近 3 步窗口内 >=1 次检测即确认
#   DELETE=(5,12)：最近 12 步窗口内 >=5 次漏检即删除（不要求连续）
CONFIRM = (1, 3)
DELETE = (5, 12)
PROCESS_NOISE = 1e-3
MEAS_NOISE = 1e-3

# ---- 评估半径 ρ=5cm（论文6.1节）----
RHO = 0.05

# ---- 实验扫描（论文Fig.6/7：粒子数 {1,2,3,5,10}P）----
PARTICLE_COUNTS = [1, 2, 3, 5, 10]
# 质量评分阈值扫描（论文Fig.12：ν∈[0,1]）
NU_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)

# ---- 场景 ----
# 每步每侧每粒子发的事件数：真实 DAVIS346 里一个粒子会扫过多个像素，
# 单粒子就能产生数千 ev/s（论文 1P 为 5×10³ ev/s）。旧值 1 会让互斥匹配后
# 每帧只剩单点、凑不成簇（MIN_POINTS=2），2026-10-21 随互斥修复配套提到 3。
N_PER_STEP = 3
DURATION_S = 10.0
DT_S = 0.01                # 每步10ms
NOISE_RATE = 6             # 每步随机噪声事件数
HOT_PIXELS = 5             # 固定热像素个数
SEED = 42
