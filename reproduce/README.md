# 远程粒子轨迹跟踪（事件视觉流）软件版复现骨架

对应论文：*Remote Particle Trajectory Tracking using Event-Based Vision Streams* (MMSys '26)

这是一个**轻量教学复现骨架**：用合成的双目事件流代替真实 DAVIS346 相机和 OptiTrack 动捕，
把论文从"事件采集 → 噪声滤波 → 立体匹配 → 三角测量 → 聚类 → 跟踪 → 精度评估 → 传输滤波"
整条流水线用纯 Python（numpy/scipy/matplotlib）跑通，帮助逐段读懂论文。
**不需要硬件、不需要联网下载、不写 C 盘。**

## 一、文件与论文章节对照

| 文件 | 对应论文章节 | 做的事 |
|---|---|---|
| `config.py` | 第4节实验参数 | 相机间距14cm、匹配阈值 tmax=150us/ytol=15px/d=5~150px、式(2)权重 α=0.5(时间) β=0.4(纵向) γ=0.1(视差)、聚类1cm、跟踪确认/删除阈值、评估半径ρ=5cm |
| `gen_events.py` | 第4节 事件采集 | 合成平滑3D轨迹，投影到左右相机生成带时间戳事件，加入噪声/热像素/突发(burst) |
| `filter.py` | 4节 DVS噪声滤波[19] | 热像素删除 + 不应期（同像素最小间隔） |
| `match_triangulate.py` | 4节 事件匹配+三角测量 | 式(2)评分匹配左右事件；标准立体公式 Z=f·b/d 三角测量 |
| `cluster.py` | 4节 DBSCAN | 半径1cm贪心聚类，每簇≥2点，去掉单点/离群 |
| `track.py` | 4节 MATLAB trackerGNN | 恒定速度卡尔曼 + 全局最近邻(Hungarian) + 确认/删除管理（演示模块） |
| `metrics.py` | 6.1节 式(3)(4)(7) | 覆盖率Cov、精确率Prec、倒角距离Chamfer |
| `streaming.py` | 6.4节 | 均匀随机子采样κ + 质量分数s阈值ν滤波 |
| `run_all.py` | 第5-6节实验 | 一键跑粒子数扫描 + 子采样/质量阈值扫描，出图出CSV |

## 二、运行

```bash
cd reproduce
python run_all.py
```

依赖：numpy、scipy、matplotlib（本次环境已自带，未新安装任何包）。

## 三、输出（results/）

- `fig6_particle_count.png`：不同粒子数 {1,2,3,5,10}P 下的覆盖率/精确率/倒角距离（类比论文 Fig.6/7）
- `fig12_streaming.png`：子采样κ 与质量阈值ν 对覆盖率/精确率的影响（类比论文 Fig.12）
- `results_particle_count.csv`：粒子数扫描数值表

本次跑通的典型结果：覆盖率≈1.0、精确率≈0.98、倒角距离≈4mm；κ降到0.2时覆盖率下降而精确率基本不变，
ν升高时覆盖率单调下降——与论文"子采样加速且不损失精度、拥塞时按质量过滤"的结论方向一致。

## 四、与真实实验的差距（重要）

本骨架是**理想合成数据**，目的是跑通流程、理解每一步在干什么；论文真实结果依赖：
1. 硬件：两台 DAVIS346、OptiTrack Prime 13 动捕真值、0.8/2cm反射粒子、0.5m腔室、1Gbps专网；
2. 真实事件有更复杂的噪声、运动模糊和跨粒子混淆，所以论文里粒子数增多时精确率下降、倒角距离上升；
3. 跟踪器这里是最小参考实现，论文用的是 MATLAB trackerGNN 完整版本。

## 五、如何升级到真实数据（后续路线）

1. 用 iniVation **dv-gui**（免费）采集/读取 AEDAT 事件流，替换 `gen_events.py`；
2. 用 dv-gui 做完双目标定，把内外参接入 `match_triangulate.py` 的投影矩阵；
3. 三角测量可用 OpenCV `triangulatePoints` 替换简化公式；
4. 跟踪可换 MATLAB trackerGNN 或 Python 的实现；
5. 网络部分用开源 **TQUIC** 复现 CUBIC/BBR3/COPA 拥塞反馈；
6. 没有动捕真值时，覆盖率/精确率/倒角距离无法计算，只能看相对趋势。
