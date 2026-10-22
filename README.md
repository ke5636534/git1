# Remote Particle Trajectory Tracking — 复现代码与原文文献

复现论文 **《Remote Particle Trajectory Tracking using Event-Based Vision Streams》**
（MMSys '26，Kolling / Freeman / Rizk，DOI 10.1145/3793853.3795751）的复现代码，
连同原文文献一并归档。

## 论文一句话

用**两台事件相机（DAVIS346，14 cm 基线）**远距离追踪空中的**反射粒子**：
靠事件级双目匹配 + 三角测量还原粒子 3D 轨迹。核心卖点是**低延迟**——
事件到达即处理，不用等整帧曝光。

## 目录结构

```
repo_upload/
├── reproduce/                  ★ 算法本体（单文件流水线，共 8 个 .py）
│   ├── config.py               全部参数（基线/焦距/三关卡/聚类/rho/粒子数）
│   ├── gen_events.py           合成双目事件流 + 真值
│   ├── filter.py               热像素 + 不应期去噪
│   ├── match_triangulate.py    式(2) 双目匹配 + Z = f·B/d 三角测量
│   ├── cluster.py              1 cm 贪心聚类
│   ├── track.py                GNN + 恒速卡尔曼 + 匈牙利全局配对
│   ├── metrics.py              式(3)(4)(7) 精度指标
│   ├── streaming.py            质量分数 / 子采样
│   └── run_all.py              一键全流程（pipeline() 被下游复用）
├── py_repro/                   ★★ 分步复现（当前主线，10-21 更新）
│   ├── PTV_启动器.py           统一 GUI 入口（选数据集/步骤/输出目录）
│   ├── _core.py                算法转接头（把 ../reproduce 加进 sys.path）
│   ├── evt3_decoder.py         EVT3 格式解码
│   ├── step0_parse_dataset.py  Step0 解析与翻译数据集 → npz + meta
│   ├── step1_filter_events.py  Step1 去噪
│   ├── step2_match_triangulate.py  Step2 匹配 + 三角测量
│   ├── step3_cluster.py        Step3 聚类
│   ├── step4_track.py          Step4 跟踪
│   ├── step5_metrics.py        Step5 评估
│   ├── step6_streaming.py      Step6 流式实验 κ/ν（Fig.12）
│   ├── step7_experiments.py    Step7 粒子数扫描（Fig.6）
│   ├── step8_latency.py        Step8 延迟分解（Fig.8）
│   ├── step9_latency_kappa.py  Step9 延迟 vs κ（Fig.11）/ 点·秒 vs ν（Fig.13）
│   ├── step10_congestion.py    Step10 拥塞实验（Fig.14）
│   ├── test_all_steps.py       全步骤测试（36 项断言）
│   ├── 复现进度台账.md          ★ 干到哪一步看这里
│   ├── 参考答案/                锚点重建参考数据
│   └── _guide/                 精讲素材生成脚本与片段
└── paper/                      ★ 原文文献
    ├── Remote Particle Trajectory Tracking using Event-Based Vision Streams.pdf          英文原文
    ├── Remote Particle Trajectory Tracking using Event-Based Vision Streams_中文版.pdf    中文版
    └── 【翻译】Remote Particle Trajectory Tracking using Event-Based Vision Streams.pdf    全篇翻译
```

## 核心概念

| 概念 | 含义 |
|---|---|
| 事件相机 (DVS) | 不拍整帧，只报告「哪个像素、什么时刻、变亮 ON=+1 或变暗 OFF=−1」。一条事件 = `(t, x, y, 极性)`，时间单位 µs |
| 为什么用它追粒子 | 粒子是瞬时光点，帧相机曝光期间它是拖影；事件相机只记「亮度变了」，没有曝光窗口 → 没有运动模糊 |
| 双目匹配（式 2） | 判断左右相机的事件是不是同一次闪烁。三道关卡：Δt ≤ 150 µs、Δy ≤ 15 px、视差 d ∈ [5, 150] px。打分 `α·Δt + β·Δy + γ·Δd`，α=0.5 / β=0.4 / γ=0.1 |
| 三角测量 | `Z = f·B/d`（f=300 px，B=0.14 m，d=视差像素）。视差 1 px 量化误差 → Z 误差约 1.2 cm |
| 聚类 | 按 1 cm 半径抱团、每簇至少 2 点 |
| 跟踪 (trackerGNN) | 帧间恒速卡尔曼预测 + 匈牙利算法全局最优配对；确认阈值 (1,3)、删除阈值 (5,12) |
| 精度指标 | Cov(ρ)=召回、Prec(ρ)=准确、Chamfer=双向平均最近邻距离，ρ=5 cm |

## 运行

```bash
# 解释器（唯一装了 scipy 的，Step4 匈牙利算法依赖它）
python = D:\Users\admuj\anaconda3\envs\opencv_env\python.exe

cd py_repro
python PTV_启动器.py        # GUI：选数据集 / 步骤 / 输出目录
python test_all_steps.py   # 全步骤测试，36/36 PASS
```

**复现状态**：仿真 1 粒子跑通全链，Cov 0.985 / Prec 1.000 / Chamfer 6.73 mm；
已复刻 Fig.6（粒子数扫描）、Fig.8（延迟分解）、Fig.11（L_proc vs κ）、
Fig.12（κ/ν 精度）、Fig.13（点/秒 vs ν，跨 2.6 个数量级）。

## 上传说明

- 上传范围：这两天（2026-10-21 / 10-22）复现的代码 + 原文文献
- 未上传：`__pycache__`、数据产物（npz / csv / png）、STELLA 子仓库、
  `.git`、IDE 配置、`.mpl` 字体缓存
- 本地完整工程仍在 `E:\paper\粒子追踪\Remote Particle Trajectory Tracking using Event-Based Vision Streams\`
