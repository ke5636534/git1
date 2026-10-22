# -*- coding: utf-8 -*-
r"""
_core.py —— 算法共享库的「转接头」
============================================================
本文件不写算法，只干一件事：把上层 `../reproduce` 目录加进 sys.path，
再把 7 个算法模块统一 import 进来，给 py_repro 下所有 step 脚本用。

★ 为什么这么设计（对齐 Lattice / RATE 的 py_repro 惯例）：
  "算法只写一份"。真正的实现仍然住在上层 reproduce/ 里，
  这里只是转个手，所以：
    - 改算法 -> 改 ../reproduce/xxx.py，两个入口（reproduce 原流程 + py_repro 分步）同时生效
    - 不会出现"两边代码不一样、结果对不上"的经典惨案

本文件导出的东西（保持原名，方便阅读）：
  C         = config.py     参数表（基线/焦距/匹配三关卡/聚类半径/rho...）
  ge        = gen_events    合成双目事件流 + 真值（Step0 的仿真数据源）
  flt       = filter        热像素 + 不应期去噪（Step1）
  mt        = match_triangulate  式(2)匹配评分 + 三角测量 Z=f*B/d（Step2）
  cl        = cluster       1cm 半径贪心聚类（Step3）
  tr        = track         GNN + 卡尔曼 + 匈牙利（Step4）
  metrics   评估 Cov/Prec/Chamfer（Step5）
  streaming 质量分数/子采样（Step6）
  ra        = run_all       一键流水线里的 pipeline()（Step6/7 复用实验口径）
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.abspath(os.path.join(HERE, "..", "reproduce"))

if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

# matplotlib 缓存一律写到本项目里，绝不落 C 盘用户目录
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcache"))

import config as C                      # noqa: E402
import gen_events as ge                 # noqa: E402
import filter as flt                    # noqa: E402
import match_triangulate as mt         # noqa: E402
import cluster as cl                    # noqa: E402
import track as tr                      # noqa: E402
import metrics                          # noqa: E402
import streaming                        # noqa: E402
import run_all as ra                    # noqa: E402  （只借它的 pipeline()，不触发 main()）

# 数据大本营：py_repro/data/<数据集名>/
DATA_DIR = os.path.join(HERE, "data")

# 每个数据集目录里必须有的四件套
STEP0_FILES = ("events_left.npz", "events_right.npz", "gt.npz", "meta.json")


def ensure_dir(path):
    """目录不存在就建，返回原路径（省得每处都写两行）。"""
    os.makedirs(path, exist_ok=True)
    return path


def load_events(dataset_dir, side):
    """读 Step0 产出的单侧事件流 -> (N,4) ndarray [t_us, x, y, pol]。"""
    import numpy as np
    path = os.path.join(dataset_dir, f"events_{side}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"[×] 找不到 {path}\n    请先跑 Step0：python step0_parse_dataset.py --dataset sim:1")
    return np.load(path)["events"]


def load_gt(dataset_dir):
    """读真值 (M,4)=[t_us, X, Y, Z]；真实数据集没有 OptiTrack 真值则返回 None。"""
    import numpy as np
    path = os.path.join(dataset_dir, "gt.npz")
    if not os.path.exists(path):
        return None
    arr = np.load(path)["gt"]
    return None if arr.size == 0 else arr


def load_meta(dataset_dir):
    """读数据集身份证 meta.json（丢了就返回空 dict，不阻塞流程）。"""
    import json
    path = os.path.join(dataset_dir, "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_path(kind="dir", title="请选择", default_dir=None):
    """弹窗选路径（PyCharm 直接点 Run 时的兜底交互）。
    kind: 'dir' 选文件夹 / 'file' 选文件
    返回 None 表示用户取消。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:                                    # 没有 tkinter 就别硬撑
        print(f"[!] 弹窗不可用({e})，请用命令行参数指定路径")
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if kind == "dir":
        path = filedialog.askdirectory(title=title, initialdir=default_dir or DATA_DIR)
    else:
        path = filedialog.askopenfilename(title=title, initialdir=default_dir or DATA_DIR)
    root.destroy()
    return path or None


def resolve_dir(arg, title="选择数据集文件夹"):
    """把命令行参数变成真实目录：给了就用，没给就弹窗。"""
    if arg:
        return os.path.abspath(arg)
    p = pick_path("dir", title)
    if p is None:
        raise SystemExit("[×] 没选文件夹，退出")
    return p


def banner(step, title, paper_ref):
    """统一步骤抬头，跑起来一眼知道这是第几步、对应论文哪节。"""
    print("=" * 72)
    print(f"  {step}  {title}")
    print(f"  论文依据：{paper_ref}")
    print("=" * 72)
