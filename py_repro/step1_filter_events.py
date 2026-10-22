# -*- coding: utf-8 -*-
r"""
step1_filter_events.py —— Step1：事件去噪（热像素 + 不应期）
==================================================================
★★★ 本步输入：data/<名>/events_left.npz + events_right.npz（Step0 产物）★★★
★★★ 本步输出：events_left_f.npz / events_right_f.npz ★★★

论文依据
--------
§4 DVS 噪声滤波（原文引用 [19] 的滤波做法）：
  ① 热像素删除 —— 某个像素长期疯狂触发，它不是粒子，是坏点。
     判据：该像素事件数占全帧比例 > HOT_RATIO(=0.02) 就整条枪毙。
  ② 不应期 (refractory) —— 同一像素两次触发间隔 < REFRACT_US(=1000 µs)，
     后一次是电路噪声，丢掉。（真实 DVS 的物理特性，这两个常量在 filter.py 顶部）

★ 关键理解：这一步**不做空间滤波**。粒子还是粒子，噪声还是噪声，
  只是把"根本不可能是信号"的事件先清掉，给 Step2 的匹配减负。
  所以事件数只会小幅下降（实测 7490 -> 7475）。

两种调用方式
------------
  1) python step1_filter_events.py --in "py_repro\data\sim1p"
  2) from step1_filter_events import run_step;  run_step(dataset_dir)
"""
import argparse
import os

import numpy as np

import _core as K


def run_step(dataset_dir, verbose=True):
    K.banner("Step1", "事件去噪（热像素 + 不应期）", "§4 DVS 噪声滤波 [19]")

    left = K.load_events(dataset_dir, "left")
    right = K.load_events(dataset_dir, "right")

    lf = K.flt.filter_events(left)
    rf = K.flt.filter_events(right)

    np.savez_compressed(os.path.join(dataset_dir, "events_left_f.npz"), events=lf)
    np.savez_compressed(os.path.join(dataset_dir, "events_right_f.npz"), events=rf)

    if verbose:
        print(f"[filter] 参数：热像素比例阈值 {K.flt.HOT_RATIO:.2f}，不应期 {K.flt.REFRACT_US:.0f} µs")
        print(f"[filter] 左相机 {len(left):>6} -> {len(lf):>6}   （删 {len(left) - len(lf)}）")
        print(f"[filter] 右相机 {len(right):>6} -> {len(rf):>6}   （删 {len(right) - len(rf)}）")
        print(f"[out   ] 已保存：events_left_f.npz / events_right_f.npz")
        print(f"         目录：{dataset_dir}")
    return lf, rf


def main():
    ap = argparse.ArgumentParser(description="Step1 事件去噪")
    ap.add_argument("--in", dest="inp", default=None, help="数据集目录（Step0 产物）")
    args = ap.parse_args()
    d = K.resolve_dir(args.inp, "选择 Step0 生成的数据集文件夹")
    run_step(d)


if __name__ == "__main__":
    main()
