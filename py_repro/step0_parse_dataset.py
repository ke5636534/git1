# -*- coding: utf-8 -*-
r"""
step0_parse_dataset.py —— Step0：解析与翻译数据集（永远是第一步！）
==================================================================
★★★ 本步输入：仿真源 `sim:N`（N=粒子数） 或 真实数据文件夹 ★★★
★★★ 本步输出：data/<数据集名>/ 下四件套：events_left.npz / events_right.npz / gt.npz / meta.json ★★★

论文依据
--------
§4.2 + §5.2：场景里是**反射粒子盘**（14 cm 基线、双 DAVIS346 异步输出事件），
真值由 OptiTrack 给出。原始数据三种来源：
  ① 仿真 —— gen_events.py 按针孔投影 + Lissajous 轨迹造（本项目默认走这条，先跑通算法）
  ② 真实 —— ChopperWheel / GEM 数据集的 .txt / .npy（四列 t x y pol）
  ③ 任意中间格式 —— 你自己导出的 npz

本步只做「翻译」，不做任何算法：把千奇百怪的原始格式统一成
  events_left.npz   左相机 (N,4)=[t_us, x, y, pol]
  events_right.npz  右相机 (N,4)=[t_us, x, y, pol]
  gt.npz            真值   (M,4)=[t_us, X, Y, Z]（真实数据没有就存空数组）
  meta.json         身份证：来源/规模/相机参数/时间范围，后面每步都靠它自证

数据源四种写法（RATE 模式：真实公开数据集是第一公民）
  ① sim:N          仿真 —— gen_events 造（先跑通算法用）
  ② evt3:<.raw>    真实 —— Prophesee EVT3 raw（如 vonKarmanFlow 卡门涡街，
                    纯 numpy 解码见 evt3_decoder.py，无需商业 SDK）。
                    单目数据：右目存空数组；加 --pseudo-stereo D0 可造"伪双目"
                    （右目=左目水平平移 D0 px + 时间/纵向抖动 + 1% 丢失），
                    ★ 仅供管线联调，3D 坐标无物理意义，meta 里会大声标注。
  ③ 真实文件夹      左右两侧 events_left.* / events_right.*（+可选 gt.*）
  ④ 已翻译数据集    四件套齐全的目录直接跳过

★ 为什么必须 Step0 先做：
  和 Lattice 的 convert_logyarp2npz.py 一个道理——下游 6 步只认 npz，
  以后换数据集只改这一步，Step1~7 一行都不用动。

两种调用方式
------------
  1) 单独跑（PyCharm 右键 Run，不给参数就弹窗选输入）：
       python step0_parse_dataset.py
       python step0_parse_dataset.py --dataset sim:3
       python step0_parse_dataset.py --dataset "E:\\data\\chopperwheel" --name chopper
       python step0_parse_dataset.py --dataset "evt3:D:\\...\\karman_199ms.raw" ^
              --out data\\karman_real --max-dur-us 200000
       python step0_parse_dataset.py --dataset "evt3:D:\\...\\karman_199ms.raw" ^
              --out data\\karman_real_pseudo --max-dur-us 50000 ^
              --pseudo-stereo 42 --kappa 0.1
  2) 被启动器调用：
       from step0_parse_dataset import run_step
       run_step("sim:3", out_dir=...)
"""
import argparse
import json
import os
import time

import numpy as np

import _core as K


# ------------------------------------------------------------------
# 一、三种数据源的解析器
# ------------------------------------------------------------------

def _save(dataset_dir, left, right, gt, meta):
    """四件套落盘。meta 里的数字都从真实数组现算，不靠调用方传。"""
    K.ensure_dir(dataset_dir)
    np.savez_compressed(os.path.join(dataset_dir, "events_left.npz"), events=left)
    np.savez_compressed(os.path.join(dataset_dir, "events_right.npz"), events=right)
    np.savez_compressed(os.path.join(dataset_dir, "gt.npz"),
                        gt=gt if gt is not None else np.zeros((0, 4)))
    with open(os.path.join(dataset_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return dataset_dir


def parse_sim(n_particles=1, seed_offset=0, dataset_dir=None):
    """① 仿真源：调 gen_events 造 N 个粒子的事件流 + 真值。"""
    left, right, gt = K.ge.generate(n_particles, seed_offset)
    name = f"sim{n_particles}p" if dataset_dir is None else dataset_dir
    if dataset_dir is None:
        dataset_dir = os.path.join(K.DATA_DIR, name)
    meta = {
        "source": f"sim:{n_particles}",
        "source_desc": "gen_events 合成（针孔投影 + Lissajous 轨迹 + 噪声/热像素/burst）",
        "n_particles": n_particles,
        "seed_offset": seed_offset,
        "n_left": int(len(left)),
        "n_right": int(len(right)),
        "n_gt": int(len(gt)),
        "t_left_us": [float(left[:, 0].min()), float(left[:, 0].max())] if len(left) else None,
        "t_right_us": [float(right[:, 0].min()), float(right[:, 0].max())] if len(right) else None,
        "camera": {
            "baseline_m": K.C.BASELINE,
            "focal_px": K.C.F_PX,
            "img_w": K.C.IMG_W,
            "img_h": K.C.IMG_H,
        },
        "cols_events": ["t_us", "x_px", "y_px", "pol"],
        "cols_gt": ["t_us", "X_m", "Y_m", "Z_m"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save(dataset_dir, left, right, gt, meta)
    return dataset_dir, meta


def parse_evt3(raw_path, dataset_dir=None, max_dur_us=None, pseudo_stereo_px=None,
               kappa=1.0, jitter_us=50.0, jitter_px=1.0, right_drop=0.01, seed=42):
    """② 真实数据：Prophesee EVT3 .raw -> npz（单目；可选伪双目 + κ 事件子采样）。

    伪双目构造（仅供管线联调，meta 会标注）：
      右目事件 = 左目事件水平平移 pseudo_stereo_px（x<d0 的丢掉，视场无重叠），
      时间戳加 U(−jitter_us, +jitter_us) µs 抖动（模拟两相机独立时间戳），
      纵向加 ±jitter_px 抖动（模拟独立像素噪声），再随机丢 right_drop 比例（模拟漏检）。
      ⇒ 式(2) 三关卡全部有真实工作量，但 3D 坐标无物理意义。
    kappa < 1 时按论文 §6.4 的"逐事件逐相机均匀保留 κ"分别对左右独立子采样
      （计算量控制手段，meta 记录 kappa_applied）。
    """
    import evt3_decoder as e3

    X, Y, T, P, st = e3.decode_raw_file(raw_path, max_dur_us=max_dur_us)
    left = np.column_stack([T, X, Y, P]).astype(float)
    n_full = len(left)

    rng = np.random.default_rng(seed)
    if pseudo_stereo_px:
        d0 = float(pseudo_stereo_px)
        xr = X - int(round(d0))
        ok = xr >= 0                                   # 视场无重叠的丢掉
        tr = T[ok] + rng.uniform(-jitter_us, jitter_us, int(ok.sum()))
        yr = Y[ok] + rng.choice((-1.0, 0.0, 1.0), int(ok.sum()),
                                p=(0.15, 0.70, 0.15)) * jitter_px
        keep = rng.random(int(ok.sum())) >= right_drop
        right = np.column_stack([tr[keep], xr[ok][keep], yr[keep], P[ok][keep]]).astype(float)
    else:
        right = np.zeros((0, 4))
        d0 = None

    if kappa is not None and 0 < kappa < 1:
        kl = rng.random(len(left)) < kappa
        kr = rng.random(len(right)) < kappa if len(right) else np.zeros(0, bool)
        left, right = left[kl], right[kr]

    if dataset_dir is None:
        dataset_dir = os.path.join(K.DATA_DIR, "evt3_real")

    mono = not pseudo_stereo_px
    meta = {
        "source": f"evt3:{os.path.abspath(raw_path)}",
        "source_desc": "真实事件相机数据（Prophesee EVT3，纯 numpy 解码 evt3_decoder.py）",
        "dataset_origin": "STELLA 配套 Zenodo vonKarmanFlow（卡门涡街流场，真实 Gen4 事件，无 OptiTrack 真值）",
        "mono": mono,
        "pseudo_stereo": {
            "enabled": bool(pseudo_stereo_px),
            "shift_px": d0,
            "jitter_us": jitter_us if pseudo_stereo_px else None,
            "jitter_px": jitter_px if pseudo_stereo_px else None,
            "right_drop": right_drop if pseudo_stereo_px else None,
            "warning": "右目=左目平移拷贝+抖动，非物理双目；3D 坐标无物理意义，仅供管线联调" if pseudo_stereo_px else None,
        },
        "kappa_applied": float(kappa) if (kappa is not None and kappa < 1) else None,
        "n_events_decoded_full": int(n_full),
        "decode_stats": {k: st.get(k) for k in ("n_words", "n_addr_x", "n_vect12", "n_vect8")},
        "sensor": {"width": int(e3.WIDTH), "height": int(e3.HEIGHT), "format": "EVT3 (Prophesee Gen4)"},
        "n_left": int(len(left)),
        "n_right": int(len(right)),
        "n_gt": 0,
        "t_left_us": [float(left[:, 0].min()), float(left[:, 0].max())] if len(left) else None,
        "t_right_us": [float(right[:, 0].min()), float(right[:, 0].max())] if len(right) else None,
        "camera": {
            "baseline_m": K.C.BASELINE,
            "focal_px": K.C.F_PX,
            "img_w": K.C.IMG_W,
            "img_h": K.C.IMG_H,
            "note": "真实传感器 1280x720；config 的 346x260/焦距是 DAVIS346 仿真口径，"
                    "三角测量的 CX/CY 与真实分辨率不一致，伪双目下 X/Y 有系统性偏移（Z 不受影响）",
        },
        "cols_events": ["t_us", "x_px", "y_px", "pol"],
        "cols_gt": ["t_us", "X_m", "Y_m", "Z_m"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save(dataset_dir, left, right, None, meta)
    return dataset_dir, meta


def _read_event_file(path):
    """② 真实数据：读单侧事件文件（.txt/.csv 四列，或 .npy/.npz）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".npy",):
        arr = np.load(path)
    elif ext in (".npz",):
        z = np.load(path)
        arr = z[z.files[0]]
    else:
        arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None, ndmin=2)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise SystemExit(f"[×] {path} 形状 {arr.shape} 不是 (N,>=4)，读不了")
    return arr[:, :4]


def _find_side(folder, side):
    """在一个文件夹里按常见命名找某一侧的事件文件。"""
    names = [f"events_{side}", f"{side}_events", side, f"{side}_filtered", f"left" if side == "left" else "right"]
    exts = [".txt", ".csv", ".npy", ".npz"]
    for n in names:
        for e in exts:
            p = os.path.join(folder, n + e)
            if os.path.exists(p):
                return p
    return None


def parse_folder(folder, dataset_dir=None):
    """② 真实数据：文件夹里 left.* / right.*（命名见 _find_side），可选 gt.*。"""
    lp = _find_side(folder, "left")
    rp = _find_side(folder, "right")
    if lp is None or rp is None:
        raise SystemExit(f"[×] {folder} 里没找到左右两侧事件文件（需 events_left.* / events_right.*）")
    left, right = _read_event_file(lp), _read_event_file(rp)
    gp = None
    for n in ("gt", "ground_truth", "truth"):
        for e in (".txt", ".csv", ".npy", ".npz"):
            p = os.path.join(folder, n + e)
            if os.path.exists(p):
                gp = p
                break
        if gp:
            break
    gt = _read_event_file(gp) if gp else None
    if dataset_dir is None:
        dataset_dir = os.path.join(K.DATA_DIR, os.path.basename(os.path.abspath(folder)))
    meta = {
        "source": os.path.abspath(folder),
        "source_desc": f"真实数据文件夹（左={os.path.basename(lp)} 右={os.path.basename(rp)} 真值={os.path.basename(gp) if gp else '无'}）",
        "n_particles": None,
        "n_left": int(len(left)),
        "n_right": int(len(right)),
        "n_gt": int(len(gt)) if gt is not None else 0,
        "t_left_us": [float(left[:, 0].min()), float(left[:, 0].max())] if len(left) else None,
        "t_right_us": [float(right[:, 0].min()), float(right[:, 0].max())] if len(right) else None,
        "camera": {
            "baseline_m": K.C.BASELINE,
            "focal_px": K.C.F_PX,
            "img_w": K.C.IMG_W,
            "img_h": K.C.IMG_H,
        },
        "cols_events": ["t_us", "x_px", "y_px", "pol"],
        "cols_gt": ["t_us", "X_m", "Y_m", "Z_m"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save(dataset_dir, left, right, gt, meta)
    return dataset_dir, meta


# ------------------------------------------------------------------
# 二、统一入口
# ------------------------------------------------------------------

def run_step(src="sim:1", out_dir=None, name=None, seed_offset=0, **evt3_opts):
    """src 四种写法：'sim:3' / 'evt3:<raw>' / 真实数据文件夹 / 已存在的 dataset 目录（直接返回）。
    evt3: 源可带 evt3_opts：max_dur_us / pseudo_stereo_px / kappa / jitter_us / jitter_px。
    """
    K.banner("Step0", "解析与翻译数据集", "§4.2 双 DAVIS346 采集 + §5.2 数据集与真值")

    if isinstance(src, str) and src.startswith("sim:"):
        n = int(src.split(":")[1])
        dataset_dir, meta = parse_sim(n, seed_offset, out_dir)
    elif isinstance(src, str) and src.startswith("evt3:"):
        raw = src[len("evt3:"):].strip('"')
        if not os.path.isfile(raw):
            raise SystemExit(f"[×] 找不到 EVT3 raw 文件：{raw}")
        dataset_dir, meta = parse_evt3(raw, out_dir, seed=42 + seed_offset, **evt3_opts)
    elif out_dir and os.path.isdir(out_dir) and \
            all(os.path.exists(os.path.join(out_dir, f)) for f in K.STEP0_FILES):
        print(f"[=] {out_dir} 已经是翻译好的数据集，跳过")
        return out_dir
    elif isinstance(src, str) and os.path.isdir(src):
        dataset_dir, meta = parse_folder(src, out_dir)
    else:
        raise SystemExit(f"[×] 看不懂的数据源：{src}（要 'sim:N' / 'evt3:<raw>' 或文件夹路径）")

    print(f"[gen ] 左相机事件 : {meta['n_left']:>7} 个")
    print(f"[gen ] 右相机事件 : {meta['n_right']:>7} 个")
    print(f"[gen ] 真值点     : {meta['n_gt']:>7} 个")
    if meta.get("mono"):
        print("[warn] 单目真实数据：右目为空，Step2 起无匹配可做（Step1 去噪可正常跑）")
    ps = meta.get("pseudo_stereo") or {}
    if ps.get("enabled"):
        print(f"[warn] 伪双目模式：右目=左目平移 {ps['shift_px']:.0f}px+抖动，"
              f"3D 坐标无物理意义，仅供管线联调")
    if meta.get("kappa_applied"):
        print(f"[warn] 已按 κ={meta['kappa_applied']} 做逐事件均匀子采样（计算量控制）")
    if meta["t_left_us"]:
        print(f"[gen ] 时间范围   : 左 {meta['t_left_us'][0]:.1f} ~ {meta['t_left_us'][1]:.1f} µs")
    print(f"[out ] 四件套落盘 : {dataset_dir}")
    print("        events_left.npz / events_right.npz / gt.npz / meta.json")
    return dataset_dir


def main():
    ap = argparse.ArgumentParser(description="Step0 解析与翻译数据集")
    ap.add_argument("--dataset", default=None,
                    help="sim:N / evt3:<raw> / 真实数据文件夹；不给则弹窗")
    ap.add_argument("--out", default=None, help="输出数据集目录（默认 py_repro/data/<名字>）")
    ap.add_argument("--name", default=None, help="数据集别名（默认 sim<N>p 或源文件夹名）")
    ap.add_argument("--seed-offset", type=int, default=0)
    # ---- evt3: 源专用 ----
    ap.add_argument("--max-dur-us", type=int, default=None,
                    help="[evt3] 只保留前 N µs（如 200000 = 前 0.2 s）")
    ap.add_argument("--pseudo-stereo", type=float, default=None, metavar="D0PX",
                    help="[evt3] 造伪双目：右目=左目水平平移 D0 px（如 42）")
    ap.add_argument("--kappa", type=float, default=1.0,
                    help="[evt3] 逐事件均匀子采样比例（论文 §6.4 口径，1.0=不抽）")
    ap.add_argument("--jitter-us", type=float, default=50.0, help="[evt3] 伪双目时间抖动 µs")
    ap.add_argument("--jitter-px", type=float, default=1.0, help="[evt3] 伪双目纵向抖动 px")
    args = ap.parse_args()

    src = args.dataset
    if src is None:
        p = K.pick_path("dir", "选择原始数据文件夹（或直接取消，改用仿真）")
        src = p if p else "sim:1"
        print(f"[i] 输入源：{src}")
    out = args.out
    if out is None and args.name:
        out = os.path.join(K.DATA_DIR, args.name)

    run_step(src, out_dir=out, seed_offset=args.seed_offset,
             max_dur_us=args.max_dur_us, pseudo_stereo_px=args.pseudo_stereo,
             kappa=args.kappa, jitter_us=args.jitter_us, jitter_px=args.jitter_px)


if __name__ == "__main__":
    main()
