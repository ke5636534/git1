# -*- coding: utf-8 -*-
r"""
test_all_steps.py —— 分步复现的自动化测试（回归测试 / 冒烟测试）
==================================================================
干嘛的
------
把《复现进度台账.md》里那些"预期值"变成**可自动判定的断言**。
跑一遍，直接打 PASS/FAIL 表格。改完代码后跑它，就知道有没有改坏。

★ 为什么这些数字能当判据（不是拍脑袋）：
  config.py 里 SEED=42 固定了随机种子 → 同输入必然同输出。
  所以每一步的产物数量、指标数值都是**确定值**，可以写死。

测试用例清单
------------
  TC0  环境：numpy / scipy / matplotlib 可导入（scipy 是 Step4 匈牙利算法用的）
  TC1  Step0 数据翻译：左 9494 / 右 9487 / 真值 1000
  TC2  Step1 去噪：9494→9010、9487→8998
  TC3  Step2 匹配：2334 个 3D 点，Z∈[0.280, 8.400] m，中位数 0.689（互斥后）
  TC4  Step3 聚类：957 帧 → 533 簇
  TC5  Step4 跟踪：999 帧（490 帧有检测）→ 110 条轨迹 / 533 点
  TC6  Step5 评估：Cov 1.0000 / Prec 1.0000 / Chamfer 3.7501578973164786 mm
  TC7  断点续跑：只重跑 Step3+Step5，指标必须**逐位相同**
  TC8  Step7 主实验：5 组粒子数全部 Cov>0.95 且不崩（Fig.6 复刻）
  TC9  异常用例：数据集没有真值 → Step5 必须优雅跳过、不崩
  TC10 边界用例：匹配结果为空 → Step3 返回空数组、不崩
  TC11 流式延迟：窗口化匹配累计必须 = 全局一次匹配（坑#8 回归）
  TC12 Fig.11 锚点：κ=1.0 @10P 与 step8 逐位一致 + L_proc 中位随 κ 单调不增且降 ≥3×
       ★ 隐含回归：κ 必须作用在【原始事件】上（filter 之后、match 之前）。
         一旦把抽样挪到 match 之后，L_proc 就不会随 κ 下降，这条立刻 FAIL。
  TC13 Fig.14 拥塞闭环：三档 ν（论文原值 0.1/0.25/1.0）零违例 + 三分位 严<全 + 保留率单调
       + 互相关滞后 ≤2 格；r 对非迟滞算法（CUBIC/BBR）≥0.5，COPA 只报告
       （★ 2026-10-21 随论文 s 口径调整：旧 q 反标定口径人为制造 25%/55%/100% 保留率区分，
         r≥0.6 是那个口径下的自我标准；论文 s 口径下好数据 ν=0.25≈不滤，中/全档天然贴近，
         区分度主要来自严档 ν=0.1。COPA 跟随慢是论文承认的现象——"延迟型不如丢包型灵敏"。）

运行方式
--------
  cd py_repro
  D:\Users\admuj\anaconda3\envs\opencv_env\python.exe test_all_steps.py

退出码：0 = 全部通过；1 = 有 FAIL（可以挂到 CI / 提交前跑）。
测试数据写在 data\_selftest\ 里，**不会碰你自己的 data\sim1p**。
"""
import contextlib
import io
import json
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _core as K                            # noqa: E402
import step0_parse_dataset as s0             # noqa: E402
import step1_filter_events as s1             # noqa: E402
import step2_match_triangulate as s2         # noqa: E402
import step3_cluster as s3                   # noqa: E402
import step4_track as s4                     # noqa: E402
import step5_metrics as s5                   # noqa: E402
import step7_experiments as s7               # noqa: E402
import step8_latency as s8                   # noqa: E402
import step9_latency_kappa as s9            # noqa: E402
import step10_congestion as s10             # noqa: E402

# ------------------------------------------------------------------
# 判据（改代码后如果这些"合理"了，才允许改这里的数字）
# ------------------------------------------------------------------
# ★ 2026-10-21 锚点全面更新（P-A 算法修正：互斥匹配 + 滑窗跟踪 + 每步多事件）：
#   互斥销坑#13 后假重复点消失；N_PER_STEP=1->3（P-D13，配套互斥，单粒子多像素事件）；
#   跟踪改论文滑窗 (1,3)/(5,12)。数字为 SEED=42 下重测的确定值。
EXPECT = {
    "n_left": 9494,
    "n_right": 9487,
    "n_gt": 1000,
    "n_left_f": 9010,
    "n_right_f": 8998,
    "n_match": 2334,
    "z_min": 0.280,
    "z_max": 8.400,
    "z_median": 0.689,
    "n_frames": 957,
    "n_centers": 533,
    "n_track_frames": 999,
    "n_det_frames": 490,
    "n_tracks": 110,
    "n_track_pts": 533,
    "cov": 1.0,
    "prec": 1.0,
    "chamfer": 0.0037501578973164786,
}

RESULTS = []          # [(编号, 用例名, 是否通过, 说明)]


def check(cid, name, ok, detail=""):
    RESULTS.append((cid, name, bool(ok), detail))
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {cid:<5} {name}" + (f"   -> {detail}" if detail else ""))
    return bool(ok)


def near(got, want, tol):
    return abs(float(got) - float(want)) <= tol


@contextlib.contextmanager
def quiet():
    """把被测函数的一大堆打印吞掉，失败时再把日志吐出来。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def clean_dir(path):
    """只清测试目录里的文件，别的目录一律不碰。"""
    K.ensure_dir(path)
    for f in os.listdir(path):
        p = os.path.join(path, f)
        if os.path.isfile(p):
            os.remove(p)
    return path


# ==================================================================
#  各个测试用例
# ==================================================================

def tc0_env():
    print("\n【TC0】环境检查")
    ok = True
    for mod in ("numpy", "scipy", "matplotlib"):
        try:
            __import__(mod)
            check(f"TC0-{mod}", f"import {mod}", True)
        except Exception as e:
            ok = False
            check(f"TC0-{mod}", f"import {mod}", False, str(e))
    return ok


def run_chain_into(dataset):
    """跑 Step0~Step5 到指定目录，返回每步的关键中间量。"""
    with quiet() as log1:
        s0.run_step("sim:1", out_dir=dataset)
    meta = K.load_meta(dataset)
    left = K.load_events(dataset, "left")
    right = K.load_events(dataset, "right")
    gt = K.load_gt(dataset)

    with quiet() as log2:
        lf, rf = s1.run_step(dataset)
    with quiet() as log3:
        matched = s2.run_step(dataset, make_fig=False)
    with quiet() as log4:
        centers = s3.run_step(dataset, make_fig=False)
    with quiet() as log5:
        pts, ids, n_trk = s4.run_step(dataset, make_fig=False)
    with quiet() as log6:
        m = s5.run_step(dataset)

    logs = "\n".join(b.getvalue() for b in (log1, log2, log3, log4, log5, log6))
    return {
        "meta": meta, "left": left, "right": right, "gt": gt,
        "lf": lf, "rf": rf, "matched": matched, "centers": centers,
        "pts": pts, "ids": ids, "n_trk": n_trk, "metrics": m, "log": logs,
    }


def tc1_to_tc6(dataset):
    print("\n【TC1~TC6】数据翻译 → 去噪 → 匹配 → 聚类 → 跟踪 → 评估")
    try:
        r = run_chain_into(dataset)
    except Exception:
        traceback.print_exc()
        check("TC1-6", "全链路跑通", False, "抛异常，见上面 traceback")
        return None

    # ---- TC1 Step0 ----
    check("TC1-a", "左相机事件数 = 9494", len(r["left"]) == EXPECT["n_left"],
          f"实测 {len(r['left'])}")
    check("TC1-b", "右相机事件数 = 9487", len(r["right"]) == EXPECT["n_right"],
          f"实测 {len(r['right'])}")
    check("TC1-c", "真值点数 = 1000", r["gt"] is not None and len(r["gt"]) == EXPECT["n_gt"],
          f"实测 {0 if r['gt'] is None else len(r['gt'])}")
    check("TC1-d", "meta.json 四件套齐全",
          all(os.path.exists(os.path.join(dataset, f)) for f in K.STEP0_FILES))

    # ---- TC2 Step1 ----
    check("TC2-a", "左相机去噪后 = 9010", len(r["lf"]) == EXPECT["n_left_f"],
          f"实测 {len(r['lf'])}（删 {len(r['left']) - len(r['lf'])}）")
    check("TC2-b", "右相机去噪后 = 8998", len(r["rf"]) == EXPECT["n_right_f"],
          f"实测 {len(r['rf'])}（删 {len(r['right']) - len(r['rf'])}）")

    # ---- TC3 Step2 ----
    mp = r["matched"]
    check("TC3-a", "匹配 3D 点数 = 2334", len(mp) == EXPECT["n_match"], f"实测 {len(mp)}")
    if len(mp):
        z = mp[:, 3]
        check("TC3-b", "Z 最小值 = 0.280 m", near(z.min(), EXPECT["z_min"], 5e-3),
              f"实测 {z.min():.3f}")
        check("TC3-c", "Z 最大值 = 8.400 m", near(z.max(), EXPECT["z_max"], 5e-3),
              f"实测 {z.max():.3f}")
        check("TC3-d", "Z 中位数 = 0.689 m", near(np.median(z), EXPECT["z_median"], 5e-3),
              f"实测 {np.median(z):.3f}")

    # ---- TC4 Step3 ----
    c = r["centers"]
    n_frames = len(np.unique((mp[:, 0] // (K.C.DT_S * 1e6)).astype(int))) if len(mp) else 0
    check("TC4-a", "分帧数 = 957", n_frames == EXPECT["n_frames"], f"实测 {n_frames}")
    check("TC4-b", "聚类簇心 = 533", len(c) == EXPECT["n_centers"], f"实测 {len(c)}")

    # ---- TC5 Step4 ----
    frames = K.tr.frames_from_centers(c)
    n_det = sum(1 for f in frames if len(f))
    check("TC5-a", "跟踪帧数 = 999", len(frames) == EXPECT["n_track_frames"], f"实测 {len(frames)}")
    check("TC5-b", "有检测帧数 = 490", n_det == EXPECT["n_det_frames"], f"实测 {n_det}")
    check("TC5-c", "轨迹条数 = 110", r["n_trk"] == EXPECT["n_tracks"], f"实测 {r['n_trk']}")
    check("TC5-d", "轨迹点数 = 533", len(r["pts"]) == EXPECT["n_track_pts"], f"实测 {len(r['pts'])}")

    # ---- TC6 Step5 ----
    m = r["metrics"]
    check("TC6-a", "Cov = 1.0000", near(m["coverage"], EXPECT["cov"], 1e-6), f"实测 {m['coverage']:.6f}")
    check("TC6-b", "Prec = 1.0000", near(m["precision"], EXPECT["prec"], 1e-6), f"实测 {m['precision']:.6f}")
    check("TC6-c", "Chamfer = 3.7502 mm", near(m["chamfer"], EXPECT["chamfer"], 1e-9),
          f"实测 {m['chamfer'] * 1000:.6f} mm")
    check("TC6-d", "重建点 533 / 真值 1000",
          m["n_recon"] == 533 and m["n_gt"] == 1000,
          f"{m['n_recon']}/{m['n_gt']}")
    return r


def tc7_resume(dataset, first_metrics):
    print("\n【TC7】断点续跑一致性（只重跑 Step3+Step5）")
    before = dict(first_metrics)
    with quiet():
        s3.run_step(dataset, make_fig=False)
        again = s5.run_step(dataset)
        # Step3 再跑一次，确认 Step2 产物没被动过
    same = all(again[k] == before[k] for k in ("coverage", "precision", "chamfer",
                                               "n_recon", "n_gt"))
    check("TC7", "重跑结果与首次逐位相同", same,
          "Cov/Prec/Chamfer 完全相等" if same else f"{before} vs {again}")

    with open(os.path.join(dataset, "metrics.json"), "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    check("TC7-b", "metrics.json 与内存值一致",
          on_disk["coverage"] == again["coverage"] and
          on_disk["chamfer"] == again["chamfer"])


def tc8_experiment():
    print("\n【TC8】主实验 Step7（粒子数扫描，Fig.6 复刻）")
    out = os.path.join(HERE, "experiments")
    try:
        with quiet():
            rows = s7.run_step(out)
    except Exception:
        traceback.print_exc()
        check("TC8", "Step7 跑通", False)
        return
    counts = [r[0] for r in rows]
    check("TC8-a", "5 组粒子数 = [1,2,3,5,10]", counts == list(K.C.PARTICLE_COUNTS), f"{counts}")
    check("TC8-b", "每组 Cov 都 > 0.95", all(r[1] > 0.95 for r in rows),
          "min Cov = %.4f" % min(r[1] for r in rows))
    check("TC8-c", "每组 Chamfer 都 < 20 mm", all(r[3] < 0.020 for r in rows),
          "max Chamfer = %.2f mm" % (max(r[3] for r in rows) * 1000))
    check("TC8-d", "图已落盘 fig6_particle_count.png",
          os.path.exists(os.path.join(out, "fig6_particle_count.png")))
    # 精度不该随粒子数明显掉（论文核心结论）
    drop = max(r[1] for r in rows) - min(r[1] for r in rows)
    check("TC8-e", "Cov 在 1P→10P 间波动 < 0.02（论文结论）", drop < 0.02,
          f"波动 {drop:.4f}")


def tc9_no_gt(base_dataset):
    print("\n【TC9】异常用例：数据集没有真值 → Step5 必须优雅跳过")
    d = clean_dir(os.path.join(K.DATA_DIR, "_selftest_nogt"))
    src = os.path.join(base_dataset, "cluster_centers.npz")
    if os.path.exists(src):
        with open(src, "rb") as fi, open(os.path.join(d, "cluster_centers.npz"), "wb") as fo:
            fo.write(fi.read())
    np.savez_compressed(os.path.join(d, "gt.npz"), gt=np.zeros((0, 4)))
    try:
        with quiet():
            m = s5.run_step(d)
        ok = m["coverage"] is None and m["chamfer"] is None
        check("TC9", "无真值时返回 None 且不崩", ok, f"返回 {m}")
    except Exception as e:
        check("TC9", "无真值时返回 None 且不崩", False, f"{type(e).__name__}: {e}")


def tc10_empty_match():
    print("\n【TC10】边界用例：匹配结果为空 → Step3 返回空数组、不崩")
    d = clean_dir(os.path.join(K.DATA_DIR, "_selftest_empty"))
    np.savez_compressed(os.path.join(d, "match_points.npz"), pts=np.zeros((0, 4)))
    try:
        with quiet():
            c = s3.run_step(d, make_fig=False)
        check("TC10", "空匹配 → 0 簇且不崩", len(c) == 0, f"返回形状 {np.shape(c)}")
    except Exception as e:
        check("TC10", "空匹配 → 0 簇且不崩", False, f"{type(e).__name__}: {e}")


def tc11_latency_streaming():
    """TC11：流式延迟的自检 —— 逐窗口匹配累计必须等于全局一次匹配。

    这是坑#8 的回归测试：窗口切分一旦丢掉回看缓冲，匹配数会凭空掉 30%~40%。
    """
    print("\n【TC11】流式延迟：窗口化匹配不得丢配对（坑#8 回归）")
    try:
        with quiet():
            r = s8.measure(1)
    except Exception as e:
        check("TC11", "Step8 1P 测量跑通", False, f"{type(e).__name__}: {e}")
        return
    got, want = r["n_matched"], r["n_matched_global"]
    check("TC11-a", "流式累计匹配 = 全局一次匹配（差 0 对）", got == want,
          f"流式 {got} / 全局 {want}")
    med = float(np.median(r["l_proc"])) if len(r["l_proc"]) else -1.0
    check("TC11-b", "L_proc 中位数落在 (0, 10) ms 合理区间", 0 < med < 10,
          f"实测 {med:.3f} ms")
    check("TC11-c", "有输出点样本（延迟统计非空）", len(r["l_proc"]) > 0,
          f"样本数 {len(r['l_proc'])}")


def tc12_fig11_kappa():
    """TC12：Fig.11 锚点 + L_proc 随 κ 下降（只有 κ 作用在原始事件上才成立）。"""
    print("\n【TC12】Fig.11 锚点一致 + L_proc 中位随 κ 单调不增且降 ≥3×")
    try:
        with quiet():
            rows = s9.sweep_kappa()
    except Exception as e:
        traceback.print_exc()
        check("TC12", "κ 扫描跑通", False, f"{type(e).__name__}: {e}")
        return

    a = rows[0]
    ok_matched = a["n_matched"] == s9.ANCHOR_MATCHED
    ok_points = a["n_points"] == s9.ANCHOR_POINTS
    # ★ L_proc 的绝对值**不作为硬失败条件**，只在结论里报告。
    #   理由：它是墙钟测量，同一份代码、同一份数据在不同机器负载下实测
    #   2.06 / 2.07 / 2.36 / 2.82 ms（锚点 1.817 ms 是当初空载单跑记录下来的），
    #   把它卡死只会把 "CPU 抖了一下" 误判成功能回退。
    #   真正能判死回归的是三条可复现的量：计数逐位一致 + 单调不增 + 降 ≥3 倍。
    ok_lat = 0 < a["p50"] < 10.0          # 只做量级 sanity
    p50 = [r["p50"] for r in rows]
    # 相邻 κ 允许 ≤20% 上跳：墙钟噪声（坑#11）会让紧邻两点偶尔反超，
    # 但"κ 越小 L_proc 越低"的整体趋势 + 端点降 ≥3× 必须成立（论文 Fig.11 下降方向）。
    MONO_TOL = 0.20
    mono = all(p50[i + 1] <= p50[i] * (1 + MONO_TOL) for i in range(len(p50) - 1))
    drop = (p50[0] / p50[-1]) if p50[-1] else float("inf")

    # P14 / 坑#11：重复测量的稳健性验收。
    #   · 拼接中位（r["p50"]）已是 N_REPEAT 趟合并样本，方差压约 √N；
    #   · 硬判据：每趟单独中位与本 κ 的拼接中位偏差 ≤ max(25% 相对, 0.35ms 绝对)。
    #     加绝对地板是因为当 L_proc 本身只有零点几 ms（低 κ 场景）时，墙钟的
    #     绝对抖动（几百 µs）会让"相对偏差"失去意义——这正是坑#11 的本质；
    #     相对闸门只在 L_proc 足够大（如锚点 κ=1.0）时真正生效。
    #   · 信息性：各 κ 的 5 趟 ptp 波动（只报告，不卡死——墙钟本身就在抖）。
    ABS_FLOOR_MS = 0.35
    rep_meds = [r["rep_medians"] for r in rows]
    worst_abs = 0.0
    worst_rel = 0.0
    for r, rm in zip(rows, rep_meds):
        if not rm or r["p50"] <= 0:
            continue
        for m in rm:
            d = abs(m - r["p50"])
            worst_abs = max(worst_abs, d)
            worst_rel = max(worst_rel, d / r["p50"])
    spreads = [np.ptp(rm) / np.median(rm) for rm in rep_meds
               if len(rm) >= 2 and np.median(rm) > 0]
    ok_fluct = worst_abs <= max(0.25 * a["p50"], ABS_FLOOR_MS)
    ok = ok_matched and ok_points and ok_lat and mono and drop >= 3.0 and ok_fluct

    check("TC12", "锚点 11734/519 逐位一致 + L_proc 趋势下降(≤20%容差)且降 ≥3× + 重复测量波动≤阈值", ok,
          f"κ=1.0: 匹配{a['n_matched']}/重建{a['n_points']}/L_proc中位{a['p50']:.3f}ms"
          f"（锚点 {s9.ANCHOR_MATCHED}/{s9.ANCHOR_POINTS}/"
          f"{s9.ANCHOR_L_PROC_MEDIAN_MS}，墙钟值受负载影响不定，仅报告）"
          f"；中位序列 {'→'.join(f'{v:.3f}' for v in p50)} ms；κ1.0→0.1 降 ×{drop:.1f}（相邻≤20%容差单调）；"
          f"5趟最大绝对偏离 {worst_abs*1000:.0f}µs（阀 max(25%相对,0.35ms)）；"
          f"最大相对偏离 {worst_rel*100:.0f}%；ptp波动 "
          + ("/".join(f"{s*100:.0f}%" for s in spreads) if spreads else "n/a"))


def tc13_congestion_loop():
    """TC13：Fig.14 拥塞闭环（pacing → ν → 点/秒），四条验收 + 红线回归。"""
    print("\n【TC13】Fig.14 拥塞闭环：pacing 低 → ν 严 → 点/秒低")
    out_dir = os.path.join(HERE, "experiments")
    # 红线回归（2026-10-21 更新）：step10 已改论文 s 口径，不再调用 quality_scores；
    # 新红线 = 三档 ν 必须是论文原值 0.1/0.25/1.0。
    try:
        with quiet():
            out = s10.run_step(out_dir=out_dir, verbose=False)
    except Exception as e:
        traceback.print_exc()
        check("TC13", "拥塞闭环跑通", False, f"{type(e).__name__}: {e}")
        return

    a, rs = out["accept"], out["results"]
    worst_r = min(float(rs[n]["r"]) for n in rs)
    worst_lag = max(abs(int(rs[n]["lag"])) for n in rs)
    red_ok = sorted(s10.NU_LEVELS) == [0.1, 0.25, 1.0]     # 红线：三档 ν = 论文原值
    # r 门槛只对非迟滞算法卡（COPA 迟滞是论文承认的现象，只报告）
    ok_r = all(rs[n]["r"] >= 0.5 for n in rs if n != "COPA")
    ok = bool(a["all_ok"]) and red_ok and ok_r

    detail = "；".join(
        f"{n}: r={rs[n]['r']:.3f} 三分位{'<'.join(f'{v:.0f}' for v in rs[n]['tercile'])} "
        f"违例{rs[n]['violations']} lag={rs[n]['lag']}格" for n in s10.ALGO_ORDER)
    detail += (f"；ν 取值 {a['n_nu_values']} 个（论文原值 {tuple(s10.NU_LEVELS)}）；"
               f"最小 r {worst_r:.3f}；最大滞后 {worst_lag} 格")
    check("TC13", "三算法 × 四条验收 + 红线（三档 ν = 论文原值 0.1/0.25/1.0）", ok, detail)


# ==================================================================
#  主流程
# ==================================================================
def main():
    print("=" * 72)
    print("  粒子追踪复现 · 自动化测试")
    print("  解释器：", sys.executable)
    print("=" * 72)

    dataset = clean_dir(os.path.join(K.DATA_DIR, "_selftest"))

    if not tc0_env():
        report()
        return 1

    r = tc1_to_tc6(dataset)
    if r is not None:
        tc7_resume(dataset, r["metrics"])
    tc8_experiment()
    tc9_no_gt(dataset)
    tc10_empty_match()
    tc11_latency_streaming()
    tc12_fig11_kappa()
    tc13_congestion_loop()

    return report()


def report():
    print("\n" + "=" * 72)
    total = len(RESULTS)
    passed = sum(1 for x in RESULTS if x[2])
    failed = [x for x in RESULTS if not x[2]]
    print(f"  结果：{passed}/{total} 通过" + (f"，{len(failed)} 项 FAIL" if failed else "，全部 PASS ✅"))
    if failed:
        print("\n  失败清单：")
        for cid, name, ok, detail in failed:
            print(f"    - {cid}  {name}   {detail}")
    print("=" * 72)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
