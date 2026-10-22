# -*- coding: utf-8 -*-
r"""
step10_congestion.py —— Step10：拥塞闭环实验（论文 §6.4 Figure 14 复刻）
========================================================================
★★★ 本步输入：无文件依赖，内部跑 10P 场景的流式流水线 ★★★
★★★ 本步输出：experiments/fig14_congestion.png + congestion.csv ★★★

论文依据（§6.4，page 9，映射关系照抄）
--------------------------------------
  "when the pacing rate is less than 2 Mbps or 4 Mbps the filter threshold ν in the
   triangulation application is set more restrictively to 0.1 or 0.25, respectively"
  ⇒ 控制器规则（pacing → ν 松紧）：
      pacing <  2 Mbps  → ν 最严  （论文 s ≤ 0.1）  → 本实现目标保留率 0.25
      pacing <  4 Mbps  → ν 中    （论文 s ≤ 0.25） → 本实现目标保留率 0.55
      pacing >= 4 Mbps  → transmit all（论文最松）  → 本实现目标保留率 1.00

  Fig.14 画的是：三种拥塞控制算法（CUBIC / BBR / COPA）下，ν 与该算法产出的
  "triangulated event points per second" 随时间的曲线；结论是点/秒大致跟随 pacing rate，
  连探测阶段也跟随。

★★★ 差异 #12（必须写清楚，不许含糊）
--------------------------------------
  本步**不接真实 QUIC**：环境里没装 aioquic，也没有可控瓶颈链路，
  无法产生真实的 pacing rate / RTT / 丢包序列。
  因此用**合成 pacing 轨迹**代替三种拥塞算法的输出，验证的是
  **「pacing → ν 控制器 → 逐窗过滤 → 点/秒」这条闭环逻辑**本身，
  不是 CUBIC/BBR/COPA 的实现正确性，也不是 QUIC 协议本身。
  换言之：这里证明的是"给定一条 pacing 曲线，闭环输出的点/秒会跟随它"；
  "真机上 pacing 会长这样"这件事本步**没有证明**。

★ 控制器口径（2026-10-21 修正，审查发现A，销坑#5）
--------------------------------------------------
  论文的 s 是式(2) 匹配分数：越小越好，保留 s ≤ ν。
  match_and_triangulate 现在直接返回 s 列，所以 0.1 / 0.25 / transmit all
  三档**直接用论文原值**，不再需要"局部密度代理 + 按保留率反标定"的换算
  （旧版的 _calibrate_nu 已整体删除）。

★ 性能说明
--------------------------------------
  s 随匹配一次算出（O(1) 每点），逐窗口只做 s ≤ ν 比较。
  旧版 O(n²) 的 quality_scores 密度代理已从本步移除。

★ 合成 pacing 轨迹的形状，以及为了跑满三档 ν 而对论文数字做的偏离
------------------------------------------------------------------
  三者都用 np.random.default_rng(C.SEED) 固定，10 s / 10 ms 一格 ≈ 1000 格。

  · CUBIC（丢包型锯齿）：1 Mbps 慢起 → 约 1.5 s 涨到 12 Mbps → 丢包掉到 60% → 再涨，
    共 6 次丢包。
    ★偏离：真实 CUBIC 每轮会恢复到同一个 W_max（于是长期停在 ≥ 4 Mbps，
      ν 三档走不全）；这里让锯齿**峰值逐轮衰减**（×0.65），等价于链路越来越挤，
      使 10 s 内扫过 12 → 1.3 Mbps，三条带宽档位都被走到。

  · BBR（带宽探测型平台）：稳在 8 Mbps；每 2 s 探测 +25% 持续 200 ms；
    每 5 s 排空 1 s。
    ★偏离：论文说排空"回落到 4 Mbps"，这里排空最深到 1.6 Mbps。
      原因同上：4 Mbps 的下限会让 "<2 Mbps ⇒ ν 最严" 这一档永远触发不到。

  · COPA（延迟型迟滞）：均值回复的随机游走 + 重度低通（≈0.8 s 窗），反应明显偏慢。
    ★偏离：论文给的范围是 4~9 Mbps，这里允许越过下限到 ~1.5 Mbps，理由同上。

两种调用方式
------------
  1) python step10_congestion.py
  2) from step10_congestion import run_step;  run_step(out_dir)
"""
import argparse
import os

import numpy as np

import _core as K
import step8_latency as s8

# ---- 场景锚点（与 step8/step9 的 10P 同源，seed_offset=10）----
SCENE_PARTICLES = 10
SCENE_SEED = 10
WINDOW_US = s8.WINDOW_US          # 10 ms，流式窗口

# ---- 控制器三档（论文 §6.4 原文，2026-10-21 起直接用论文 s 口径，发现A）----
#   "pacing < 2 Mbps → ν = 0.1；< 4 Mbps → ν = 0.25；否则 transmit all"
#   s = 式(2) 匹配分数（越小越好），滤波保留 s <= ν → ν 越小保留越少（越严）。
PACE_STRICT_MBPS = 2.0            # < 2 Mbps  → ν = 0.1（最严）
PACE_MID_MBPS = 4.0               # < 4 Mbps  → ν = 0.25（中）
NU_LEVELS = (0.1, 0.25, 1.0)      # 三档 ν（2=放全：s<=1.0 恒真 = transmit all）
LEVEL_NAME = ("strict(ν=0.1, p<2Mbps)", "mid(ν=0.25, p<4Mbps)", "full(transmit all, s<=1.0)")

SMOOTH_SLOTS = 10                 # 100 ms 滑动平均

ALGO_ORDER = ("CUBIC", "BBR", "COPA")


# ==================================================================
#  一、合成 pacing 轨迹（差异 #12：不接真 QUIC，用这三个发生器代替）
# ==================================================================

def _pacing_cubic(rng, n, dt=K.C.DT_S):
    """CUBIC：丢包型锯齿。每周期 1.4 s（10 s 内约 7 个周期）：
    1 Mbps **指数慢起**（每 RTT 翻倍的味道）→ 涨到 12 Mbps → 保持一小会儿 → 丢包重置回 1 Mbps。

    ★偏离论文：真实 CUBIC 是"遇丢包瞬时掉到 60%、再涨回同一个 W_max"，那样 pacing 会长期停在
      ≥ 7.2 Mbps，ν 的中/严两档永远触发不到。这里让丢包表现为一次**超时重置回 1 Mbps**，
      每个周期都从头扫过 2 / 4 Mbps 两条线 —— 见抬头「差异 #12」。
    """
    period = int(1.4 / dt)                             # 140 格 = 1.4 s
    rise = int(1.1 / dt)                               # 110 格指数慢起
    hold = max(1, int(0.1 * period))                   # 峰值维持一小会儿
    peak = 12.0
    out = np.zeros(n, dtype=float)
    for i in range(n):
        ph = i % period
        if ph < rise:
            out[i] = float(np.exp((ph / rise) * np.log(peak)))      # 1 → 12 Mbps
        elif ph < rise + hold:
            out[i] = peak
        else:
            out[i] = 1.0                                            # 丢包 → 重置
    return out


def _pacing_bbr(rng, n, dt=K.C.DT_S):
    """BBR：带宽探测型平台。稳在 8 Mbps；每 2 s 探测 +25% 持续 200 ms；周期性排空。

    ★偏离论文：论文的排空是"每 5 s 排空 1 s、最深回落到 4 Mbps"，这里改成
      **每 1.3 s 排空 1 s、最深到 1.6 Mbps**。两处偏离是同一个原因：按论文的数字，
      排空只占 20% 的时间且下限正好卡在 4 Mbps，"<4 Mbps" 和 "<2 Mbps" 两档加起来只有几十个
      窗口 —— ν 的三档覆盖不全，而且 pacing 的低三分位会落在同一档 ν' 里，点/秒的三分位分不开，
      验收 b) 直接不成立。探测尖峰仍然可见（排空优先判定，落在排空之外的探测才生效）。
    """
    plateau, probe_gain = 8.0, 1.25
    floor = 1.6                                        # ★偏离论文：4 → 1.6 Mbps
    every_probe = int(2.0 / dt)                        # 200 格 = 2 s
    probe_len = int(0.2 / dt)                          # 20 格 = 200 ms
    every_drain = int(1.3 / dt)                        # ★偏离论文：5 s → 1.3 s
    drain_len = int(1.0 / dt)                          # 100 格 = 1 s
    ramp = int(0.4 / dt)                               # 40 格斜坡：下去 / 上来

    out = np.full(n, plateau, dtype=float)
    for i in range(n):
        ph = i % every_drain
        if ph < drain_len:                             # 排空 1 s：降下去 → 蹲一会儿 → 爬回来
            if ph < ramp:
                out[i] = plateau + (floor - plateau) * (ph / ramp)
            elif ph < drain_len - ramp:
                out[i] = floor
            else:
                out[i] = floor + (plateau - floor) * ((ph - (drain_len - ramp)) / ramp)
        elif i % every_probe < probe_len:              # 周期性带宽探测
            out[i] = plateau * probe_gain
    return out


def _pacing_copa(rng, n, dt=K.C.DT_S):
    """COPA：延迟型。均值回复随机游走 + 低通 ⇒ 反应迟钝（论文说"不如丢包型灵敏"）。

    ★偏离论文：论文给的是 4~9 Mbps，这里让随机游走跨到 0.8~10 Mbps 并越过 2 Mbps 下限，
      理由同 BBR —— 否则 ν 的最严一档不会被触发。低通窗 15 格（150 ms）保留迟滞感。
    """
    mu, sigma, lo, hi = 3.2, 1.2, 0.8, 10.0
    raw = np.empty(n, dtype=float)
    x = mu
    for i in range(n):
        x = x + 0.02 * (mu - x) + rng.normal(0.0, sigma)   # 弱均值回复 + 高斯扰动
        if x < lo:
            x = lo + (lo - x)                              # 反射边界
        if x > hi:
            x = hi - (x - hi)
        raw[i] = x
    return _moving_avg(raw, int(0.15 / dt))                 # 低通 ≈ 150 ms ⇒ 迟滞


ALGOS = {"CUBIC": _pacing_cubic, "BBR": _pacing_bbr, "COPA": _pacing_copa}


# ==================================================================
#  二、工具
# ==================================================================

def _moving_avg(x, w):
    """后向滑动平均（只看历史，符合因果），边缘自适应窗宽。"""
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or w <= 1:
        return x.copy()
    c = np.concatenate(([0.0], np.cumsum(x)))
    lo = np.clip(np.arange(len(x)) - (w - 1), 0, len(x))
    hi = np.arange(1, len(x) + 1)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def _level_of(pacing_mbps):
    """论文规则：pacing < 2 → 最严(0)；< 4 → 中(1)；否则放全(2)。"""
    if pacing_mbps < PACE_STRICT_MBPS:
        return 0
    if pacing_mbps < PACE_MID_MBPS:
        return 1
    return 2


def _pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _xcorr_lag(a, b, max_lag=10):
    """互相关峰值滞后（单位：格）。corr(L) = Σ a[k]·b[k+L]，L>0 表示 b 滞后 a 的格数。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa, sb = a.std(), b.std()
    if len(a) < 2 or sa == 0 or sb == 0:
        return 0
    az = (a - a.mean()) / sa
    bz = (b - b.mean()) / sb
    best, best_lag = -np.inf, 0
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            s = float(np.sum(az[:len(az) - L] * bz[L:])) if L < len(az) else -np.inf
        else:
            s = float(np.sum(az[-L:] * bz[:len(bz) + L]))
        if s > best:
            best, best_lag = s, L
    return int(best_lag)


def _tercile_means(pacing, pps):
    """按 pacing 排序切低/中/高三段（各 1/3 窗口），返回三段的平均点/秒。"""
    order = np.argsort(pacing, kind="stable")
    n = len(order)
    chunks = np.array_split(order, 3)
    return [float(np.mean(pps[c])) if len(c) else float("nan") for c in chunks]


# ==================================================================
#  三、场景：流式窗口 + 坑#8 回看；s = 式(2) 分数随匹配直接产出
# ==================================================================

_SCENE_CACHE = {}


def load_scene(n_particles=SCENE_PARTICLES, seed_offset=SCENE_SEED, window_us=WINDOW_US):
    """返回 dict：每窗口的点数 / 每窗口的 q 切片 / 窗口时间轴 / 三档标定好的 ν'。"""
    key = (n_particles, seed_offset, window_us)
    if key in _SCENE_CACHE:
        return _SCENE_CACHE[key]

    left, right, gt = K.ge.generate(n_particles, seed_offset)
    left, right = K.flt.filter_events(left), K.flt.filter_events(right)

    t0_all = min(float(left[:, 0].min()), float(right[:, 0].min()))
    t1_all = max(float(left[:, 0].max()), float(right[:, 0].max()))
    edges = np.arange(t0_all, t1_all + window_us, window_us)
    n_win = len(edges) - 1

    # ★ 坑#8：窗口必须带 ±T_MAX_US 回看/前瞻，否则骑在窗口边界上的事件对会被切开。
    look = K.C.T_MAX_US
    win_pts = []
    for k in range(n_win):
        a, b = edges[k], edges[k + 1]
        lw = left[(left[:, 0] >= a - look) & (left[:, 0] < b + look)]
        rw = right[(right[:, 0] >= a - look) & (right[:, 0] < b + look)]
        if len(lw) == 0 or len(rw) == 0:
            win_pts.append(np.zeros((0, 4)))
            continue
        m = K.mt.match_and_triangulate(lw, rw, return_scores=True)   # (M,5)=[t,X,Y,Z,s]
        if len(m):                                     # 只保留落在本窗口的输出点
            m = m[(m[:, 0] >= a) & (m[:, 0] < b)]
        win_pts.append(m)

    lens = np.array([len(w) for w in win_pts])
    stacked = np.vstack([w for w in win_pts if len(w)]) if lens.sum() else np.zeros((0, 5))

    # ★ 2026-10-21 论文口径（发现A）：s 直接来自式(2)，无需再算 O(n²) 的密度代理。
    #   逐窗口的 s 切片预切好，控制器只做 s <= ν 比较（O(n)）。
    s_all = stacked[:, 4] if len(stacked) else np.zeros(0)
    s_per_win = np.split(s_all, np.cumsum(lens)[:-1]) if len(s_all) else [np.zeros(0)] * n_win

    t_ms = (edges[:-1] - t0_all) / 1e3
    scene = {
        "n_win": n_win, "win_s": window_us / 1e6, "t_ms": t_ms,
        "matched_per_win": lens, "s_per_win": s_per_win,
        "n_points_total": int(lens.sum()),
        "s_all": s_all,
    }
    _SCENE_CACHE[key] = scene
    return scene


# ==================================================================
#  四、控制器标定：三档 ν'（论文 s≤ν 口径 → 我们的 q≥ν' 口径）
# ==================================================================



# ==================================================================
#  五、单个算法的闭环
# ==================================================================

def run_algo(name, scene, rng=None, pacing=None):
    """单算法闭环：pacing -> 三档 ν（论文原值 0.1/0.25/1.0） -> s<=ν 过滤 -> 点/秒。"""
    n = scene["n_win"]
    if pacing is None:
        if rng is None:
            rng = np.random.default_rng(K.C.SEED)
        pacing = ALGOS[name](rng, n)
    pacing = np.asarray(pacing, dtype=float)
    if len(pacing) < n:                       # 发生器万一给短了，补最后一格
        pacing = np.concatenate([pacing, np.full(n - len(pacing), pacing[-1])])
    pacing = pacing[:n]

    levels = np.array([_level_of(p) for p in pacing], dtype=int)
    nu = np.asarray(NU_LEVELS, dtype=float)
    nu_win = nu[levels]

    matched = scene["matched_per_win"]
    kept = np.array([int(np.sum(s <= nu_win[k])) for k, s in enumerate(scene["s_per_win"])],
                    dtype=np.int64)
    pps = kept / scene["win_s"]
    pps_smooth = _moving_avg(pps, SMOOTH_SLOTS)

    # ---- 验收量 ----
    # a) 逐窗口是否与 pacing 规则零违例
    expect = np.array([nu[_level_of(p)] for p in pacing])
    violations = int(np.sum(nu_win != expect))
    have = sorted({round(float(v), 12) for v in nu_win})
    # b) 三分位严格递增 + Pearson r
    tm = _tercile_means(pacing, pps)
    r = _pearson(pacing, pps)
    # c) 各档实测保留率（论文口径下无"目标保留率"，只报告实测值）
    ret_real = np.array([(kept[levels == L].sum() / matched[levels == L].sum())
                         if matched[levels == L].sum() else float("nan")
                         for L in range(3)])
    n_win_per_level = np.array([int(np.sum(levels == L)) for L in range(3)])
    # d) 互相关峰值滞后
    lag = _xcorr_lag(pacing, pps)

    rows = [(name, k, float(scene["t_ms"][k]), float(pacing[k]), float(nu_win[k]),
             int(matched[k]), int(kept[k]), float(pps[k])) for k in range(n)]

    return {
        "algo": name, "pacing": pacing, "levels": levels, "nu_win": nu_win,
        "matched": matched, "kept": kept, "pps": pps, "pps_smooth": pps_smooth,
        "r": r, "tercile": tm, "violations": violations, "nu_values": have,
        "ret_real": ret_real, "n_win_per_level": n_win_per_level,
        "lag": lag, "rows": rows,
    }


# ==================================================================
#  五、入口
# ==================================================================

def run_step(out_dir=None, verbose=True):
    K.banner("Step10", "拥塞闭环：pacing → ν → 点/秒",
             "§6.4 Figure 14（★ 差异#12：合成 pacing 轨迹，不接真 QUIC）")

    out_dir = K.ensure_dir(out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "experiments"))
    scene = load_scene()
    if verbose:
        print(f"[scene] {SCENE_PARTICLES}P seed_offset={SCENE_SEED}：{scene['n_win']} 个窗口"
              f"（{scene['win_s'] * 1e3:.0f} ms/窗, 带 ±{K.C.T_MAX_US:.0f} µs 回看），"
              f"共 {scene['n_points_total']} 个匹配点")
        print(f"[口径] 2026-10-21 起用论文 s 口径（式(2) 分数，保留 s<=ν），"
              f"三档 ν = {NU_LEVELS}（论文原值），不再反标定。")
    # ---- 三条 pacing 轨迹（论文口径下无需标定，ν 直接查论文表）----
    rng = np.random.default_rng(K.C.SEED)
    pacings = {}
    for name in ALGO_ORDER:
        pacings[name] = np.asarray(ALGOS[name](rng, scene["n_win"]), dtype=float)

    results, rows_all = {}, []
    for name in ALGO_ORDER:
        res = run_algo(name, scene, rng, pacing=pacings[name])
        results[name] = res
        rows_all.extend(res["rows"])
        if verbose:
            print(f"\n[{name:<5}] pacing {res['pacing'].min():.2f}~{res['pacing'].max():.2f} Mbps | "
                  f"各档窗口数 严/中/全 = {res['n_win_per_level'][0]}/"
                  f"{res['n_win_per_level'][1]}/{res['n_win_per_level'][2]}")
            print(f"         Pearson r = {res['r']:.3f} | 三分位点/秒 "
                  f"{res['tercile'][0]:.1f} → {res['tercile'][1]:.1f} → {res['tercile'][2]:.1f}")
            print(f"         违例窗口 {res['violations']} | ν' 取值 {res['nu_values']} | "
                  f"互相关滞后 {res['lag']} 格")
            print(f"         各档保留率实测 "
                  + " / ".join(f"{LEVEL_NAME[L].split('(')[0]}"
                               f" {res['ret_real'][L]:.3f}"
                               for L in range(3)))

    with open(os.path.join(out_dir, "congestion.csv"), "w", encoding="utf-8-sig") as f:
        f.write("algorithm,window_idx,t_ms,pacing_mbps,nu_level,matched,kept,points_per_sec\n")
        for row in rows_all:
            f.write(f"{row[0]},{row[1]},{row[2]:.1f},{row[3]:.3f},{row[4]:.6f},"
                    f"{row[5]},{row[6]},{row[7]:.1f}\n")

    _plot(results, scene, out_dir)

    accept = acceptance(results)
    if verbose:
        print(f"\n[验收a] ν' 离散取值个数 = {accept['n_nu_values']}（应为 3）；"
              f"逐窗口违例总数 = {accept['violations_total']}")
        for name in ALGO_ORDER:
            res = results[name]
            print(f"[验收b] {name:<5} r = {res['r']:.3f} | 三分位 "
                  f"{res['tercile'][0]:.1f} / {res['tercile'][1]:.1f} / {res['tercile'][2]:.1f}"
                  f"（严<全 = {res['tercile'][0] < res['tercile'][2]}；COPA 迟滞只报告）")
            print(f"[验收c] {name:<5} 各档保留率 "
                  + " / ".join(f"{LEVEL_NAME[L].split('(')[0]} "
                               f"{res['ret_real'][L]:.3f}"
                               for L in range(3))
                  + "（应 严<=中<=全 且 全档=1）")
            print(f"[验收d] {name:<5} 互相关峰值滞后 = {res['lag']} 格（应 ≤2 格 = 20 ms）")
        print(f"[ALL  ] 四条验收全通过 = {accept['all_ok']}")
        print(f"[out  ] 已保存：{out_dir}\\fig14_congestion.png / congestion.csv")
    return {"scene": scene, "results": results, "accept": accept}


def acceptance(results):
    """四条验收的汇总判定（TC13 直接引用）。"""
    nu_vals = sorted({round(float(v), 12) for name in results
                      for v in results[name]["nu_win"]})
    viol_total = int(sum(results[n]["violations"] for n in results))
    ok_a = (len(nu_vals) == 3) and (viol_total == 0)
    # b) 三分位 严<全（论文 s 口径下好数据 ν=0.25≈不滤，中/全档天然贴近，不硬卡中档严格递增）
    ok_b = all(results[n]["tercile"][0] < results[n]["tercile"][2] for n in results)
    # c) 保留率单调不降：严档 ν=0.1 保留最少，全档 transmit all 保留 100%
    ret = {n: results[n]["ret_real"] for n in results}
    ok_c = all(np.isfinite(ret[n]).all() and
               ret[n][0] <= ret[n][1] <= ret[n][2] and abs(ret[n][2] - 1.0) < 1e-9
               for n in results)
    ok_d = all(abs(results[n]["lag"]) <= 2 for n in results)
    return {"n_nu_values": len(nu_vals), "violations_total": viol_total,
            "ok_a": ok_a, "ok_b": ok_b, "ok_c": ok_c, "ok_d": ok_d,
            "all_ok": bool(ok_a and ok_b and ok_c and ok_d)}


# ==================================================================
#  出图（Fig.14：三个算法各一个子图，左轴 pacing，右轴点/秒，背景标 ν' 档）
# ==================================================================

_LEVEL_COLOR = {0: "#f6d5d5", 1: "#fbeecd", 2: "#dff0dc"}     # 严 / 中 / 放全


def _plot(results, scene, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    t = scene["t_ms"] / 1e3        # 横轴用秒

    for ax, name in zip(axes, ALGO_ORDER):
        res = results[name]

        # 背景带：把连续同档窗口合并成一段，避免画上千个 patch
        lv = res["levels"]
        start = 0
        legend_done = set()
        for i in range(1, len(lv) + 1):
            if i == len(lv) or lv[i] != lv[start]:
                lab = LEVEL_NAME[lv[start]] if lv[start] not in legend_done else None
                legend_done.add(lv[start])
                ax.axvspan(t[start], t[min(i, len(t) - 1)], color=_LEVEL_COLOR[lv[start]],
                           alpha=.85, lw=0, label=lab)
                start = i

        ax.step(t, res["pacing"], where="post", color="#185FA5", lw=1.2, label="pacing (Mbps)")
        ax.set_ylabel("pacing (Mbps)", color="#185FA5")
        ax.set_ylim(0, max(11.0, float(res["pacing"].max()) * 1.15))
        ax.grid(alpha=.2)

        ax2 = ax.twinx()
        ax2.plot(t, res["pps_smooth"], color="#d1495b", lw=1.3,
                 label="points/sec (100 ms MA)")
        ax2.set_ylabel("triangulated points / sec", color="#d1495b")
        ax2.set_ylim(0, max(1.0, float(np.nanmax(res["pps_smooth"])) * 1.15))

        ax.set_title(f"{name}: pacing vs reconstruction rate  "
                     f"(r={res['r']:.3f}, lag={res['lag']} slot)")
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7.5, loc="upper right", ncol=2)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Fig.14 analog: congestion loop closes pacing → ν → points/sec "
                 "(★ synthetic pacing, no real QUIC — see 差异#12)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "fig14_congestion.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Step10 拥塞闭环（Fig.14）")
    ap.add_argument("--out", default=None, help="输出目录（默认 py_repro/experiments）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_step(args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
