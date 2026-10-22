# -*- coding: utf-8 -*-
r"""
evt3_decoder.py —— Prophesee EVT3 (.raw) -> numpy 事件数组（纯 numpy，无需商业 SDK）
====================================================================================
★★★ 用途：Step0 的真实数据源解析器（RATE 模式：真实公开数据集是第一公民）★★★

格式（EVT 3.0，Prophesee Gen4 / IMX636 级传感器，1280x720）：
  - 文本头：每行以 % 开头，以 "% end\n" 结束
  - 之后全部是 16-bit 小端 word，高 4 bit (bits 15:12) 为事件类型：
      0x0 ADDR_Y      : bits 10:0 = y, bit 11 = system_type
      0x2 ADDR_X      : bits 10:0 = x, bit 11 = polarity
      0x3 VECT_BASE_X : bits 10:0 = 基准 x, bit 11 = polarity
      0x4 VECT_12     : bits 11:0 = 12-bit 有效位掩码 (x = 基准x + 位移i)
      0x5 VECT_8      : bits 7:0  = 8-bit  有效位掩码 (x = 基准x + 位移i)
      0x6 TIME_LOW    : bits 11:0 = 时间戳低 12 bit (µs)
      0x8 TIME_HIGH   : bits 11:0 = 时间戳中 12 bit (bits 23:12)
      0xA EXT_TRIGGER : 外触发
  - 完整时间戳 = (回绕计数 * 0x1000 + time_high) * 0x1000 + time_low  [µs]

来历（台账 2026-10-19）：本解码器最早由 workbuddy 写在 faithful_repro/evt3_to_npz.py
（当时修过一次关键 bug：向量事件掩码必须 .astype(bool)，否则整数索引把事件虚增 12 倍）。
这里移植进 py_repro 主线供 Step0 调用——解析器也是"只写一份"的资产，
faithful_repro 的 CLI 版保留作历史对照，两边算法一致。

独立验证：对 karman_199ms.raw 的解码结果与 faithful_repro 版逐位一致
（3,111,957 事件 @ 前 0.2 s，见 参考答案/karman_real.json）。
"""
import numpy as np

WIDTH, HEIGHT = 1280, 720          # Prophesee Gen4（karmanFlow 数据集的实际分辨率）


def read_words(raw_path):
    """读取 raw 文件，跳过文本头，返回 16-bit 小端 word 数组。"""
    with open(raw_path, "rb") as f:
        data = f.read()
    end = data.find(b"% end\n")
    if end < 0:
        raise ValueError("未找到 '% end' 头部结束标记, 不是 EVT3 文件?")
    off = end + len(b"% end\n")
    if off % 2 == 1:               # word 对齐
        off += 1
    return np.frombuffer(data, dtype="<u2", count=(len(data) - off) // 2, offset=off)


def decode_evt3(words):
    """解码 EVT3 word 数组，返回 (x, y, t, p) 四个数组（按文件顺序）与统计 dict。"""
    types = (words >> 12).astype(np.uint8)
    pos = np.arange(len(words), dtype=np.int64)

    def last_value(mask, vals):
        """对每个位置取它之前(含本位)最近一个 mask 事件的值。"""
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return np.zeros(len(words), dtype=vals.dtype), np.full(len(words), -1, dtype=np.int64)
        k = np.searchsorted(idx, pos, side="right") - 1
        k = np.maximum(k, 0)       # 第一个事件之前无定义，取首个值（不应出现 CD 事件）
        return vals[k], k

    # ---- 时间戳 ----
    th_mask = types == 8
    th_idx = np.flatnonzero(th_mask)
    th_vals = (words[th_idx] & 0x0FFF).astype(np.int64)
    # time_high 回绕计数：值变小说明进位
    if len(th_vals) > 1:
        wraps = np.concatenate([[0], np.cumsum(th_vals[1:] < th_vals[:-1])])
    else:
        wraps = np.zeros(len(th_vals), dtype=np.int64)
    high_for, k_hi = last_value(th_mask, th_vals)
    cyc_for = wraps[k_hi]

    tl_mask = types == 6
    tl_vals = (words[tl_mask] & 0x0FFF).astype(np.int64)
    low_for, _ = last_value(tl_mask, tl_vals)

    t_all = (cyc_for * 0x1000 + high_for) * 0x1000 + low_for   # µs

    # ---- 当前 y (ADDR_Y) ----
    ay_mask = types == 0
    ay_vals = (words[ay_mask] & 0x07FF).astype(np.int64)
    y_for, _ = last_value(ay_mask, ay_vals)

    # ---- 单事件 ADDR_X ----
    ax_mask = types == 2
    pos_ax = pos[ax_mask]
    x_ax = (words[ax_mask] & 0x07FF).astype(np.int64)
    y_ax = y_for[ax_mask]
    t_ax = t_all[ax_mask]
    p_ax = ((words[ax_mask] >> 11) & 0x1).astype(np.int8)

    # ---- 向量事件 ----
    vb_mask = types == 3
    vb_vals = (words[vb_mask] & 0x07FF).astype(np.int64)
    vb_pol = ((words[vb_mask] >> 11) & 0x1).astype(np.int8)
    base_for, _ = last_value(vb_mask, vb_vals)
    basepol_for, _ = last_value(vb_mask, vb_pol)

    vec_pos, vec_x, vec_y, vec_t, vec_p = [], [], [], [], []
    for vtype, nbits in ((4, 12), (5, 8)):
        vmask = types == vtype
        if not vmask.any():
            continue
        vpos = pos[vmask]
        valid = (words[vmask] & (0x0FFF if nbits == 12 else 0x00FF)).astype(np.int64)
        bx = base_for[vmask]
        bp = basepol_for[vmask]
        vy = y_for[vmask]
        vt = t_all[vmask]
        for i in range(nbits):
            # 必须转 bool：(valid>>i)&1 是 0/1 整数数组，若直接当索引会变成
            # "按位置取 vpos[0]/vpos[1]" 的整数花式索引，每位都返回全部元素（虚增 12 倍）
            sel = ((valid >> i) & 1).astype(bool)
            if not sel.any():
                continue
            vec_pos.append(vpos[sel])
            vec_x.append(bx[sel] + i)
            vec_y.append(vy[sel])
            vec_t.append(vt[sel])
            vec_p.append(bp[sel])

    if vec_pos:
        pos_v = np.concatenate(vec_pos)
        x_v = np.concatenate(vec_x)
        y_v = np.concatenate(vec_y)
        t_v = np.concatenate(vec_t)
        p_v = np.concatenate(vec_p)
    else:
        pos_v = np.zeros(0, dtype=np.int64)
        x_v = y_v = t_v = np.zeros(0, dtype=np.int64)
        p_v = np.zeros(0, dtype=np.int8)

    # ---- 合并并按文件顺序排序 ----
    pos_all = np.concatenate([pos_ax, pos_v])
    X = np.concatenate([x_ax, x_v])
    Y = np.concatenate([y_ax, y_v])
    T = np.concatenate([t_ax, t_v])
    P = np.concatenate([p_ax, p_v])
    order = np.argsort(pos_all, kind="stable")
    X, Y, T, P = X[order], Y[order], T[order], P[order]

    stats = {
        "n_words": int(len(words)),
        "n_events": int(len(X)),
        "n_addr_x": int(ax_mask.sum()),
        "n_vect12": int((types == 4).sum()),
        "n_vect8": int((types == 5).sum()),
    }
    return X, Y, T, P, stats


def decode_raw_file(raw_path, max_dur_us=None):
    """一步到位：raw 路径 -> (x, y, t, p)（t 已减去首事件时间、从 0 起，µs）。"""
    words = read_words(raw_path)
    X, Y, T, P, st = decode_evt3(words)
    if len(X) == 0:
        raise SystemExit(f"[×] {raw_path} 未解出任何事件")
    T = T - T.min()
    if max_dur_us is not None:
        keep = T <= max_dur_us
        X, Y, T, P = X[keep], Y[keep], T[keep], P[keep]
        st = dict(st, n_events_kept=int(len(X)), max_dur_us=int(max_dur_us))
    return X, Y, T, P, st
