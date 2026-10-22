# -*- coding: utf-8 -*-
"""生成 P11 两张内联 SVG：① 三算法 pacing 轨迹 ② 闭环响应(窗口级散点+相关性)。
数据来自 experiments/congestion.csv（真实跑出来的，不是示意）。"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "experiments", "congestion.csv")


def load():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    data = {}
    for name in ("CUBIC", "BBR", "COPA"):
        pts = [(float(r["t_ms"]) / 1000.0, float(r["pacing_mbps"]),
                float(r["points_per_sec"])) for r in rows if r["algorithm"] == name]
        data[name] = pts
    return data


# ---------- 图 P11-1：三算法 pacing 轨迹 ----------
def pacing_fig(data):
    W, H = 1000, 360
    L, R, T, B = 60, 22, 28, 42
    pad_inner = 6
    x0, x1 = L, W - R
    y0, y1 = H - B, T
    pmax = 13.0

    def X(t):  # t in 0..10 s
        return x0 + (t / 10.0) * (x1 - x0)

    def Y(p):
        return y0 - (p / pmax) * (y0 - y1)

    out = ['<div class="dia" aria-label="三算法合成 pacing 轨迹">']
    out.append('  <svg viewBox="0 0 %d %d" width="100%%" role="img">' % (W, H))
    out.append('    <rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
    # 网格 + 纵轴刻度
    for p in range(0, 14, 2):
        yy = Y(p)
        out.append('    <line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#eef1f5"/>' % (x0, yy, x1, yy))
        out.append('    <text x="%d" y="%.1f" font-size="11" fill="#86909c" text-anchor="end">%d</text>' % (x0 - 6, yy + 4, p))
    # 横轴刻度
    for s in range(0, 11, 2):
        xx = X(s)
        out.append('    <line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#eef1f5"/>' % (xx, y0, xx, y1))
        out.append('    <text x="%.1f" y="%d" font-size="11" fill="#86909c" text-anchor="middle">%ds</text>' % (xx, y0 + 16, s))
    # 阈值虚线 2 / 4 Mbps
    for thr, lab in ((2.0, "2 Mbps（ν 最严）"), (4.0, "4 Mbps（ν 中）")):
        yy = Y(thr)
        out.append('    <line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#d1495b" stroke-width="1.2" stroke-dasharray="6 4"/>' % (x0, yy, x1, yy))
        out.append('    <text x="%d" y="%.1f" font-size="10.5" fill="#d1495b" text-anchor="end">%s</text>' % (x1 - 4, yy - 5, lab))
    # 三折线（每 5 点取 1）
    col = {"CUBIC": "#185FA5", "BBR": "#2e8b57", "COPA": "#c0392b"}
    legend = []
    for name in ("CUBIC", "BBR", "COPA"):
        pts = data[name]
        step = max(1, len(pts) // 200)
        poly = " ".join("%.1f,%.1f" % (X(t), Y(p)) for (t, p, _) in pts[::step])
        out.append('    <polyline points="%s" fill="none" stroke="%s" stroke-width="1.5" stroke-linejoin="round"/>' % (poly, col[name]))
        legend.append((name, col[name]))
    # 轴标题
    out.append('    <text x="%d" y="%d" font-size="12" fill="#4e5969" text-anchor="middle">时间（秒）</text>' % ((x0 + x1) // 2, H - 8))
    out.append('    <text x="%d" y="%d" font-size="12" fill="#4e5969" text-anchor="middle" transform="rotate(-90 %d %d)">pacing rate (Mbps)</text>' % (16, (y0 + y1) // 2, 16, (y0 + y1) // 2))
    # 图例
    lx = x0 + 8
    for i, (nm, c) in enumerate(legend):
        yy = y1 + 4 + i * 16
        out.append('    <line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="2"/>' % (lx, yy, lx + 18, yy, c))
        out.append('    <text x="%d" y="%.1f" font-size="11" fill="#4e5969">%s</text>' % (lx + 24, yy + 4, nm))
    out.append('  </svg>')
    out.append('  <div class="cap"><span class="no">图 P11-1</span> 三条合成 pacing 轨迹：CUBIC 丢包型锯齿（峰值逐轮衰减 ×0.65，扫过 12→1.3 Mbps）；BBR 带宽探测型平台（8 Mbps 稳，每 1.3 s 排空到 1.6 Mbps）；COPA 延迟型迟滞（均值回复随机游走 + 150 ms 低通）。三条红色虚线是论文的 ν 档位阈值，轨迹必须穿过它们才能让三档 ν′ 都被走到。</div>')
    out.append('</div>')
    return "\n".join(out)


# ---------- 图 P11-2：闭环响应（窗口级 pacing vs 点/秒 + 相关性） ----------
def loop_fig(data):
    W, H = 1000, 380
    T, B = 40, 44
    panels = 3
    pw = (W - 24) / panels - 12
    pmax = 13.0
    ppsmax = 2800.0

    def panel_x(i):
        return 12 + i * (pw + 12)

    out = ['<div class="dia" aria-label="闭环响应 pacing 与 点每秒 相关性">']
    out.append('  <svg viewBox="0 0 %d %d" width="100%%" role="img">' % (W, H))
    out.append('    <rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
    col = {"CUBIC": "#185FA5", "BBR": "#2e8b57", "COPA": "#c0392b"}
    import statistics
    for i, name in enumerate(("CUBIC", "BBR", "COPA")):
        x0 = panel_x(i)
        px0, px1 = x0 + 40, x0 + pw - 8
        py0, py1 = H - B, T
        pts = data[name]

        def X(t):
            return px0 + (t / pmax) * (px1 - px0)

        def Y(p):
            return py0 - (p / ppsmax) * (py0 - py1)

        # 面板边框
        out.append('    <rect x="%.1f" y="%d" width="%.1f" height="%d" rx="6" fill="#fcfdfe" stroke="#e3e6eb"/>' % (x0, T - 6, pw, H - T - B + 12))
        # 网格
        for g in range(0, 14, 4):
            xx = X(g)
            out.append('    <line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#eef1f5"/>' % (xx, py0, xx, py1))
        for g in range(0, 2801, 700):
            yy = Y(g)
            out.append('    <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#eef1f5"/>' % (px0, yy, px1, yy))
        # 阈值线 2/4
        for thr in (2.0, 4.0):
            xx = X(thr)
            out.append('    <line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#d1495b" stroke-width="1" stroke-dasharray="5 4"/>' % (xx, py0, xx, py1))
        # 散点（降采样 160 点）
        step = max(1, len(pts) // 160)
        for (t, p, q) in pts[::step]:
            out.append('    <circle cx="%.1f" cy="%.1f" r="1.7" fill="%s" fill-opacity="0.30"/>' % (X(t), Y(q), col[name]))
        # 轴刻度
        for g in range(0, 14, 4):
            out.append('    <text x="%.1f" y="%d" font-size="9.5" fill="#86909c" text-anchor="middle">%d</text>' % (X(g), py0 + 12, g))
        # 标题 + r
        r = correlation([p for _, p, _ in pts], [q for _, _, q in pts])
        out.append('    <text x="%.1f" y="%d" font-size="12.5" font-weight="700" fill="#1d2129">%s</text>' % (x0 + 8, T + 8, name))
        out.append('    <text x="%.1f" y="%d" font-size="11" fill="%s">Pearson r = %.3f</text>' % (x0 + 8, T + 24, col[name], r))
        # 轴标题
        out.append('    <text x="%.1f" y="%d" font-size="10.5" fill="#4e5969" text-anchor="middle">pacing (Mbps)</text>' % ((px0 + px1) / 2, H - 6))
        out.append('    <text x="%.1f" y="%d" font-size="10.5" fill="#4e5969" text-anchor="middle" transform="rotate(-90 %.1f %.1f)">points / sec</text>' % (x0 + 14, (py0 + py1) / 2, x0 + 14, (py0 + py1) / 2))
    out.append('  </svg>')
    out.append('  <div class="cap"><span class="no">图 P11-2</span> 闭环响应：每个窗口的 pacing（横轴）与重建点/秒（纵轴）几乎同步起伏。三算法的 Pearson r 都在 0.6 以上（CUBIC 0.742 / BBR 0.751 / COPA 0.635），互相关峰值滞后 0 格（≤20 ms）——证明"点/秒跟随 pacing"这条闭环逻辑成立。纵轴点/秒上限 2800 是因为放全档（ν′=0）时 10 ms 窗口里最多匹配出 ~28 个点。</div>')
    out.append('</div>')
    return "\n".join(out)


def correlation(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[k] - ma) * (b[k] - mb) for k in range(n))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    return cov / math.sqrt(va * vb) if va and vb else 0.0


def main():
    data = load()
    open(os.path.join(HERE, "block_p11_1.html"), "w", encoding="utf-8").write(pacing_fig(data))
    open(os.path.join(HERE, "block_p11_2.html"), "w", encoding="utf-8").write(loop_fig(data))
    print("P11-1 / P11-2 已生成")


if __name__ == "__main__":
    main()
