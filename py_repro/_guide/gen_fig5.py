# -*- coding: utf-8 -*-
"""生成 block_5.html：图 5-1 聚类半径 ρ 的三档后果（同一份点云）"""
import math
import random

random.seed(23)

# ---- 造 3 个团（模拟 3 颗粒子在同一帧里的点簇）----
groups = [(70, 105, 24), (178, 75, 23), (142, 168, 22)]
pts = []
for cx, cy, r in groups:
    for _ in range(10):
        a = random.uniform(0, 2 * math.pi)
        d = random.uniform(1, r)
        pts.append((cx + d * math.cos(a), cy + d * math.sin(a)))

# 1 cm 对应的像素（点云散布 20px ≈ 2 cm）
PX_PER_CM = 46.0


def greedy_cluster(points, radius):
    used = [False] * len(points)
    centers = []
    for i in range(len(points)):
        if used[i]:
            continue
        centers.append(i)
        used[i] = True
        for j in range(len(points)):
            if not used[j]:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                if math.hypot(dx, dy) <= radius:
                    used[j] = True
    return centers


CASES = [
    ("ρ = 0.5 cm　太小", "同一颗粒子被劈成好几个簇 → 簇数虚高、轨迹频繁断", 0.5),
    ("ρ = 1.0 cm　本项目的取值", "一颗粒子的点刚好归成一簇", 1.0),
    ("ρ = 2.5 cm　太大", "相邻的两颗粒子被并成一簇 → 轨迹打架", 2.5),
]

out = []
out.append('<div class="dia">')
out.append('<svg viewBox="0 0 1000 372" xmlns="http://www.w3.org/2000/svg" role="img" '
           'aria-label="聚类半径过大过小的三种后果">')
out.append('  <rect width="1000" height="372" fill="#ffffff"/>')
out.append('  <text x="30" y="30" font-size="15.5" font-weight="700" fill="#1d2129">'
           '聚类半径 ρ 是这套系统最“一根筋”的参数：小一点就碎，大一点就并</text>')
out.append('  <text x="30" y="50" font-size="12" fill="#86909c">'
           '同一份点云（3 颗粒子、每颗 10 个点），只改 ρ 跑三次。圈 = 一个簇的势力范围，实心大点 = 簇心</text>')

for pi, (title, sub, r_cm) in enumerate(CASES):
    x0 = 30 + pi * 320
    rad = r_cm * PX_PER_CM
    centers = greedy_cluster(pts, rad)
    ox, oy = x0 + 26, 86

    out.append('  <rect x="%d" y="66" width="300" height="264" rx="8" fill="#fcfdfe" stroke="#e3e6eb"/>'
               % x0)
    out.append('  <text x="%d" y="88" font-size="13.5" font-weight="700" fill="#1d2129">%s</text>'
               % (x0 + 12, title))

    out.append('  <clipPath id="cp%d"><rect x="%d" y="66" width="300" height="264" rx="8"/></clipPath>'
               % (pi, x0))
    out.append('  <g clip-path="url(#cp%d)">' % pi)
    # 势力圈
    for ci in centers:
        out.append('  <circle cx="%.1f" cy="%.1f" r="%.1f" fill="#e8f1fa" fill-opacity="0.55" '
                   'stroke="#7fb0d8" stroke-width="1"/>'
                   % (ox + pts[ci][0], oy + pts[ci][1], rad))

    # 点
    for (px, py) in pts:
        out.append('  <circle cx="%.1f" cy="%.1f" r="2.8" fill="#9db4c8"/>'
                   % (ox + px, oy + py))

    out.append('  </g>')
    # 簇心
    for ci in centers:
        out.append('  <circle cx="%.1f" cy="%.1f" r="5.2" fill="#b3352f"/>'
                   % (ox + pts[ci][0], oy + pts[ci][1]))

    out.append('  <text x="%d" y="300" font-size="12.5" font-weight="700" fill="#b3352f">'
               '→ %d 个簇</text>' % (x0 + 12, len(centers)))
    out.append('  <text x="%d" y="318" font-size="10.8" fill="#86909c">%s</text>' % (x0 + 12, sub))

out.append('  <text x="30" y="354" font-size="12" fill="#2f7a4f">'
           '本项目选 1 cm：粒子间距通常在 3–5 cm 以上，而同一颗粒子的点散布在 1 cm 内 —— '
           'ρ 就卡在“簇内”和“簇间”两个尺度中间，这是纯几何约束，不是调出来的经验值。（本图是 2D 投影，真实 3D 下 ρ 减半的后果比图里更碎）</text>')
out.append('</svg>')

out.append('<div class="cap"><span class="no">图 5-1</span>'
           '<b>贪心聚类的规则只有一句话：按顺序取一个还没被归簇的点当簇心，把它半径 ρ 内的所有点都划给它，然后继续。</b>'
           '所以 ρ 直接决定了“一颗粒子算几个簇”。'
           '<b>注意图中的圈会重叠</b>——这不是 bug：贪心算法里后选的簇心只吸收“还没被前面抢走”的点，两个圈重叠的部分早就归先来的那个了。'
           '真实实现用的是 3D 欧氏距离（<code>cluster.py</code> 的 <code>np.linalg.norm(pts - seed, axis=1) &lt;= R</code>），'
           '本图为了看清才投到 2D。图上数字是本机按同一套贪心规则跑出来的，不是示意值。</div>')
out.append('</div>')

open("block_5.html", "w", encoding="utf-8").write("\n".join(out))
print("生成 block_5.html")
for title, _, r in CASES:
    print("  %-22s → %d 个簇" % (title, len(greedy_cluster(pts, r * PX_PER_CM))))
