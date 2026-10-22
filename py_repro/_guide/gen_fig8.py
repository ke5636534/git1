# -*- coding: utf-8 -*-
"""生成 block_8.html：图 8-1 κ 均匀抽稀 vs ν 质量门槛（同一份点云的三种命运）"""
import math
import random

random.seed(11)

# ---- 生成点云：3 个簇（信号）+ 12 个孤立点（噪声）----
clusters = [(70, 62, 26), (176, 118, 24), (84, 158, 22)]
pts = []
for ci, (cx, cy, r) in enumerate(clusters):
    for _ in range(9):
        a = random.uniform(0, 2 * math.pi)
        d = random.uniform(2, r)
        pts.append((cx + d * math.cos(a), cy + d * math.sin(a), ci))

noise = []
while len(noise) < 12:
    x, y = random.uniform(14, 246), random.uniform(14, 186)
    if min(math.hypot(x - cx, y - cy) for cx, cy, _ in clusters) > 42:
        noise.append((x, y))

# ---- 三种命运 ----
keep_kappa = [random.random() < 0.3 for _ in pts]
keep_kappa_noise = [random.random() < 0.3 for _ in noise]
# ν：簇内全留，噪声全删（真实算法是密度阈值，这里画极限情况）
keep_nu = [True] * len(pts)
keep_nu_noise = [False] * len(noise)

PANELS = [
    ("原始点云", "3 个粒子簇（蓝）+ 12 个孤立噪声（红）", None),
    ("κ = 0.3　均匀抽稀", "每个点各 30% 概率保留 —— 噪声和信号一起扔", "kappa"),
    ("ν = 0.7　质量门槛", "邻居 &lt; 阈值就整条丢掉 —— 孤立点先死，簇留下", "nu"),
]

out = []
out.append('<div class="dia">')
out.append('<svg viewBox="0 0 1000 350" xmlns="http://www.w3.org/2000/svg" role="img" '
           'aria-label="均匀抽稀与质量门槛对同一份点云的不同效果">')
out.append('  <rect width="1000" height="350" fill="#ffffff"/>')
out.append('  <text x="30" y="30" font-size="15.5" font-weight="700" fill="#1d2129">'
           '同一份点云，两种降载策略的差别：κ 是闭眼抓，ν 是先看质量再抓</text>')
out.append('  <text x="30" y="50" font-size="12" fill="#86909c">'
           '实心 = 保留下来送进网络的点；浅灰空心 = 被丢弃的点（位置不变，便于对比）</text>')

for pi, (title, sub, mode) in enumerate(PANELS):
    x0 = 30 + pi * 320
    # 面板框
    out.append('  <rect x="%d" y="70" width="300" height="232" rx="8" fill="#fcfdfe" '
               'stroke="#e3e6eb"/>' % x0)
    out.append('  <text x="%d" y="92" font-size="13.5" font-weight="700" fill="#1d2129">%s</text>'
               % (x0 + 12, title))
    out.append('  <text x="%d" y="108" font-size="10.5" fill="#86909c">%s</text>' % (x0 + 12, sub))

    ox, oy = x0 + 22, 116

    def dot(px, py, keep, is_noise):
        cx, cy = ox + px, oy + py * 0.84
        if keep:
            c = '#b3352f' if is_noise else '#185FA5'
            return ('  <circle cx="%.1f" cy="%.1f" r="3.4" fill="%s"/>\n' % (cx, cy, c))
        return ('  <circle cx="%.1f" cy="%.1f" r="2.6" fill="none" stroke="#dfe3e9" '
                'stroke-width="1.1"/>\n' % (cx, cy))

    if mode is None:
        for px, py, _ in pts:
            out.append(dot(px, py, True, False))
        for px, py in noise:
            out.append(dot(px, py, True, True))
    elif mode == "kappa":
        for i, (px, py, _) in enumerate(pts):
            out.append(dot(px, py, keep_kappa[i], False))
        for i, (px, py) in enumerate(noise):
            out.append(dot(px, py, keep_kappa_noise[i], True))
    else:
        for i, (px, py, _) in enumerate(pts):
            out.append(dot(px, py, keep_nu[i], False))
        for i, (px, py) in enumerate(noise):
            out.append(dot(px, py, keep_nu_noise[i], True))

    # 统计
    if mode is None:
        n_sig, n_noi = len(pts), len(noise)
    elif mode == "kappa":
        n_sig, n_noi = sum(keep_kappa), sum(keep_kappa_noise)
    else:
        n_sig, n_noi = sum(keep_nu), sum(keep_nu_noise)
    out.append('  <text x="%d" y="290" font-size="11.5" fill="#185FA5">保留 %d 个信号点</text>'
               % (x0 + 12, n_sig))
    out.append('  <text x="%d" y="290" font-size="11.5" fill="#b3352f">噪声 %d 个</text>'
               % (x0 + 140, n_noi))

# 箭头
for ax in (338, 658):
    out.append('  <text x="%d" y="190" font-size="19" fill="#185FA5" font-weight="700" text-anchor="middle">→</text>' % ax)

out.append('  <text x="30" y="334" font-size="12" fill="#2f7a4f">'
           'κ 保的是"公平"：每个点机会均等，但噪声也照样活下来四成（5/12）；'
           'ν 保的是"纯度"：噪声整条死，代价是得分低的真信号可能被误杀。</text>')
out.append('</svg>')

out.append('<div class="cap"><span class="no">图 8-1</span>'
           '<b>这张图要纠正一个最常见的误读：κ 和 ν 不是"两种降载率"，是两种完全不同的"扔谁"的规则。</b>'
           'κ 只管<b>扔多少</b>（每个点独立掷骰子，与它长什么样无关）；ν 只管<b>扔谁</b>'
           '（按局部密度打分，不达标的整条丢）。'
           '所以论文 §6.4 把它们分别对应到两个目标：<b>κ 压延迟</b>（Fig.11）、<b>ν 减噪声匹配</b>（Fig.12），'
           '而不是"谁的精度更高"。严格说 ν 的分数是 <code>quality_scores</code> 算的"半径 2 cm 内邻居数"，'
           '不是图上手绘的簇形状——但"孤立点分数低"这个结论是一致的。</div>')
out.append('</div>')

open("block_8.html", "w", encoding="utf-8").write("\n".join(out))
print("生成 block_8.html，", len(out), "行")
print("信号点保留：κ=%d/%d  ν=%d/%d；噪声保留：κ=%d/%d  ν=%d/%d"
      % (sum(keep_kappa), len(pts), sum(keep_nu), len(pts),
         sum(keep_kappa_noise), len(noise), sum(keep_nu_noise), len(noise)))
