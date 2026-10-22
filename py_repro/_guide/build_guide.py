# -*- coding: utf-8 -*-
r"""
build_guide.py —— 把 guide_body.html 里的占位符换成"带真实行号的源码"
=====================================================================
为什么要有这玩意儿：
  讲解文档里要贴几十段源码，手抄必然抄错行号（然后读者按图索骥找不到）。
  这里直接按【文件:起始行:结束行】从磁盘抽，行号就是编辑器里的行号，
  一遍抄错的机会都不给。顺便做 HTML 转义（< > &）。

占位符语法（写在 guide_body.html 里）：
  {{CODE:相对路径:起:止}}            例：{{CODE:reproduce/config.py:7:20}}
  {{CODE:相对路径:起:止:高亮行}}      例：...:7:20:8,9,10   （高亮行加底色）
  {{FIG:相对路径|图注}}               例：{{FIG:py_repro/experiments/fig6.png|图9-1 ...}}

用法：
  D:\Users\admuj\anaconda3\envs\opencv_env\python.exe _guide\build_guide.py
输出：
  项目根目录/复现每一步精讲.html
"""
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))     # 项目根
BODY = os.path.join(HERE, "guide_body.html")
OUT = os.path.join(ROOT, "复现每一步精讲.html")

# {{CODE:path:a:b}} 或 {{CODE:path:a:b:1,2,3}}
RE_CODE = re.compile(r"\{\{CODE:([^:}]+):(\d+):(\d+)(?::([\d,]+))?\}\}")
# {{FIG:path|caption}}
RE_FIG = re.compile(r"\{\{FIG:([^|}]+)\|([^}]*)\}\}")
# {{FILE:path}}  -> 只显示文件名（小标题用）
RE_FILE = re.compile(r"\{\{FILE:([^}]+)\}\}")


def code_block(rel, a, b, hl):
    """抽 [a,b] 行（1-based，闭区间），生成 <pre class="code">。"""
    path = os.path.join(ROOT, rel)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    a, b = max(1, int(a)), min(len(lines), int(b))
    hlset = {int(x) for x in hl.split(",")} if hl else set()
    buf = ['<pre class="code">']
    for i in range(a, b + 1):
        cls = "cl hl" if i in hlset else "cl"
        buf.append('<span class="%s"><span class="ln">%d</span>%s</span>'
                   % (cls, i, html.escape(lines[i - 1])))
    buf.append("</pre>")
    return "\n".join(buf)


def figure(rel, cap):
    return ('<figure><img src="%s" alt="%s"><figcaption>%s</figcaption></figure>'
            % (rel, html.escape(cap), cap))


def main():
    with open(BODY, "r", encoding="utf-8") as f:
        s = f.read()

    n_code = [0]
    n_fig = [0]

    def _code(m):
        n_code[0] += 1
        return code_block(m.group(1), m.group(2), m.group(3), m.group(4))

    def _fig(m):
        n_fig[0] += 1
        return figure(m.group(1).strip(), m.group(2).strip())

    s = RE_CODE.sub(_code, s)
    s = RE_FIG.sub(_fig, s)
    s = RE_FILE.sub(lambda m: '<code>%s</code>' % html.escape(os.path.basename(m.group(1))), s)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(s)

    size = os.path.getsize(OUT) / 1024
    print("[ok] %s" % OUT)
    print("     代码块 %d 段, 插图 %d 张, %.1f KB" % (n_code[0], n_fig[0], size))


if __name__ == "__main__":
    sys.exit(main())
