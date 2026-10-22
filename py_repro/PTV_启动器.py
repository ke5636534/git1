# -*- coding: utf-8 -*-
r"""
============================================================
 粒子追踪复现 · 统一启动器（图形界面版）
============================================================
 一个窗口搞定所有事：
   ① 下拉框挑数据集（自动扫描 py_repro/data 下所有翻译好的数据集）
   ② 挑要跑哪一步（Step0~Step7，对应论文流程）
   ③ Step0 可以直接填粒子数新建仿真数据集
   ④ 点「开始处理」，日志实时滚动

 步骤与论文的对应关系：
   Step0  解析与翻译数据集 → npz + meta     （永远是第一步！§4.2/§5.2）
   Step1  事件去噪（热像素 + 不应期）        （§4 DVS 噪声滤波 [19]）
   Step2  双目匹配 式(2) + 三角测量 Z=f·B/d （§4 事件匹配）
   Step3  3D 点云聚类（1cm / 每簇≥2点）     （§4 DBSCAN）
   Step4  逐帧跟踪 GNN+卡尔曼+匈牙利        （§4 trackerGNN）
   Step5  精度评估 Cov/Prec/Chamfer         （§6.1 式(3)(4)(7)）
   Step6  流式实验 κ / ν                    （§6.4，Fig.12 复刻）
   Step7  粒子数扫描 1/2/3/5/10P            （§5，Fig.6 复刻）
   Step8  延迟分解 L_proc/L_track/L_lat     （§6.2，Fig.8 复刻）
   Step9  延迟 vs κ + 点/秒 vs ν            （§6.4，Fig.11 / Fig.13 复刻）
   Step10 拥塞闭环 pacing→ν→点/秒           （§6.4，Fig.14 复刻，合成 pacing）

   运行方式（PyCharm 里右键 Run）：
   解释器用 D:\Users\admuj\anaconda3\envs\opencv_env\python.exe

 也支持命令行（不用窗口，适合被别的脚本调）：
   python PTV_启动器.py --cli step0 --dataset sim:3
   python PTV_启动器.py --cli step1 --data-dir "py_repro\data\sim3p"
   python PTV_启动器.py --cli chain --dataset sim:3     # 从Step0一路跑到Step5
============================================================
"""
import argparse
import os
import sys

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
import step6_streaming as s6                 # noqa: E402
import step7_experiments as s7               # noqa: E402
import step8_latency as s8                  # noqa: E402
import step9_latency_kappa as s9            # noqa: E402
import step10_congestion as s10             # noqa: E402

STEPS = [
    ("Step0  解析与翻译数据集 → npz+meta", "step0"),
    ("Step1  事件去噪（热像素+不应期）", "step1"),
    ("Step2  双目匹配 式(2) + 三角测量", "step2"),
    ("Step3  3D 点云聚类（1cm）", "step3"),
    ("Step4  逐帧跟踪（GNN+卡尔曼）", "step4"),
    ("Step5  精度评估 Cov/Prec/Chamfer", "step5"),
    ("Step6  流式实验 κ / ν（Fig.12）", "step6"),
    ("Step7  粒子数扫描（Fig.6）", "step7"),
    ("Step8  延迟分解 L_proc/L_track（Fig.8）", "step8"),
    ("Step9  延迟 vs κ（Fig.11）+ 点/秒 vs ν（Fig.13）", "step9"),
    ("Step10 拥塞闭环 pacing→ν（Fig.14）", "step10"),
]
STEP_KEYS = [k for _, k in STEPS]


# ============================================================
#  一、数据集侦察兵
# ============================================================

def scan_datasets():
    """扫描 py_repro/data 下所有"四件套齐全"的数据集 -> [(显示名, 路径)]"""
    out = []
    if not os.path.isdir(K.DATA_DIR):
        return out
    for name in sorted(os.listdir(K.DATA_DIR)):
        p = os.path.join(K.DATA_DIR, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "meta.json")):
            meta = K.load_meta(p)
            tag = f"{name}  [{meta.get('n_left', '?')}/{meta.get('n_right', '?')} 事件, {meta.get('n_gt', 0)} 真值]"
            # 真实数据集标记：单目（Step2 起无匹配）/ 伪双目（3D 无物理意义）
            flags = []
            if meta.get("mono"):
                flags.append("单目")
            if (meta.get("pseudo_stereo") or {}).get("enabled"):
                flags.append("伪双目·3D无物理意义")
            if meta.get("kappa_applied"):
                flags.append(f"κ={meta['kappa_applied']}")
            if flags:
                tag += "  ⚠ " + " / ".join(flags)
            out.append((tag, p))
    return out


# ============================================================
#  二、执行引擎（GUI 与 CLI 共用）
# ============================================================

def run_one(step_key, data_dir=None, dataset=None, n_particles=3, out_dir=None):
    """跑一步。data_dir 为 None 时用 dataset 新建（只对 step0 有意义）。"""
    if step_key == "step0":
        src = dataset or f"sim:{n_particles}"
        return s0.run_step(src, out_dir=data_dir)
    # Step6/7 是独立实验，不依赖数据集目录
    if step_key == "step6":
        return s6.run_step(out_dir)
    if step_key == "step7":
        return s7.run_step(out_dir)
    if step_key == "step8":
        return s8.run_step(out_dir)
    if step_key == "step9":
        return s9.run_step(out_dir)
    if step_key == "step10":
        return s10.run_step(out_dir)
    if not data_dir:
        raise SystemExit("[×] 这一步需要一个数据集目录（先跑 Step0）")
    if step_key == "step1":
        return s1.run_step(data_dir)
    if step_key == "step2":
        return s2.run_step(data_dir)
    if step_key == "step3":
        return s3.run_step(data_dir)
    if step_key == "step4":
        return s4.run_step(data_dir)
    if step_key == "step5":
        return s5.run_step(data_dir)
    raise SystemExit(f"[×] 不认识的步骤：{step_key}")


def run_chain(data_dir=None, dataset="sim:3", out_dir=None):
    """从 Step0 一路跑到 Step5（数据集链）。Step6/7 是独立实验，不进链。"""
    d = run_one("step0", data_dir=data_dir, dataset=dataset)
    for k in ("step1", "step2", "step3", "step4", "step5"):
        run_one(k, data_dir=d)
    print("\n" + "=" * 72)
    print(f"  全链路完成，产物都在：{d}")
    print("=" * 72)
    return d


# ============================================================
#  三、GUI
# ============================================================

def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("粒子追踪复现 · 统一启动器")
    root.geometry("980x700")
    root.attributes("-topmost", False)

    # ---------- 顶部：数据集 ----------
    frm = ttk.LabelFrame(root, text="① 数据集（Step0 的产物）")
    frm.pack(fill="x", padx=10, pady=(10, 4))

    ds_var = tk.StringVar()
    ds_box = ttk.Combobox(frm, textvariable=ds_var, state="readonly", width=72)
    ds_box.pack(side="left", padx=6, pady=6)

    def refresh_ds(select=None):
        items = scan_datasets()
        ds_box["values"] = [t for t, _ in items]
        _cache["items"] = items
        if items:
            idx = 0
            if select:
                for i, (_, p) in enumerate(items):
                    if os.path.normcase(p) == os.path.normcase(select):
                        idx = i
                        break
            ds_box.current(idx)

    _cache = {"items": []}

    def browse_ds():
        p = filedialog.askdirectory(title="选一个数据集文件夹", initialdir=K.DATA_DIR)
        if p:
            items = list(_cache["items"])
            items.append((os.path.basename(p) + "  [外部]", p))
            ds_box["values"] = [t for t, _ in items]
            _cache["items"] = items
            ds_box.current(len(items) - 1)

    ttk.Button(frm, text="浏览…", command=browse_ds).pack(side="left", padx=4)
    ttk.Button(frm, text="刷新", command=refresh_ds).pack(side="left", padx=4)

    # ---------- 中部：步骤 ----------
    frm2 = ttk.LabelFrame(root, text="② 跑哪一步")
    frm2.pack(fill="x", padx=10, pady=4)

    step_var = tk.StringVar(value=STEPS[0][0])
    ttk.Combobox(frm2, textvariable=step_var, state="readonly", width=52,
                 values=[t for t, _ in STEPS]).pack(side="left", padx=6, pady=6)

    ttk.Label(frm2, text="Step0 新仿真粒子数：").pack(side="left", padx=(16, 2))
    np_var = tk.StringVar(value="3")
    ttk.Entry(frm2, textvariable=np_var, width=6).pack(side="left")

    # ---------- 输出目录 ----------
    frm3 = ttk.LabelFrame(root, text="③ Step6/7 的输出文件夹（其它步骤直接写在数据集目录里）")
    frm3.pack(fill="x", padx=10, pady=4)

    out_var = tk.StringVar(value=os.path.join(HERE, "experiments"))
    ttk.Entry(frm3, textvariable=out_var, width=70).pack(side="left", padx=6, pady=6)

    def browse_out():
        p = filedialog.askdirectory(title="选输出文件夹", initialdir=HERE)
        if p:
            out_var.set(p)

    ttk.Button(frm3, text="浏览…", command=browse_out).pack(side="left")
    ttk.Button(frm3, text="打开", command=lambda: os.startfile(out_var.get())
               if os.path.isdir(out_var.get()) else None).pack(side="left", padx=4)

    # ---------- 按钮 ----------
    frm4 = ttk.Frame(root)
    frm4.pack(fill="x", padx=10, pady=6)

    log = tk.Text(root, height=24, wrap="word", font=("Consolas", 9))
    log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    class _Tee:
        """把 print 同时送到 Text 控件（GUI 里看得到进度）。"""
        def write(self, s):
            log.insert("end", s)
            log.see("end")
            root.update_idletasks()

        def flush(self):
            pass

    def do_run(chain=False):
        items = _cache["items"]
        sel = ds_box.current()
        data_dir = items[sel][1] if (items and sel >= 0) else None
        key = step_var.get().split()[0].lower()
        log.delete("1.0", "end")
        old = sys.stdout
        sys.stdout = _Tee()
        try:
            if chain:
                run_chain(data_dir=data_dir, dataset=f"sim:{np_var.get()}",
                          out_dir=out_var.get())
                refresh_ds()
            else:
                out = run_one(key, data_dir=data_dir,
                              dataset=f"sim:{np_var.get()}", out_dir=out_var.get())
                if key == "step0":
                    refresh_ds(select=out)
        except SystemExit as e:
            print(e)
        except Exception as e:
            print(f"[×] 出错：{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = old

    ttk.Button(frm4, text="▶  开始处理", command=do_run).pack(side="left", padx=6)
    ttk.Button(frm4, text="▶▶ 从 Step0 一路跑到 Step5", command=lambda: do_run(True)).pack(side="left", padx=6)
    ttk.Button(frm4, text="清空日志", command=lambda: log.delete("1.0", "end")).pack(side="left", padx=6)

    refresh_ds()
    print(f"数据大本营：{K.DATA_DIR}")
    print(f"算法库目录：{K.CORE_DIR}")
    print("先跑 Step0 造数据集，再往后走。")
    root.mainloop()


# ============================================================
#  四、入口
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="粒子追踪复现统一启动器")
    ap.add_argument("--cli", choices=["step0", "step1", "step2", "step3", "step4",
                                      "step5", "step6", "step7", "step8", "step9", "step10",
                                      "chain"], default=None,
                    help="命令行模式：指定要跑的步骤")
    ap.add_argument("--dataset", default=None, help="Step0 数据源：sim:N 或文件夹")
    ap.add_argument("--data-dir", default=None, help="已翻译好的数据集目录")
    ap.add_argument("--n-particles", type=int, default=3)
    ap.add_argument("--out", default=None, help="Step6/7 输出目录")
    args = ap.parse_args()

    if args.cli is None:
        launch_gui()
        return

    if args.cli == "chain":
        run_chain(data_dir=args.data_dir, dataset=args.dataset or f"sim:{args.n_particles}",
                  out_dir=args.out)
    else:
        run_one(args.cli, data_dir=args.data_dir, dataset=args.dataset,
                n_particles=args.n_particles, out_dir=args.out)


if __name__ == "__main__":
    main()
