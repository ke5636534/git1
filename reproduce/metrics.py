# -*- coding: utf-8 -*-
"""精度指标（对应论文6.1节，式(3)覆盖率Cov、式(4)精确率Prec、式(7)倒角距离Chamfer）。
O=真值点集，R=重建点集，半径ρ=5cm。
"""
import numpy as np
import config as C


def _nn_dist(a, b):
    """a中每个点到b的最近邻距离。"""
    if len(a) == 0 or len(b) == 0:
        return np.array([np.inf] * len(a))
    out = []
    for p in a:
        out.append(np.min(np.linalg.norm(b - p, axis=1)))
    return np.array(out)


def evaluate(recon, gt, rho=C.RHO):
    """O = 真值点集 gt，R = 重建点集 recon，容差球半径 ρ = rho。

    式(5) d_NN(O,R) = 1/|O|·Σ_{o∈O} NN(o,R)  -> d_nn_gt2recon   （真值点 -> 重建点集）
    式(6) d_NN(R,O) = 1/|R|·Σ_{r∈R} NN(r,O)  -> d_nn_recon2gt   （重建点 -> 真值点集）
    式(7) d_C(O,R)  = ½·d_NN(O,R) + ½·d_NN(R,O) -> chamfer      （双向，缺一不可）
    单位：米。点集为空时对应量取 0.0（与覆盖率/精确率的空集兜底保持一致）。
    """
    d_o = _nn_dist(gt, recon)     # 真值 -> 重建：逐点 NN(o,R)
    d_r = _nn_dist(recon, gt)     # 重建 -> 真值：逐点 NN(r,O)
    cov = float(np.mean(d_o <= rho)) if len(gt) else 0.0
    prec = float(np.mean(d_r <= rho)) if len(recon) else 0.0
    mo, mr = d_o[np.isfinite(d_o)], d_r[np.isfinite(d_r)]
    # 式(5)：真值点到重建点集的平均最近邻距离（m）
    d_nn_gt2recon = float(mo.mean()) if len(mo) else 0.0
    # 式(6)：重建点到真值点集的平均最近邻距离（m）
    d_nn_recon2gt = float(mr.mean()) if len(mr) else 0.0
    # 式(7)：倒角距离 = ½·式(5) + ½·式(6)
    # 空集语义（审查发现G1）：重建集为空时应为 inf（"什么都没重建出来"），
    # 返回 0 会伪装成"完美重建"。单侧空时该侧 NN 全为 inf，自然不会是 0。
    chamfer = float("inf") if (len(gt) == 0 or len(recon) == 0) \
        else float(0.5 * (d_nn_gt2recon + d_nn_recon2gt))
    return {"coverage": cov, "precision": prec, "chamfer": chamfer,
            "d_nn_gt2recon": d_nn_gt2recon, "d_nn_recon2gt": d_nn_recon2gt,
            "n_recon": len(recon), "n_gt": len(gt)}


if __name__ == "__main__":
    import os
    import gen_events

    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(HERE, "output")
    os.makedirs(OUT, exist_ok=True)

    src = os.path.join(OUT, "track_points.npy")
    if not os.path.exists(src):
        raise SystemExit("缺少 output/track_points.npy，请先运行 track.py")
    recon = np.load(src).reshape(-1, 3)

    # 真值：与 output/match_points.npy 同源（gen_events.generate(1, 0)，SEED 固定）
    gt = gen_events.generate(1, 0)[2][:, 1:]

    m = evaluate(recon, gt)

    with open(os.path.join(OUT, "metrics.csv"), "w", encoding="utf-8-sig") as f:
        f.write("metric,value\n")
        f.write(f"coverage,{m['coverage']:.6f}\n")
        f.write(f"precision,{m['precision']:.6f}\n")
        f.write(f"chamfer_m,{m['chamfer']:.6f}\n")
        f.write(f"n_recon,{m['n_recon']}\n")
        f.write(f"n_gt,{m['n_gt']}\n")

    print(f"重建点数 {m['n_recon']} / 真值点数 {m['n_gt']}  评估半径 rho = {C.RHO * 100:.0f} cm")
    print(f"Coverage (Cov)    = {m['coverage']:.4f}")
    print(f"Precision (Prec)  = {m['precision']:.4f}")
    print(f"Chamfer           = {m['chamfer']:.6f} m ({m['chamfer'] * 1000:.2f} mm)")
    print("已保存: output/metrics.csv")
