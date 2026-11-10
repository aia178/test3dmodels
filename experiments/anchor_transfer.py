"""「近くで撮ったスイッチの枠から、壁の 1 点までの実寸の距離が分かる」場合の効き目を、公開データで先に確かめる（再推論なし）。

近くで撮るコマの代わりに、奥行きの中央値が最も小さいコマ k を使う。k の画像の中央付近の数点について、正解の奥行きに誤差（標準偏差 NOISE）を
足した実寸の 3D 点 P を作る（枠の画素の大きさから距離を出すときの誤差の想定）。
P を正解のカメラの位置で、ほかのコマ v に写す。v で P が見えていれば（正解の奥行きと 5% 以内で一致）、
「モデルの v のカメラから、その画素の点までの距離 ÷ 実寸の距離」を集める。その中央値で全体を割る。
k 自身のモデルの奥行きは使わない（コマごとの大きさは 3〜6% ばらつくため。one_view_anchor.py）。

使い方（server フォルダで）: uv run python ../experiments/anchor_transfer.py [public | posed]
"""
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent / "server"))
from plate_ruler import fit_K  # noqa: E402
from scale_bench import BANDS, BENCH, load_case, public_metrics  # noqa: E402

NOISE = 0.01
ANCHOR_OFFSETS = [(dx, dy) for dx in (-0.15, 0, 0.15) for dy in (-0.15, 0, 0.15)]   # 画像の幅・高さに対する割合


def anchor_ratios(case, rng):
    depth = [np.nanmedian(case["gt"][v][..., 2]) for v in range(case["n"])]
    k = int(np.nanargmin(depth))
    G = case["gt_c2w"]
    Ks = [fit_K(case["gt"][v].astype(np.float64), np.isfinite(case["gt"][v]).all(-1)) for v in range(case["n"])]
    H, W = case["mask"][k].shape
    out = []
    for dx, dy in ANCHOR_OFFSETS:
        i, j = int(H * (0.5 + dy)), int(W * (0.5 + dx))
        q = case["gt"][k][i, j].astype(np.float64)
        if not np.isfinite(q).all():
            continue
        P = G[k][:3, :3] @ (q * (1 + rng.normal(0, NOISE))) + G[k][:3, 3]
        for v in range(case["n"]):
            if v == k:
                continue
            pc = G[v][:3, :3].T @ (P - G[v][:3, 3])
            if pc[2] < 0.2:
                continue
            u, w = Ks[v][0, 0] * pc[0] / pc[2] + Ks[v][0, 2], Ks[v][1, 1] * pc[1] / pc[2] + Ks[v][1, 2]
            a, b = int(w), int(u)
            if not (0 <= a < H and 0 <= b < W) or not case["mask"][v][a, b]:
                continue
            g = case["gt"][v][a, b]
            if not np.isfinite(g).all() or abs(g[2] / pc[2] - 1) > 0.05:
                continue
            m = case["ma"][v][a, b].astype(np.float64)
            if np.isfinite(m).all():
                out.append(np.linalg.norm(m) / np.linalg.norm(pc))
    return k, depth[k], np.array(out)


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "posed"
    idx = json.loads((BENCH / "cases" / "index.json").read_text(encoding="utf-8"))
    rows = []
    for name, ds in idx[group].items():
        case = load_case(name)
        k, dk, r = anchor_ratios(case, np.random.default_rng(0))
        if len(r) < 3:
            print(f"{name:22s} 近いコマ {k}（{dk:.1f}m）の点が、ほかのコマで {len(r)} 回しか見えない", flush=True)
            continue
        s = 1 / float(np.median(r))
        m0, m1 = public_metrics(case, 1.0), public_metrics(case, s)
        rows.append({"case": name, "ds": ds, "near_view": k, "n": int(len(r)), "s": s, **{f"none {a}": b for a, b in m0.items()}, **{f"corr {a}": b for a, b in m1.items()}})
        print(f"{name:22s} 近いコマ {k}（{dk:.1f}m）→ 遠いコマで {len(r)} 回 ｜ 倍率 {s:.3f} ｜ 大きさ {m0['size_err_pct']:+.1f}% → {m1['size_err_pct']:+.1f}% ｜ "
              + "  ".join(f"{b} {m0[b]:.1f}→{m1[b]:.1f}" for b, _, _ in BANDS), flush=True)
    for ds in ("tum", "arkit"):
        sub = [r for r in rows if r["ds"] == ds]
        if sub:
            print(f"{ds}（{len(sub)} 本）: 大きさ|.| {np.mean([abs(r['none size_err_pct']) for r in sub]):.1f}% → {np.mean([abs(r['corr size_err_pct']) for r in sub]):.1f}% ｜ "
                  + "  ".join(f"{b} {np.nanmean([r[f'none {b}'] for r in sub]):.1f}→{np.nanmean([r[f'corr {b}'] for r in sub]):.1f}" for b, _, _ in BANDS))
    (BENCH / f"anchor_transfer_{group}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
