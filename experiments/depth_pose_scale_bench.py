"""depth_pose_scale.py の直し方（カメラの位置を固定し、写真どうしが一番よく重なる奥行きの倍率を選ぶ）を、公開データで確かめる。

AR の位置の代わりに、データセットの正解のカメラの位置と向きを使う。奥行きは保存済みの MapAnything の点（写真だけの復元、カメラ座標）。
比べるもの:
- 大きさ: 写真の中の距離の、正解に対する比（scale_bench.py の size_err_pct と同じ）。倍率 1 のままと、選んだ倍率で。
- 写真をまたぐ距離の誤差（中央値・cm）。点を「正解のカメラ位置 + 倍率 × 正解の向き × モデルの点」に置いて、正解の点と比べる。
使い方（server フォルダで）: uv run python ../experiments/depth_pose_scale_bench.py [ケース名 ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
from depth_pose_scale import overlap_cost  # noqa: E402
from scale_bench import BANDS, BENCH, inview_ratio, load_case  # noqa: E402

S_GRID = np.round(np.arange(0.80, 1.201, 0.01), 3)


def sample(case, rng, n):
    out = []
    for v in range(case["n"]):
        P, Q = case["ma"][v], case["gt"][v]
        ok = case["mask"][v] & np.isfinite(P).all(-1) & np.isfinite(Q).all(-1) & (P[..., 2] > 0.05)
        idx = np.argwhere(ok)
        sel = idx[rng.choice(len(idx), min(n, len(idx)), replace=False)] if len(idx) else idx
        out.append((P[sel[:, 0], sel[:, 1]].astype(np.float64), Q[sel[:, 0], sel[:, 1]].astype(np.float64)))
    return out


def cross_errors(case, pts, s, rng):
    T = case["gt_c2w"]
    P = np.concatenate([T[v][:3, 3] + s * p @ T[v][:3, :3].T for v, (p, _) in enumerate(pts)])
    Q = np.concatenate([T[v][:3, 3] + q @ T[v][:3, :3].T for v, (_, q) in enumerate(pts)])
    V = np.concatenate([np.full(len(p), v) for v, (p, _) in enumerate(pts)])
    i, j = rng.integers(0, len(P), 400000), rng.integers(0, len(P), 400000)
    dp, dq = np.linalg.norm(P[i] - P[j], axis=1), np.linalg.norm(Q[i] - Q[j], axis=1)
    out = {}
    for name, lo, hi in BANDS:
        m = (V[i] != V[j]) & (dq > lo) & (dq <= hi)
        out[name] = float(np.median(np.abs(dp[m] - dq[m]))) * 100 if m.sum() >= 500 else np.nan
    return out


def main():
    idx = json.loads((BENCH / "cases" / "index.json").read_text(encoding="utf-8"))
    names = sys.argv[1:] or list(idx["public"])
    rows = []
    for name in names:
        case = load_case(name)
        rng = np.random.default_rng(0)
        g = [inview_ratio(case["ma"][v], case["gt"][v], case["mask"][v] & np.isfinite(case["gt"][v]).all(-1), rng, min_d=0.5) for v in range(case["n"])]
        size = float(np.median([x for x in g if x is not None]))   # モデルの大きさ ÷ 正解
        cost_pts = sample(case, np.random.default_rng(1), 2000)
        Rs, Cs = [T[:3, :3] for T in case["gt_c2w"]], [T[:3, 3] for T in case["gt_c2w"]]
        costs = [overlap_cost(Rs, Cs, [p for p, _ in cost_pts], s)[0] for s in S_GRID]
        s_best = float(S_GRID[int(np.argmin(costs))])
        pts = sample(case, np.random.default_rng(2), 3000)
        e1, eb = cross_errors(case, pts, 1.0, np.random.default_rng(3)), cross_errors(case, pts, s_best, np.random.default_rng(3))
        row = {"case": name, "s_best": s_best, "s_true": round(1 / size, 3), "size_err_s1": round((size - 1) * 100, 2), "size_err_best": round((s_best * size - 1) * 100, 2),
               **{f"s1 {b}": round(e1[b], 1) for b, _, _ in BANDS}, **{f"best {b}": round(eb[b], 1) for b, _, _ in BANDS}}
        rows.append(row)
        print(f"{name:14s} 倍率 選んだ {s_best:.2f} / 正しい {1 / size:.3f} ｜ 大きさ {row['size_err_s1']:+.1f}% → {row['size_err_best']:+.1f}% ｜ "
              + "  ".join(f"{b} {e1[b]:.1f}→{eb[b]:.1f}" for b, _, _ in BANDS), flush=True)
    for ds in ("tum", "arkit"):
        sub = [r for r in rows if idx["public"][r["case"]] == ds]
        if sub:
            print(f"{ds} 平均: 大きさ|.| {np.mean([abs(r['size_err_s1']) for r in sub]):.1f}% → {np.mean([abs(r['size_err_best']) for r in sub]):.1f}% ｜ "
                  + "  ".join(f"{b} {np.nanmean([r[f's1 {b}'] for r in sub]):.1f}→{np.nanmean([r[f'best {b}'] for r in sub]):.1f}" for b, _, _ in BANDS))
    (BENCH / "depth_pose_scale.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
