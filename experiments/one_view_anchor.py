"""1 つのコマだけで実寸を知ったとき（近くで撮ったスイッチの枠など）、その倍率が復元全体にどこまで通用するかを公開データで見る（再推論なし）。

写真ごとの「モデルの大きさ ÷ 正解」（写真の中の 2 点の距離の比の中央値）を r_v とする。
コマ k で実寸を知って全体を 1 / r_k 倍すると、全体の大きさのずれは r_全体 / r_k − 1 になる（r_全体 は全コマの中央値）。
k を全コマで変えて、このずれの絶対値の平均・最大と、写真をまたぐ距離の誤差（scale_bench.py と同じ）を、何もしない場合と比べる。
「近いコマ」で決めた場合も見る（近くで撮るコマは、ほかより手前を写すので、奥行きの中央値が最も小さいコマで代用）。

使い方（server フォルダで）: uv run python ../experiments/one_view_anchor.py [public（写真だけ、既定）| posed（正解の位置つき）]
"""
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
from scale_bench import BANDS, BENCH, inview_ratio, load_case, public_metrics  # noqa: E402


def main():
    idx = json.loads((BENCH / "cases" / "index.json").read_text(encoding="utf-8"))
    rows = []
    group = sys.argv[1] if len(sys.argv) > 1 else "public"
    for name, ds in idx[group].items():
        case = load_case(name)
        rng = np.random.default_rng(0)
        r, depth = [], []
        for v in range(case["n"]):
            ok = case["mask"][v] & np.isfinite(case["gt"][v]).all(-1) & np.isfinite(case["ma"][v]).all(-1)
            x = inview_ratio(case["ma"][v], case["gt"][v], ok, rng, min_d=0.5)
            r.append(np.nan if x is None else x)
            depth.append(float(np.median(case["gt"][v][ok][:, 2])) if ok.any() else np.nan)
        r, depth = np.array(r), np.array(depth)
        good = np.isfinite(r)
        r_all = float(np.median(r[good]))
        errs = r_all / r[good] - 1
        near = int(np.nanargmin(np.where(good, depth, np.nan)))
        base = public_metrics(case, 1.0)
        per_k = [public_metrics(case, 1 / r[k]) for k in np.flatnonzero(good)]
        nearm = public_metrics(case, 1 / r[near])
        row = {"case": name, "ds": ds, "views": int(good.sum()), "size_none": base["size_err_pct"],
               "anchor_abs_mean": float(np.mean(np.abs(errs)) * 100), "anchor_abs_max": float(np.max(np.abs(errs)) * 100),
               "near_size": nearm["size_err_pct"], "near_depth": depth[near],
               **{f"none {b}": base[b] for b, _, _ in BANDS}, **{f"anchor {b}": float(np.nanmean([m[b] for m in per_k])) for b, _, _ in BANDS},
               **{f"near {b}": nearm[b] for b, _, _ in BANDS}}
        rows.append(row)
        print(f"{name:14s} 写真 {row['views']:2d} ｜ 大きさ 何もしない {row['size_none']:+5.1f}% ／ 1 コマで合わせる |ずれ| 平均 {row['anchor_abs_mean']:.1f}% 最大 {row['anchor_abs_max']:.1f}%"
              f" ／ 最も近いコマ（{row['near_depth']:.1f}m）で {row['near_size']:+5.1f}% ｜ >3m {row['none >3m']:.1f} → 平均 {row['anchor >3m']:.1f}・近 {row['near >3m']:.1f}", flush=True)
    for ds in ("tum", "arkit"):
        sub = [r for r in rows if r["ds"] == ds]
        print(f"{ds}: 大きさ|.| 何もしない {np.mean([abs(r['size_none']) for r in sub]):.1f}% ／ 1 コマで合わせる 平均 {np.mean([r['anchor_abs_mean'] for r in sub]):.1f}%（最悪のコマで {np.mean([r['anchor_abs_max'] for r in sub]):.1f}%）"
              f" ／ 最も近いコマで {np.mean([abs(r['near_size']) for r in sub]):.1f}% ｜ "
              + "  ".join(f"{b} {np.nanmean([r[f'none {b}'] for r in sub]):.1f}→{np.nanmean([r[f'anchor {b}'] for r in sub]):.1f}・近 {np.nanmean([r[f'near {b}'] for r in sub]):.1f}" for b, _, _ in BANDS))
    (BENCH / f"one_view_anchor_{group}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
