"""モデルの点を、本当の内部パラメータで引き直す（再推論なし）。

モデルの点（カメラ座標）から、モデルが使った内部パラメータを当てる（fit_K）。各画素の奥行き z はモデルのまま、
画素から点への向きだけを本当の内部パラメータ（公開データは正解の点の格子に当てた値。データセットの K は元の画素のままなので使えない）で引き直す。
焦点距離の見積もりが 3% ずれていれば、画像の面に沿った長さ（扉の幅・高さ）も 3% ずれる、という仮説を確かめる。
§22 の 8 行目（内部パラメータをモデルに渡す、再推論）とは別。こちらはモデルの奥行きを変えない。

使い方（server フォルダで）: uv run python ../experiments/focal_reproject.py [public | posed]
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


def reproject(P, Km, Kg):
    z = P[..., 2:3]
    u = Km[0, 0] * P[..., 0:1] / z + Km[0, 2]
    v = Km[1, 1] * P[..., 1:2] / z + Km[1, 2]
    return np.concatenate([z * (u - Kg[0, 2]) / Kg[0, 0], z * (v - Kg[1, 2]) / Kg[1, 1], z], -1).astype(P.dtype)


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "public"
    idx = json.loads((BENCH / "cases" / "index.json").read_text(encoding="utf-8"))
    rows = []
    for name, ds in idx[group].items():
        case = load_case(name)
        f = []
        new = []
        for v in range(case["n"]):
            Km = fit_K(case["ma"][v].astype(np.float64), case["mask"][v])
            Kg = fit_K(case["gt"][v].astype(np.float64), case["mask"][v] & np.isfinite(case["gt"][v]).all(-1))
            f.append(Km[0, 0] / Kg[0, 0])
            new.append(reproject(case["ma"][v], Km, Kg))
        m0 = public_metrics(case, 1.0)
        m1 = public_metrics({**case, "ma": new}, 1.0)
        rows.append({"case": name, "ds": ds, "f_ratio_median": float(np.median(f)), **{f"none {a}": b for a, b in m0.items()}, **{f"repr {a}": b for a, b in m1.items()}})
        print(f"{name:22s} 焦点距離 モデル÷本当 {np.median(f):.3f}（{min(f):.3f}〜{max(f):.3f}）｜ 大きさ {m0['size_err_pct']:+.1f}% → {m1['size_err_pct']:+.1f}% ｜ "
              + "  ".join(f"{b} {m0[b]:.1f}→{m1[b]:.1f}" for b, _, _ in BANDS), flush=True)
    for ds in ("tum", "arkit"):
        sub = [r for r in rows if r["ds"] == ds]
        if sub:
            print(f"{ds}（{len(sub)} 本）: 大きさ|.| {np.mean([abs(r['none size_err_pct']) for r in sub]):.1f}% → {np.mean([abs(r['repr size_err_pct']) for r in sub]):.1f}% ｜ "
                  + "  ".join(f"{b} {np.nanmean([r[f'none {b}'] for r in sub]):.1f}→{np.nanmean([r[f'repr {b}'] for r in sub]):.1f}" for b, _, _ in BANDS))
    (BENCH / f"focal_reproject_{group}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
