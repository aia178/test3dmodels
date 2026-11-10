"""評価（TUM RGB-D）: 「物の大きさ」の縮尺を、深度センサーの実測と比べる。

tum_eval.py は「カメラからの距離」を比べたが、このモデルはカメラの位置を壁に近く見積もる癖がある（検証記録 §10）ので、
距離の比は物の大きさの比と同じにならない。ここでは、同じ写真の中の 2 つの画素の間の 3D 距離を、
復元（world 座標の点どうし）と実測（深度とカメラの内部パラメータから作った点どうし）で比べる。
使い方（server フォルダで）: uv run python ../experiments/tum_size_eval.py <データのフォルダ> <tum_eval の出力フォルダ>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
sys.path.insert(0, str(here))
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402
from tum_eval import associate, model_grid_to_source, pick_evenly, read_list, sample_nearest  # noqa: E402

FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3   # freiburg1 のカラーカメラ（データセットの公開値）
src, out = Path(sys.argv[1]), Path(sys.argv[2])
raw = load_raw(out / "raw.npz")
n = len(raw["cam2world"])
args = json.loads((out / "args.json").read_text()) if (out / "args.json").exists() else None
t0, t1 = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.0, None)
pairs = pick_evenly(associate(read_list(src / "rgb.txt"), read_list(src / "depth.txt")), n, t0, t1)
rng = np.random.default_rng(0)
per_view, all_r = [], []
for v, ((mask, xyz), (_, _, dname)) in enumerate(zip(view_grids(raw), pairs)):
    gt = np.asarray(Image.open(src / dname)).astype(np.float64) / 5000.0
    xs, ys = model_grid_to_source(xyz.shape[0], xyz.shape[1], gt.shape[0], gt.shape[1])
    z = sample_nearest(gt, xs, ys)
    Q = np.stack([(xs - CX) / FX * z, (ys - CY) / FY * z, z], -1)      # 実測の 3D 点（カメラ座標）
    ok = mask & (z > 0.3) & (z < 5.0) & np.isfinite(xyz).all(-1)
    idx = np.argwhere(ok)
    if len(idx) < 5000:
        continue
    a, b = idx[rng.integers(0, len(idx), 40000)], idx[rng.integers(0, len(idx), 40000)]
    dp = np.linalg.norm(xyz[a[:, 0], a[:, 1]].astype(np.float64) - xyz[b[:, 0], b[:, 1]], axis=1)
    dq = np.linalg.norm(Q[a[:, 0], a[:, 1]] - Q[b[:, 0], b[:, 1]], axis=1)
    far = dq > 0.5                                                   # 50cm 以上離れた組だけ（近い組は雑音が効く）
    r = dp[far] / dq[far]
    per_view.append(float(np.median(r)))
    all_r.append(r)
r = np.concatenate(all_r)
g = float(np.median(r))
pv = np.array(per_view)
res = {"n_views": len(pv), "size_scale_pred_over_true": round(g, 4), "size_scale_error_pct": round((g - 1) * 100, 1),
       "per_view_min_max": [round(float(pv.min()), 4), round(float(pv.max()), 4)], "per_view_spread_pct": round(float((pv.max() - pv.min()) / pv.mean() * 100), 1),
       "pair_error_after_global_scale_median_pct": round(float(np.median(np.abs(r / g - 1))) * 100, 2)}
(out / "tum_size_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(out.name, res)
print("  写真ごとの値:", [round(x, 3) for x in per_view])
