"""評価（TUM RGB-D）: 写真をまたぐ長さ（部屋の中の離れた 2 点の距離）を、実測と比べる。

実測側: 深度センサーの値とカメラの内部パラメータで作った点を、モーションキャプチャで測ったカメラの位置・向き（groundtruth.txt）で
       部屋の座標へ移す。復元側: world 座標の点。別々の写真に写った 2 点の距離を、両方で出して比べる。
使い方（server フォルダで）: uv run python ../experiments/tum_cross_eval.py <データのフォルダ> <tum_eval の出力フォルダ> [開始秒 終了秒]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
sys.path.insert(0, str(here))
from roomfit.arcore_import import quat_to_rot  # noqa: E402
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402
from tum_eval import associate, model_grid_to_source, pick_evenly, read_list, sample_nearest  # noqa: E402

FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3
src, out = Path(sys.argv[1]), Path(sys.argv[2])
t0, t1 = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.0, None)
raw = load_raw(out / "raw.npz")
n = len(raw["cam2world"])
pairs = pick_evenly(associate(read_list(src / "rgb.txt"), read_list(src / "depth.txt")), n, t0, t1)
gt = np.array([[float(x) for x in line.split()] for line in (src / "groundtruth.txt").read_text().splitlines() if not line.startswith("#")])
rng = np.random.default_rng(0)
P_all, Q_all, V_all, dt_max = [], [], [], 0.0
for v, ((mask, xyz), (t, _, dname)) in enumerate(zip(view_grids(raw), pairs)):
    k = int(np.argmin(np.abs(gt[:, 0] - t)))
    dt_max = max(dt_max, abs(gt[k, 0] - t))
    R, tr = quat_to_rot(gt[k, 4:8]), gt[k, 1:4]
    d = np.asarray(Image.open(src / dname)).astype(np.float64) / 5000.0
    xs, ys = model_grid_to_source(xyz.shape[0], xyz.shape[1], d.shape[0], d.shape[1])
    z = sample_nearest(d, xs, ys)
    Qc = np.stack([(xs - CX) / FX * z, (ys - CY) / FY * z, z], -1)
    ok = mask & (z > 0.3) & (z < 4.0) & np.isfinite(xyz).all(-1)
    idx = np.argwhere(ok)
    sel = idx[rng.integers(0, len(idx), 3000)]
    P_all.append(xyz[sel[:, 0], sel[:, 1]].astype(np.float64))
    Q_all.append(Qc[sel[:, 0], sel[:, 1]] @ R.T + tr)
    V_all.append(np.full(len(sel), v))
P, Q, V = np.concatenate(P_all), np.concatenate(Q_all), np.concatenate(V_all)
i, j = rng.integers(0, len(P), 400000), rng.integers(0, len(P), 400000)
dp, dq = np.linalg.norm(P[i] - P[j], axis=1), np.linalg.norm(Q[i] - Q[j], axis=1)
same = V[i] == V[j]
res = {"n_views": n, "pose_time_gap_max_ms": round(dt_max * 1000, 1), "room_extent_true_m": [round(float(x), 2) for x in (np.percentile(Q, 99, axis=0) - np.percentile(Q, 1, axis=0))]}


def stat(m, name):
    if m.sum() < 500:
        return
    r = dp[m] / dq[m]
    g = float(np.median(r))
    res[name] = {"pairs": int(m.sum()), "scale_pred_over_true": round(g, 4), "error_pct": round((g - 1) * 100, 1),
                 "p10_p90_pct": [round(float(np.percentile(r, 10) - 1) * 100, 1), round(float(np.percentile(r, 90) - 1) * 100, 1)],
                 "abs_error_cm_median": round(float(np.median(np.abs(dp[m] - dq[m]))) * 100, 1)}


stat(same & (dq > 0.5), "同じ写真の中の 2 点（50cm 以上）")
stat(~same & (dq > 0.5) & (dq <= 1.5), "別の写真の 2 点（0.5〜1.5m）")
stat(~same & (dq > 1.5) & (dq <= 3.0), "別の写真の 2 点（1.5〜3m）")
stat(~same & (dq > 3.0), "別の写真の 2 点（3m 以上＝部屋の端から端）")
(out / "tum_cross_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(out.name)
for k, val in res.items():
    print("  ", k, val)
