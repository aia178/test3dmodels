"""復元の形のゆがみを、実測なしで確かめる指標を出す。
- 2 つの壁（クローゼットの面と引き戸の壁）のなす角: 実際は 90° のはず
- それぞれの壁と床のなす角: 90° のはず
- 推定された焦点距離（モデル画像の画素）
使い方（server フォルダで）: uv run python ../experiments/shape_checks.py <出力フォルダ>...
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import load_raw  # noqa: E402


def plane(P):
    c = P.mean(0)
    _, sv, vt = np.linalg.svd(P - c, full_matrices=False)
    return vt[2], float(np.sqrt(sv[2] ** 2 / len(P)))


def ransac_vertical(P, rng, iters=300, th=0.015):
    """ほぼ垂直な平面のうち、最も多くの点が乗るものを探す。"""
    best, bn, bd = 0, None, None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        if abs(n[1]) > 0.2:
            continue
        cnt = int((np.abs((P - a) @ n) < th).sum())
        if cnt > best:
            best, bn, bd = cnt, n, a
    inl = P[np.abs((P - bd) @ bn) < th]
    n, res = plane(inl)
    return n, res, inl


for d in sys.argv[1:]:
    out = Path(d)
    raw = load_raw(out / "raw.npz")
    T = np.array(json.loads((out / "room.json").read_text(encoding="utf-8"))["T_room_from_world"])
    keep = raw["conf"] >= np.percentile(raw["conf"], 15)
    P = raw["points"][keep].astype(float) @ T[:3, :3].T + T[:3, 3]
    rng = np.random.default_rng(0)
    S = P[rng.choice(len(P), min(len(P), 150000), replace=False)]
    W = S[(S[:, 1] > 0.3) & (S[:, 1] < 2.0)]
    n1, r1, in1 = ransac_vertical(W, rng)
    rest = W[np.abs((W - in1.mean(0)) @ n1) > 0.06]
    n2, r2, in2 = ransac_vertical(rest, rng)
    F = S[np.abs(S[:, 1]) < 0.03]
    nf, rf = plane(F)
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))))
    K = raw["intrinsics"]
    print(f"{out.name}: 壁どうしの角 {ang(n1, n2):.1f}°（90 が正しい） ｜ 壁と床 {ang(n1, nf):.1f}° / {ang(n2, nf):.1f}° ｜ 面のでこぼこ 壁 {r1*1000:.1f}/{r2*1000:.1f}mm 床 {rf*1000:.1f}mm ｜ 焦点距離 {np.median(K[:,0,0]):.0f}px（{K[:,0,0].min():.0f}〜{K[:,0,0].max():.0f}） ｜ 壁の点 {len(in1)}/{len(in2)}")
