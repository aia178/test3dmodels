"""実験（TUM RGB-D）: 2 つの範囲のつなぎ方を変えて、範囲をまたぐ長さの誤差がどう変わるかを比べる。

復元はやり直さない。tum_chunk.py が作った A・B の raw.npz を読むだけなので、数十秒で終わる。
背景: 共通の 3 枚の全画素を重ねると、残差の中央値が 23mm 残り、写真ごとの縮尺も 0.96〜0.99 とばらつく（検証記録 §14）。
使い方（server フォルダで）: uv run python ../experiments/tum_join_variants.py <データ> <A の出力> <B の出力> <A 開始秒> <A 終了秒> <B 終了秒> [B の新しい枚数] [共通の枚数]
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
from tum_chunk import robust_similarity, select, umeyama  # noqa: E402
from tum_eval import model_grid_to_source, sample_nearest  # noqa: E402

FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3
src, out_a, out_b = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
a0, a1, b1 = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
n_new = int(sys.argv[7]) if len(sys.argv) > 7 else 10
n_shared = int(sys.argv[8]) if len(sys.argv) > 8 else 3
raw_a, raw_b = load_raw(out_a / "raw.npz"), load_raw(out_b / "raw.npz")
pa, pb = select(src, a0, a1, b1, n_new, n_shared, len(raw_a["cam2world"]))
ga, gb = list(view_grids(raw_a)), list(view_grids(raw_b))
na = len(ga)


def cam_depth(raw, v, xyz):
    T = raw["cam2world"][v]
    return ((xyz.astype(np.float64) - T[:3, 3]) @ T[:3, :3])[..., 2]


# 共通の写真の対応（同じ画素）。奥行きと、画像の中での位置も持っておく
S, D, Z, EDGE, VIEW = [], [], [], [], []
for k in range(n_shared):
    (ma, xa), (mb, xb) = ga[na - n_shared + k], gb[k]
    ok = ma & mb & np.isfinite(xa).all(-1) & np.isfinite(xb).all(-1)
    h, w = ok.shape
    yy, xx = np.mgrid[0:h, 0:w]
    border = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx]) / min(h, w)   # 0 = 画像の端、0.5 = 中央
    S.append(xb[ok].astype(np.float64)); D.append(xa[ok].astype(np.float64))
    Z.append(cam_depth(raw_a, na - n_shared + k, xa)[ok]); EDGE.append(border[ok]); VIEW.append(np.full(int(ok.sum()), k))
S, D, Z, EDGE, VIEW = map(np.concatenate, (S, D, Z, EDGE, VIEW))
step = 5
S, D, Z, EDGE, VIEW = S[::step], D[::step], Z[::step], EDGE[::step], VIEW[::step]

# 実測との比較に使う点（写真ごとに 3000 点）。B 側は変換前の座標で持ち、つなぎ方ごとに変換する
gt = np.array([[float(x) for x in line.split()] for line in (src / "groundtruth.txt").read_text().splitlines() if not line.startswith("#")])
rng = np.random.default_rng(0)
P, Q, C = [], [], []
for chunk, grids, frames in ((0, ga, pa), (1, gb[n_shared:], pb[n_shared:])):
    for (mask, xyz), (tt, _, dname) in zip(grids, frames):
        k = int(np.argmin(np.abs(gt[:, 0] - tt)))
        Rg, tr = quat_to_rot(gt[k, 4:8]), gt[k, 1:4]
        d = np.asarray(Image.open(src / dname)).astype(np.float64) / 5000.0
        xs, ys = model_grid_to_source(xyz.shape[0], xyz.shape[1], d.shape[0], d.shape[1])
        z = sample_nearest(d, xs, ys)
        Qc = np.stack([(xs - CX) / FX * z, (ys - CY) / FY * z, z], -1)
        idx = np.argwhere(mask & (z > 0.3) & (z < 4.0) & np.isfinite(xyz).all(-1))
        sel = idx[rng.integers(0, len(idx), 3000)]
        P.append(xyz[sel[:, 0], sel[:, 1]].astype(np.float64)); Q.append(Qc[sel[:, 0], sel[:, 1]] @ Rg.T + tr); C.append(np.full(len(sel), chunk))
P, Q, C = map(np.concatenate, (P, Q, C))
i, j = rng.integers(0, len(P), 400000), rng.integers(0, len(P), 400000)
cross = C[i] != C[j]
dq = np.linalg.norm(Q[i] - Q[j], axis=1)


def evaluate(s, R, t):
    Pt = np.where((C == 1)[:, None], P @ R.T * s + t, P)
    dp = np.linalg.norm(Pt[i] - Pt[j], axis=1)
    out = {}
    for name, m in (("cross_1.5_3m", cross & (dq > 1.5) & (dq <= 3)), ("cross_over_3m", cross & (dq > 3))):
        r = dp[m] / dq[m]
        out[name] = {"error_pct": round((float(np.median(r)) - 1) * 100, 1), "abs_cm_median": round(float(np.median(np.abs(dp[m] - dq[m]))) * 100, 1),
                     "abs_cm_p90": round(float(np.percentile(np.abs(dp[m] - dq[m]), 90)) * 100, 1)}
    return out


def fit(mask, rigid_scale=None):
    """mask の対応だけで合わせる。rigid_scale を与えると、縮尺を固定して回転と移動だけを求める。"""
    s, R, t, res = robust_similarity(S[mask], D[mask])
    if rigid_scale is not None:
        keep = np.linalg.norm(S[mask] @ R.T * s + t - D[mask], axis=1) < 2.5 * np.median(res)
        X, Y = S[mask][keep] * rigid_scale, D[mask][keep]
        _, R, t = umeyama(X, Y)                       # 縮尺は使わず、回転だけ取る
        t = Y.mean(0) - rigid_scale * R @ S[mask][keep].mean(0)
        s = rigid_scale
    return s, R, t


def pair_scale(mask, n=200000):
    """同じ写真の中の 2 点間距離の比（A での長さ ÷ B での長さ）の中央値。位置合わせに頼らない縮尺の決め方。"""
    idx = np.flatnonzero(mask)
    a, b = rng.choice(idx, n), rng.choice(idx, n)
    ok = VIEW[a] == VIEW[b]
    ds, dd = np.linalg.norm(S[a] - S[b], axis=1)[ok], np.linalg.norm(D[a] - D[b], axis=1)[ok]
    far = ds > 0.5
    return float(np.median(dd[far] / ds[far]))


everything = np.ones(len(S), bool)
near = Z < np.median(Z)
center = EDGE > 0.15
results = {}
variants = {
    "全画素（今のやり方）": fit(everything),
    "近い半分の点だけ": fit(near),
    "画像の端 15% を除く": fit(center),
    "近い半分 かつ 端を除く": fit(near & center),
    "縮尺は 2 点間距離の比で決め、回転と移動だけ合わせる": fit(everything, pair_scale(everything)),
    "同上（端を除く）": fit(center, pair_scale(center)),
    "縮尺を変えない（1.0 固定）": fit(everything, 1.0),
}
for k in range(n_shared):
    variants[f"共通の写真 {k + 1} 枚目だけ"] = fit(VIEW == k)
for name, (s, R, t) in variants.items():
    res = np.linalg.norm(S @ R.T * s + t - D, axis=1)
    results[name] = {"scale": round(s, 4), "residual_mm_median": round(float(np.median(res)) * 1000, 1), **evaluate(s, R, t)}
    print(name, results[name], flush=True)

# 参考: 実測を使ったときの最良（B の評価点を実測に直接合わせる）。つなぎ方をどれだけ良くしても、これより良くはならない
m1, m0 = C == 1, C == 0
sa, Ra, ta, _ = robust_similarity(P[m0], Q[m0])
sb, Rb, tb, _ = robust_similarity(P[m1], Q[m1])
s_or = sb / sa
R_or = Ra.T @ Rb
t_or = Ra.T @ (tb - ta) / sa
results["参考: 実測から求めた最良のつなぎ"] = {"scale": round(s_or, 4), **evaluate(s_or, R_or, t_or)}
print("参考: 実測から求めた最良のつなぎ", results["参考: 実測から求めた最良のつなぎ"], flush=True)
(out_b / "tum_join_variants.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


# 追加: 共通の写真で合わせたあと、2 つの範囲の点全体どうしで合わせ直す（最近傍の点を対応として繰り返す）
def cloud(grids, n_each=20000):
    pts = []
    for mask, xyz in grids:
        p = xyz[mask & np.isfinite(xyz).all(-1)].astype(np.float64)
        pts.append(p[rng.integers(0, len(p), n_each)])
    return np.concatenate(pts)


def refine(s, R, t, fix_scale, rounds=15, reach=0.10):
    from scipy.spatial import cKDTree
    A, B = cloud(ga), cloud(gb[n_shared:])
    tree = cKDTree(A)
    used = 0
    for _ in range(rounds):
        Bt = B @ R.T * s + t
        dist, nn = tree.query(Bt, distance_upper_bound=reach)
        ok = np.isfinite(dist)
        used = int(ok.sum())
        if used < 5000:
            break
        if fix_scale:
            _, R, _ = umeyama(B[ok] * s, A[nn[ok]])
            t = A[nn[ok]].mean(0) - s * R @ B[ok].mean(0)
        else:
            s, R, t = umeyama(B[ok], A[nn[ok]])
    return s, R, t, used


for name, base, fix in (("全画素で合わせたあと、点全体で合わせ直す（縮尺も動かす）", variants["全画素（今のやり方）"], False),
                        ("全画素で合わせたあと、点全体で合わせ直す（縮尺は固定）", variants["全画素（今のやり方）"], True),
                        ("縮尺は 2 点間距離の比、点全体で合わせ直す（縮尺は固定）", variants["縮尺は 2 点間距離の比で決め、回転と移動だけ合わせる"], True)):
    s, R, t, used = refine(*base, fix)
    results[name] = {"scale": round(s, 4), "matched_points": used, **evaluate(s, R, t)}
    print(name, results[name], flush=True)
(out_b / "tum_join_variants.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
