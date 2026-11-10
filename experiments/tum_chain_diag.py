"""診断（TUM RGB-D）: 鎖つなぎの「大きさ合わせ」がどこで狂うかを、実測で切り分ける。

範囲ごとに、(1) その範囲全体の本当の縮尺（復元 ÷ 実測）、(2) 共通の 3 枚だけで見た縮尺、を出す。
つなぎで使う縮尺 = 前の範囲と今の範囲の「共通の写真での大きさ」の比。これが、範囲全体の縮尺の比とどれだけ違うかを見る。
使い方（server フォルダで）: uv run python ../experiments/tum_chain_diag.py <データ> <出力の親フォルダ> [終了秒=49]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
sys.path.insert(0, str(here))
from roomfit.arcore_import import quat_to_rot  # noqa: E402
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.stitch import robust_similarity  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402
from tum_chain import CX, CY, FX, FY, N_SHARED, plan  # noqa: E402
from tum_eval import model_grid_to_source, sample_nearest  # noqa: E402

src, base = Path(sys.argv[1]), Path(sys.argv[2])
t_end = float(sys.argv[3]) if len(sys.argv) > 3 else 49.0
gt = np.array([[float(x) for x in line.split()] for line in (src / "groundtruth.txt").read_text().splitlines() if not line.startswith("#")])
rng = np.random.default_rng(0)


def view_samples(mask, xyz, frame, n=3000):
    tt, _, dname = frame
    g = int(np.argmin(np.abs(gt[:, 0] - tt)))
    Rg, tr = quat_to_rot(gt[g, 4:8]), gt[g, 1:4]
    d = np.asarray(Image.open(src / dname)).astype(np.float64) / 5000.0
    xs, ys = model_grid_to_source(xyz.shape[0], xyz.shape[1], d.shape[0], d.shape[1])
    z = sample_nearest(d, xs, ys)
    Qc = np.stack([(xs - CX) / FX * z, (ys - CY) / FY * z, z], -1)
    idx = np.argwhere(mask & (z > 0.3) & (z < 4.0) & np.isfinite(xyz).all(-1))
    sel = idx[rng.integers(0, len(idx), n)]
    return xyz[sel[:, 0], sel[:, 1]].astype(np.float64), Qc[sel[:, 0], sel[:, 1]] @ Rg.T + tr


def pair_scale(P, Q, n=200000):
    """2 点間距離の比（復元 ÷ 実測）の中央値。50cm 以上離れた組だけ。"""
    i, j = rng.integers(0, len(P), n), rng.integers(0, len(P), n)
    dq = np.linalg.norm(Q[i] - Q[j], axis=1)
    ok = dq > 0.5
    return float(np.median(np.linalg.norm(P[i] - P[j], axis=1)[ok] / dq[ok]))


prev = None
for k, (out, frames) in enumerate(plan(src, base, t_end)):
    if not (out / "raw.npz").exists():
        break
    grids = view_grids(load_raw(out / "raw.npz"))
    per = [view_samples(m, x, f) for (m, x), f in zip(grids, frames)]
    whole = pair_scale(np.concatenate([p for p, _ in per]), np.concatenate([q for _, q in per]))
    each = [pair_scale(p, q, 40000) for p, q in per]
    head = pair_scale(np.concatenate([p for p, _ in per[:N_SHARED]]), np.concatenate([q for _, q in per[:N_SHARED]]))
    tail = pair_scale(np.concatenate([p for p, _ in per[-N_SHARED:]]), np.concatenate([q for _, q in per[-N_SHARED:]]))
    line = f"範囲 {k}: 全体 {whole:.3f}  先頭3枚 {head:.3f}  末尾3枚 {tail:.3f}  写真ごと {min(each):.2f}〜{max(each):.2f}"
    if prev is not None:
        src_pts, dst_pts = [], []
        for (ma, xa), (mb, xb) in zip(prev["tail_grids"], grids[:N_SHARED]):
            ok = ma & mb & np.isfinite(xa).all(-1) & np.isfinite(xb).all(-1)
            src_pts.append(xb[ok].astype(np.float64)); dst_pts.append(xa[ok].astype(np.float64))
        s, _, _, _ = robust_similarity(np.concatenate(src_pts)[::5], np.concatenate(dst_pts)[::5])
        line += f"  | つなぎの縮尺 {s:.3f}  共通写真での本来の比 {prev['tail'] / head:.3f}  範囲全体での理想 {prev['whole'] / whole:.3f}"
    print(line, flush=True)
    prev = {"tail_grids": grids[-N_SHARED:], "tail": tail, "whole": whole}
