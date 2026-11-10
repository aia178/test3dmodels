"""カメラの位置を固定したまま、各写真の奥行きに掛ける倍率 s を、写真どうしの点が一番よく重なるように選ぶ（再推論なし）。

仮説: MapAnything に AR の位置を渡すと、カメラの位置は AR に従うが、写真ごとの奥行きはモデル自身の見積もりのまま残る。
奥行きが s 倍ずれていると、別の位置から見た同じ面が s に比例して食い違う。正しい位置が分かっていれば、食い違いが最小になる s が奥行きの倍率の答えになる。
点は X_v(s) = C_v + s * R_v p_v（p_v はカメラ座標の点）。重なりの悪さは、写真の組ごとに「片方の点から、もう片方の点群への最近傍距離」の中央値を足したもの。

使い方（server フォルダで）: uv run python ../experiments/depth_pose_scale.py <ジョブのフォルダ>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.ar_views import _rotated  # noqa: E402
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402

S_GRID = np.round(np.arange(0.85, 1.101, 0.005), 3)
NEAR = 0.15   # これより遠い最近傍は、相手の写真に写っていない所として数えない


def cam_points(c2w, xyz, mask, rng, n=4000):
    p = xyz[mask].astype(np.float64)
    p = p[rng.choice(len(p), min(n, len(p)), replace=False)]
    return (p - c2w[:3, 3]) @ c2w[:3, :3]


def overlap_cost(Rs, Cs, pcs, s):
    clouds = [C + s * pc @ R.T for R, C, pc in zip(Rs, Cs, pcs)]
    trees = [cKDTree(c) for c in clouds]
    out = []
    for a in range(len(clouds)):
        for b in range(len(clouds)):
            if a == b:
                continue
            d, _ = trees[b].query(clouds[a], distance_upper_bound=NEAR)
            ok = np.isfinite(d)
            if ok.mean() > 0.3:   # 重なりが 3 割以上ある組だけ
                out.append(np.median(d[ok]))
    return float(np.mean(out)), len(out)


def umeyama_rigid(A, B):
    """B ≈ R A + t（倍率は 1 に固定）。"""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - cb).T @ (A - ca))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, cb - R @ ca


def main():
    job = Path(sys.argv[1])
    raw = load_raw(job / "raw.npz")
    grids = view_grids(raw)
    rng = np.random.default_rng(0)
    pcs = [cam_points(T, xyz, m, rng) for T, (m, xyz) in zip(raw["cam2world"], grids)]
    ids = raw["meta"]["ar_frames"]["ids"]
    ar = np.stack([_rotated(json.loads((job / "capture" / f"{i}.json").read_text(encoding="utf-8")))[1] for i in ids])
    R, t = umeyama_rigid(raw["cam2world"][:, :3, 3], ar[:, :3, 3])
    Rs = [T[:3, :3] for T in raw["cam2world"]]
    for label, Cs in (("復元のカメラ位置", raw["cam2world"][:, :3, 3]), ("AR のカメラ位置", (ar[:, :3, 3] - t) @ R)):
        costs = [overlap_cost(Rs, Cs, pcs, s) for s in S_GRID]
        k = int(np.argmin([c for c, _ in costs]))
        print(f"{label}: 最良の s = {S_GRID[k]:.3f}（食い違い {costs[k][0] * 1000:.1f}mm、s=1 では {costs[list(S_GRID).index(1.0)][0] * 1000:.1f}mm、組 {costs[k][1]}）")
        print("   " + " ".join(f"{s:.2f}:{c * 1000:.1f}" for s, (c, _) in zip(S_GRID[::4], costs[::4])))


if __name__ == "__main__":
    main()
