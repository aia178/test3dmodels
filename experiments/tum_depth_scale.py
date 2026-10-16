"""実験（TUM RGB-D）: 写真ごとの奥行きの倍率を、写真どうしの対応点と AR の位置から補正する。

背景（docs/検証記録.md §15）: AR の位置をモデルへ渡しても、写真ごとの大きさのばらつき（16〜28%）が残り、これが誤差の主因。
やり方:
  1. 写真どうしで同じ点を見つける（SIFT ＋ 比の検定 ＋ RANSAC。SIFT は特許切れで商用可）
  2. 渡した位置（AR 相当）で三角測量し、その点の 3D 位置を求める（モデルの出力は使わない、独立の手がかり）
  3. 各写真で、三角測量の奥行き ÷ モデルの奥行き の中央値を、その写真の倍率とする
  4. モデルのカメラ座標の点を倍率で伸縮し、渡した位置で部屋の座標へ置き直す
比べるもの:
  A = モデルの出力そのまま
  B = モデルのカメラ座標の点を、渡した位置で置き直しただけ（倍率なし）
  C = B ＋ 写真ごとの倍率補正
渡した位置は tum_posed.py と同じ乱数の順で作り直す（同じ設定なら同じ位置になる）。

使い方（server フォルダで）:
  uv run python ../experiments/tum_depth_scale.py <データ> <tum_posed の出力> <枚数> <開始秒> <終了秒> [大きさ ゆらぎm ゆらぎ度 ずれm/秒]
  出力: <tum_posed の出力>_B, _C（raw.npz）。評価は tum_cross_eval.py / tum_size_eval.py。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
sys.path.insert(0, str(here))
from roomfit.arcore_import import quat_to_rot  # noqa: E402
from roomfit.reconstruct import load_raw, save_raw  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402
from tum_eval import associate, pick_evenly, read_list  # noqa: E402
from tum_posed import rot_vec  # noqa: E402

FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
# freiburg1 のカラーカメラのレンズのゆがみ（公式の校正値）。配布画像は補正されていないので、対応点の座標から取り除く
DIST = np.array([0.262383, -0.953104, -0.005358, 0.002628, 1.163314])
REP_MAX, ANG_MIN = 4.0, 2.0
MAX_GAP = 3        # 何枚先の写真まで対応点を探すか（離れた写真どうしは誤対応が多い）
GATE_M = 0.25      # 補正前の 3D 位置の食い違いがこれを超える対応は使わない [m]   # 三角測量した点の再投影誤差の上限 [画素]、2 つの視線の角度の下限 [度]


def given_poses(src, pairs, scale, jit_t, jit_r, drift):
    """tum_posed.py と同じ乱数の順で、モデルへ渡した位置（cam2world, OpenCV）を作り直す。"""
    gt = np.array([[float(x) for x in line.split()] for line in (src / "groundtruth.txt").read_text().splitlines() if not line.startswith("#")])
    rng = np.random.default_rng(0)
    drift_dir = rng.normal(size=3)
    drift_dir /= np.linalg.norm(drift_dir)
    out, p0 = [], None
    for t, _, _ in pairs:
        k = int(np.argmin(np.abs(gt[:, 0] - t)))
        R, tr = quat_to_rot(gt[k, 4:8]), gt[k, 1:4].copy()
        if p0 is None:
            p0 = tr.copy()
        tr = p0 + (tr - p0) * scale + drift_dir * drift * (t - pairs[0][0]) + rng.normal(scale=jit_t, size=3)
        R = R @ rot_vec(np.radians(jit_r) * rng.normal(size=3))
        c = np.eye(4)
        c[:3, :3], c[:3, 3] = R, tr
        out.append(c)
    return np.stack(out)


def match(img_a, img_b, sift):
    ka, da = sift.detectAndCompute(img_a, None)
    kb, db = sift.detectAndCompute(img_b, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2))
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 20:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2))
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    ua = cv2.undistortPoints(pa.reshape(-1, 1, 2), K, DIST, P=K).reshape(-1, 2)
    ub = cv2.undistortPoints(pb.reshape(-1, 1, 2), K, DIST, P=K).reshape(-1, 2)
    E, inl = cv2.findEssentialMat(ua, ub, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
    if inl is None:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2))
    keep = inl.ravel().astype(bool)
    return pa[keep], pb[keep], ua[keep], ub[keep]   # 元の画素（モデルの画像を引くため）とゆがみ補正後（三角測量のため）


def main() -> None:
    src, run = Path(sys.argv[1]), Path(sys.argv[2])
    n, t0, t1 = int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    scale, jit_t, jit_r, drift = (float(x) for x in (sys.argv[6:10] if len(sys.argv) > 9 else (1.0, 0.0, 0.0, 0.0)))
    pairs = pick_evenly(associate(read_list(src / "rgb.txt"), read_list(src / "depth.txt")), n, t0, t1)
    raw = load_raw(run / "raw.npz")
    grids = view_grids(raw)
    N = len(grids)
    Tin = given_poses(src, pairs, scale, jit_t, jit_r, drift)
    Tout = raw["cam2world"]
    imgs = [cv2.imread(str(run / "images" / f"{i:03d}.png"), cv2.IMREAD_GRAYSCALE) for i in range(N)]
    H0, W0 = imgs[0].shape
    hm, wm = grids[0][0].shape
    s = max(wm / W0, hm / H0)
    ox, oy = (W0 * s - wm) / 2, (hm and (H0 * s - hm) / 2)

    # モデルのカメラ座標の点（出力の位置で world → camera）
    cam_pts = []
    for v, (mask, xyz) in enumerate(grids):
        pc = (xyz.astype(np.float64) - Tout[v][:3, 3]) @ Tout[v][:3, :3]
        cam_pts.append(pc)

    # 対応点ごとに「写真 i から出した 3D 位置」と「写真 j から出した 3D 位置」が一致するよう、写真ごとの倍率 a を決める。
    # X_i = R_i (a_i P_i) + t_i（P_i はモデルのカメラ座標の点、R, t は渡した位置）。a について線形なので、重みつき最小二乗をくり返す。
    sift = cv2.SIFT_create(nfeatures=4000)
    rows = []                                   # (i, j, R_i P_i, R_j P_j, t_i - t_j)
    n_pairs = 0
    for i in range(N):
        for j in range(i + 1, min(N, i + 1 + MAX_GAP)):
            pa, pb, _, _ = match(imgs[i], imgs[j], sift)
            if len(pa) < 15:
                continue
            n_pairs += 1
            ua_, va_ = np.clip(np.floor(pa[:, 0] * s - ox).astype(int), 0, wm - 1), np.clip(np.floor(pa[:, 1] * s - oy).astype(int), 0, hm - 1)
            ub_, vb_ = np.clip(np.floor(pb[:, 0] * s - ox).astype(int), 0, wm - 1), np.clip(np.floor(pb[:, 1] * s - oy).astype(int), 0, hm - 1)
            Pi, Pj = cam_pts[i][va_, ua_], cam_pts[j][vb_, ub_]
            ok = grids[i][0][va_, ua_] & grids[j][0][vb_, ub_] & np.isfinite(Pi).all(1) & np.isfinite(Pj).all(1)
            for pi_, pj_ in zip(Pi[ok], Pj[ok]):
                ri, rj, dt = Tin[i][:3, :3] @ pi_, Tin[j][:3, :3] @ pj_, Tin[i][:3, 3] - Tin[j][:3, 3]
                if np.linalg.norm(ri - rj + dt) < GATE_M:      # 補正前の時点で大きく食い違う対応は誤対応として捨てる
                    rows.append((i, j, ri, rj, dt))
    M = len(rows)
    counts = np.zeros(N, int)
    for i, j, *_ in rows:
        counts[i] += 1; counts[j] += 1
    a_used = np.ones(N)
    w = np.ones(M)
    LAM = 2.0                                   # 倍率を 1 付近に保つ弱い引き戻し（点 1 個ぶんの重み×LAM×点の数の目安）
    for it in range(10):
        A = np.zeros((3 * M + N, N)); bb = np.zeros(3 * M + N)
        for k, (i, j, ri, rj, dt) in enumerate(rows):
            A[3 * k:3 * k + 3, i] = w[k] * ri
            A[3 * k:3 * k + 3, j] = -w[k] * rj
            bb[3 * k:3 * k + 3] = -w[k] * dt
        reg = np.sqrt(LAM * max(1, M / N) * 0.05 ** 2)   # 5cm 相当の重さで 1 へ引き戻す
        A[3 * M:, :] = np.eye(N) * reg
        bb[3 * M:] = reg
        a_used = np.linalg.lstsq(A, bb, rcond=None)[0]
        res = np.array([np.linalg.norm(a_used[i] * ri - a_used[j] * rj + dt) for i, j, ri, rj, dt in rows])
        c = 1.345 * max(0.02, float(np.median(res)))      # Huber の境目
        w = np.where(res <= c, 1.0, np.sqrt(c / np.maximum(res, 1e-9)))
    a = a_used
    print(f"対応のあった組 {n_pairs}、対応点 {M}、残差の中央値 {np.median(res) * 100:.1f}cm")
    print("写真ごとの倍率:", np.round(a, 3).tolist())
    print("写真ごとの点の数:", counts.tolist())
    counts = counts.tolist()

    for tag, use_scale in (("B", False), ("C", True)):
        pts_all = []
        for v in range(N):
            mask = grids[v][0]
            pc = cam_pts[v][mask] * (a_used[v] if use_scale else 1.0)
            pts_all.append((pc @ Tin[v][:3, :3].T + Tin[v][:3, 3]).astype(np.float32))
        new = dict(raw)
        new["points"] = np.concatenate(pts_all)
        new["cam2world"] = Tin.astype(np.float64)
        new["meta"] = {**raw["meta"], "depth_scale_fix": tag, "per_view_scale": a_used.round(4).tolist(), "per_view_points": counts}
        out = Path(str(run) + "_" + tag)
        out.mkdir(exist_ok=True)
        save_raw(new, out / "raw.npz")
        (out / "per_view_scale.json").write_text(json.dumps({"scale": a.tolist(), "counts": counts}), encoding="utf-8")
    print("保存:", str(run) + "_B", str(run) + "_C")


if __name__ == "__main__":
    main()
