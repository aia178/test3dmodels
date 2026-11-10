"""AR の位置（実寸）と写真どうしの対応点から三角測量した奥行きと、MapAnything の奥行きを、同じ画素で比べる（再推論なし）。

三角測量の奥行きはモデルを介さず、AR の位置と内部パラメータだけで決まる。モデルの奥行き ÷ 三角測量の奥行き の中央値が、
モデルの大きさの倍率の見積もりになる（1.04 なら、モデルは 4% 大きい）。
対応点: SIFT ＋ 比の検定。AR の姿勢から求めたエピ極線から 2 画素以内、再投影の誤差 2 画素以内、2 本の視線の角度 3° 以上の点だけを使う。
モデルの奥行き: モデルに入った画像の画素（元の写真の画素から、preprocess_inputs の縮小・切り抜きで写す）の点を、そのコマのカメラ座標に直した z。

使い方（server フォルダで）: uv run python ../experiments/triangulate_scale.py <ジョブのフォルダ>
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.ar_views import build_views, plan_views  # noqa: E402
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402

EPI_PX, REPROJ_PX, MIN_ANGLE_DEG = 2.0, 2.0, 3.0


def projection(K, c2w):
    R, C = c2w[:3, :3], c2w[:3, 3]
    return K @ np.hstack([R.T, (-R.T @ C)[:, None]])


def fundamental(Ka, Ta, Kb, Tb):
    """a の画素 → b のエピ極線。"""
    R = Tb[:3, :3].T @ Ta[:3, :3]
    t = Tb[:3, :3].T @ (Ta[:3, 3] - Tb[:3, 3])
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return np.linalg.inv(Kb).T @ tx @ R @ np.linalg.inv(Ka)


def epi_dist(F, pa, pb):
    ha, hb = np.c_[pa, np.ones(len(pa))], np.c_[pb, np.ones(len(pb))]
    la = ha @ F.T
    return np.abs((hb * la).sum(1)) / np.linalg.norm(la[:, :2], axis=1)


def triangulated_pairs(images, Ks, c2ws, sift, matcher):
    """各組の、使える対応点の (コマ a, a の画素, a のカメラ座標での三角測量の奥行き)。"""
    feats = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None) for im in images]
    out = []
    for a in range(len(images)):
        for b in range(len(images)):
            if a == b:
                continue
            (ka, da), (kb, db) = feats[a], feats[b]
            if da is None or db is None:
                continue
            ms = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
            if len(ms) < 20:
                continue
            pa = np.float64([ka[m.queryIdx].pt for m in ms])
            pb = np.float64([kb[m.trainIdx].pt for m in ms])
            ok = epi_dist(fundamental(Ks[a], c2ws[a], Ks[b], c2ws[b]), pa, pb) < EPI_PX
            pa, pb = pa[ok], pb[ok]
            if len(pa) < 10:
                continue
            X = cv2.triangulatePoints(projection(Ks[a], c2ws[a]), projection(Ks[b], c2ws[b]), pa.T, pb.T)
            X = (X[:3] / X[3]).T
            good = np.ones(len(X), bool)
            for K, T, p in ((Ks[a], c2ws[a], pa), (Ks[b], c2ws[b], pb)):
                pc = (X - T[:3, 3]) @ T[:3, :3]
                uv = pc @ K.T
                good &= (pc[:, 2] > 0.2) & (np.linalg.norm(uv[:, :2] / uv[:, 2:] - p, axis=1) < REPROJ_PX)
            ra, rb = X - c2ws[a][:3, 3], X - c2ws[b][:3, 3]
            ang = np.degrees(np.arccos(np.clip((ra * rb).sum(1) / np.linalg.norm(ra, axis=1) / np.linalg.norm(rb, axis=1), -1, 1)))
            good &= ang > MIN_ANGLE_DEG
            za = ((X - c2ws[a][:3, 3]) @ c2ws[a][:3, :3])[:, 2]
            out += [(a, pa[i], za[i]) for i in np.flatnonzero(good)]
    return out


def model_depth_ratios(obs, grids, cam2world, to_model):
    """三角測量の点ごとの、モデルの奥行き ÷ 三角測量の奥行き。"""
    r = []
    for a, p, z in obs:
        um, vm = to_model[a] @ np.r_[p, 1.0]
        mask, xyz = grids[a]
        i, j = int(round(vm - 0.5)), int(round(um - 0.5))
        if not (0 <= i < mask.shape[0] and 0 <= j < mask.shape[1]) or not mask[i, j]:
            continue
        T = cam2world[a]
        zm = ((xyz[i, j] - T[:3, 3]) @ T[:3, :3])[2]
        r.append(zm / z)
    return np.array(r)


def main():
    job = Path(sys.argv[1])
    raw = load_raw(job / "raw.npz")
    plan = plan_views(job / "capture")
    assert plan["ids"] == raw["meta"]["ar_frames"]["ids"], "復元に使ったコマと選び直したコマが違う"
    with tempfile.TemporaryDirectory() as d:
        _, views = build_views(job / "capture", Path(d), plan)
    images, to_model = [], []
    for fid, K, v in zip(plan["ids"], plan["K"], views):
        with Image.open(job / "capture" / f"{fid}.jpg") as im:
            images.append(np.asarray(im.convert("RGB").rotate(-90, expand=True)))
        Kp = v["intrinsics"][0].numpy().astype(np.float64)
        sx, sy = Kp[0, 0] / K[0, 0], Kp[1, 1] / K[1, 1]
        to_model.append(np.array([[sx, 0, Kp[0, 2] - sx * K[0, 2]], [0, sy, Kp[1, 2] - sy * K[1, 2]]]))
    obs = triangulated_pairs(images, plan["K"], plan["c2w"], cv2.SIFT_create(4000), cv2.BFMatcher(cv2.NORM_L2))
    r = model_depth_ratios(obs, view_grids(raw), raw["cam2world"], to_model)
    zt = np.array([z for _, _, z in obs])
    print(f"三角測量の点 {len(obs)}（奥行き {np.percentile(zt, 10):.2f}〜{np.percentile(zt, 90):.2f}m）、モデルの点と比べられたもの {len(r)}")
    print(f"モデルの奥行き ÷ 三角測量の奥行き: 中央値 {np.median(r):.4f}（25〜75% {np.percentile(r, 25):.4f}〜{np.percentile(r, 75):.4f}）")
    per = {a: np.median(model_depth_ratios([o for o in obs if o[0] == a], view_grids(raw), raw["cam2world"], to_model) if any(o[0] == a for o in obs) else [np.nan]) for a in range(len(images))}
    print("コマごと: " + " ".join(f"{a}:{v:.3f}" for a, v in per.items()))
    (job / "triangulate_scale.json").write_text(json.dumps({"n": int(len(r)), "median": float(np.median(r)), "per_view": {str(k): float(v) for k, v in per.items()}}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
