"""実験: 撮影アプリの写真を MapAnything で復元し、カメラ位置を ARCore の姿勢に合わせて縮尺を比べる。

狙い: この端末では ARCore の深度が壁の位置の根拠として弱かった（信頼度がほぼ 0）。
      そこで「形は写真からの復元、縮尺と重力は ARCore の姿勢」という組み合わせが成り立つかを見る。

出力: MapAnything のカメラ位置を相似変換（回転・平行移動・一様な拡大縮小）で ARCore の位置へ合わせたときの
      拡大率（1.0 なら MapAnything の実寸推定と ARCore の実寸が一致）と、合わせた後のずれ。

使い方（server フォルダで）:
  uv run python ../experiments/run_arcore_align.py <記録フォルダ> <出力フォルダ> [回転: 0|90|180|270（時計回り, 既定 90）]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import run_mapanything, save_raw  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
rot_cw = int(sys.argv[3]) if len(sys.argv) > 3 else 90
(out / "images").mkdir(parents=True, exist_ok=True)

frames = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(src.glob("[0-9][0-9][0-9].json"))]
frames = [f for f in frames if (src / f"{f['id']}.jpg").exists()]
paths = []
for f in frames:
    with Image.open(src / f"{f['id']}.jpg") as im:
        im = im.convert("RGB").rotate(-rot_cw, expand=True)  # センサーの向き → 見たままの向き
        im.thumbnail((1024, 1024), Image.LANCZOS)
        p = out / "images" / f"{f['id']}.jpg"
        im.save(p, "JPEG", quality=92)
        paths.append(p)
print("views:", len(paths), flush=True)

t0 = time.time()
raw = run_mapanything(paths, on_stage=lambda s: print("stage:", s, flush=True))
save_raw(raw, out / "raw.npz")
print("sec:", round(time.time() - t0, 1), raw["meta"], flush=True)


def umeyama(src_pts: np.ndarray, dst_pts: np.ndarray):
    """dst ≈ s * R @ src + t となる s, R, t（最小二乗）。"""
    mu_s, mu_d = src_pts.mean(0), dst_pts.mean(0)
    xs, xd = src_pts - mu_s, dst_pts - mu_d
    cov = xd.T @ xs / len(src_pts)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (xs ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


ma = raw["cam2world"][:, :3, 3]
ar = np.array([f["t"] for f in frames], float)
s, R, t = umeyama(ma, ar)
aligned = (s * (R @ ma.T)).T + t
res = np.linalg.norm(aligned - ar, axis=1)
span_ar = float(np.linalg.norm(ar.max(0) - ar.min(0)))
span_ma = float(np.linalg.norm(ma.max(0) - ma.min(0)))
# 1 場面ずつ外して拡大率がどれだけ変わるか（少ない場面数・小さな動きでの不安定さの目安）
loo = [umeyama(np.delete(ma, i, 0), np.delete(ar, i, 0))[0] for i in range(len(ma))]
# 重力: ARCore の上(+Y)と、合わせた後の MapAnything カメラの「画像の上」(OpenCV の -Y) の平均との角度
up_ma = (R @ (-raw["cam2world"][:, :3, 1]).T).T.mean(0)
up_ma /= np.linalg.norm(up_ma)
result = {
    "n_views": len(frames),
    "scale_arcore_over_mapanything": round(float(s), 4),
    "scale_leave_one_out_min_max": [round(float(min(loo)), 4), round(float(max(loo)), 4)],
    "camera_span_arcore_m": round(span_ar, 3),
    "camera_span_mapanything_m": round(span_ma, 3),
    "residual_after_alignment_m": {"median": round(float(np.median(res)), 4), "max": round(float(res.max()), 4)},
    "angle_between_arcore_up_and_image_up_deg": round(float(np.degrees(np.arccos(np.clip(up_ma[1], -1, 1)))), 1),
    "det_R": round(float(np.linalg.det(R)), 4),
    "sec_inference": raw["meta"]["sec_inference"],
}
(out / "align.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
print(json.dumps(result, indent=1))
