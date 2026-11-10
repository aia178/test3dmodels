"""実験（TUM RGB-D）: 写真に加えて、カメラ情報・カメラの位置・奥行きをモデルへ渡すと、復元がどこまで良くなるか。

狙い: スマホのアプリにすると、AR 機能（ARCore / ARKit）から実寸のカメラ位置が、iPhone Pro なら LiDAR の奥行きも取れる。
     それをモデルの入力に足したときの効果を、実測つきの公開データで先に確かめる。
     公開データの位置はモーションキャプチャ、奥行きは Kinect で、スマホの値より正確。なので結果は「上限の目安」。
     位置にわざと誤差を足す条件で、スマホの AR がずれた場合を模す。

使い方（server フォルダで）:
  uv run python ../experiments/tum_posed.py <データ> <出力> <枚数> <開始秒> <終了秒> <mode> [大きさのずれ=1.0] [位置のゆらぎ m=0] [向きのゆらぎ 度=0] [1 秒あたりのずれ m=0]
    mode: intr = カメラ情報だけ / pose = カメラ情報＋位置 / depth = カメラ情報＋位置＋奥行き / depthonly = カメラ情報＋奥行き
  選ぶ写真は tum_eval.py と同じ（枚数・開始秒・終了秒が同じなら同じ写真）。評価は tum_cross_eval.py / tum_size_eval.py をそのまま使える。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
sys.path.insert(0, str(here))
from tum_eval import associate, evaluate, pick_evenly, read_list  # noqa: E402

FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3


def rot_vec(v: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def main() -> None:
    import torch
    from mapanything.utils.image import preprocess_inputs

    from roomfit.arcore_import import quat_to_rot
    from roomfit.reconstruct import run_mapanything, save_raw
    from roomfit.structure import StructureError
    from roomfit.worker import capture_quality, structure_from_raw

    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    n, t0, t1, mode = int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
    scale = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
    jit_t = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
    jit_r = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
    drift = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0
    every = int(sys.argv[11]) if len(sys.argv) > 11 else 1   # AR が位置を見失った場面を模す: every 枚に 1 枚だけ位置を渡す
    pairs = pick_evenly(associate(read_list(src / "rgb.txt"), read_list(src / "depth.txt")), n, t0, t1)
    gt = np.array([[float(x) for x in line.split()] for line in (src / "groundtruth.txt").read_text().splitlines() if not line.startswith("#")])
    rng = np.random.default_rng(0)
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], np.float32)
    (out / "images").mkdir(parents=True, exist_ok=True)
    views, paths, p0, drift_dir = [], [], None, rng.normal(size=3)
    drift_dir /= np.linalg.norm(drift_dir)
    for i, (t, rgb, dname) in enumerate(pairs):
        p = out / "images" / f"{i:03d}.png"
        im = Image.open(src / rgb).convert("RGB")
        im.save(p)
        paths.append(p)
        view = {"img": torch.from_numpy(np.asarray(im).copy()), "intrinsics": torch.from_numpy(K)}
        if mode in ("pose", "depth") and i % every == 0:
            k = int(np.argmin(np.abs(gt[:, 0] - t)))
            R, tr = quat_to_rot(gt[k, 4:8]), gt[k, 1:4].copy()
            if p0 is None:
                p0 = tr.copy()
            # スマホの AR のずれを模す: 全体の大きさのずれ、時間とともに溜まるずれ、コマごとのゆらぎ
            tr = p0 + (tr - p0) * scale + drift_dir * drift * (t - pairs[0][0]) + rng.normal(scale=jit_t, size=3)
            R = R @ rot_vec(np.radians(jit_r) * rng.normal(size=3))
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3], c2w[:3, 3] = R, tr
            view["camera_poses"] = torch.from_numpy(c2w)
        if mode in ("depth", "depthonly"):
            view["depth_z"] = torch.from_numpy((np.asarray(Image.open(src / dname)).astype(np.float32) / 5000.0))
        if mode != "intr":
            view["is_metric_scale"] = torch.tensor([True])
        views.append(view)
    print(f"{mode}: {len(pairs)} 枚（{pairs[-1][0] - pairs[0][0]:.1f} 秒） 大きさ×{scale} ゆらぎ {jit_t}m/{jit_r}度 ずれ {drift}m/秒 位置は {every} 枚に 1 枚", flush=True)
    t = time.time()
    raw = run_mapanything(paths, views=preprocess_inputs(views))
    raw["meta"].update({"sec_total": round(time.time() - t, 1), "input_mode": mode, "pose_scale": scale, "pose_jitter_m": jit_t, "pose_jitter_deg": jit_r, "pose_drift_mps": drift, "pose_every": every})
    save_raw(raw, out / "raw.npz")
    res = evaluate(raw, [src / d for _, _, d in pairs])
    res["capture_quality"] = capture_quality(raw)
    res["sec_inference"] = raw["meta"]["sec_inference"]
    res["input_mode"] = mode
    try:
        room = structure_from_raw(raw, out)
        res["structure"] = {"ceiling_m": room["ceiling_height"], "edges_observed": room["edge_status"].count("observed"), "edges": len(room["edge_status"])}
    except StructureError as e:
        res["structure"] = {"failed": e.code}
    (out / "tum_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "views"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
