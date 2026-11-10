"""実験（TUM RGB-D）: Depth Anything 3（DA3）で同じ写真を復元し、MapAnything と同じ物差しで比べる。

DA3 は別の実行環境（experiments/ext/da3env）で動かす。MapAnything と部品の版が合わないため。
出力は MapAnything と同じ形（raw.npz）にそろえるので、評価は tum_cross_eval.py / tum_size_eval.py / tum_eval.py をそのまま使える
（評価は server の環境で行う）。

モデル:
  nested = DA3NESTED-GIANT-LARGE-1.1（非商用。検証のみ。複数の写真から形と実寸を出す）
  metric = DA3METRIC-LARGE（Apache 2.0。写真 1 枚ずつ実寸の奥行きを出す。実寸 = 焦点距離 × 出力 ÷ 300）
  base   = DA3-BASE（Apache 2.0。複数の写真から形を出す。実寸は出ないので、写真だけだと大きさは任意）
mode:
  photo = 写真だけ / pose = カメラ情報と位置も渡す（nested のみ）

使い方（experiments/ext で）:
  da3env/Scripts/python.exe ../tum_da3.py <データ> <出力> <枚数> <開始秒> <終了秒> <nested|metric> <photo|pose>
"""
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / "ext" / "da3" / "src"))
sys.path.insert(0, str(here))
from tum_eval import associate, pick_evenly, read_list  # noqa: E402

FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3
REPO = {"nested": "depth-anything/DA3NESTED-GIANT-LARGE-1.1", "metric": "depth-anything/DA3METRIC-LARGE", "base": "depth-anything/DA3-BASE"}

# CPU では半精度の自動切り替えを使わず、32 ビットのまま計算する（精度を MapAnything の実行条件にそろえる）
torch.autocast = lambda *a, **k: contextlib.nullcontext()


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main() -> None:
    from depth_anything_3.api import DepthAnything3

    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    n, t0, t1, which, mode = int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6], sys.argv[7]
    pairs = pick_evenly(associate(read_list(src / "rgb.txt"), read_list(src / "depth.txt")), n, t0, t1)
    gt = np.array([[float(x) for x in line.split()] for line in (src / "groundtruth.txt").read_text().splitlines() if not line.startswith("#")])
    imgs = [np.asarray(Image.open(src / rgb).convert("RGB")) for _, rgb, _ in pairs]
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], np.float32)
    kw = {}
    if mode == "pose":
        w2c = []
        for t, _, _ in pairs:
            k = int(np.argmin(np.abs(gt[:, 0] - t)))
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = quat_to_rot(gt[k, 4:8]), gt[k, 1:4]
            w2c.append(np.linalg.inv(c2w))
        kw = {"extrinsics": np.stack(w2c).astype(np.float32), "intrinsics": np.stack([K] * len(pairs))}
    elif which == "metric":
        kw = {"intrinsics": np.stack([K] * len(pairs))}
    model = DepthAnything3.from_pretrained(REPO[which]).to(torch.device("cpu")).eval()
    t = time.time()
    if which == "metric":   # 1 枚ずつ（写真どうしをつなぐ力は無い）
        preds = [model.inference([im]) for im in imgs]
        depth = np.concatenate([p.depth for p in preds])
        conf = np.concatenate([p.conf if p.conf is not None else np.ones_like(p.depth) for p in preds])
        proc = np.concatenate([p.processed_images for p in preds])
    else:
        pr = model.inference(imgs, **kw)
        depth, conf, proc = pr.depth, (pr.conf if pr.conf is not None else np.ones_like(pr.depth)), pr.processed_images
    sec = round(time.time() - t, 1)
    N, H, W = depth.shape
    sx, sy = W / imgs[0].shape[1], H / imgs[0].shape[0]
    s = max(sx, sy)                                    # 縦横比を保って縮小し、はみ出しを中央で切る（tum_eval と同じ前提）
    Km = np.array([[FX * s, 0, CX * s - (imgs[0].shape[1] * s - W) / 2], [0, FY * s, CY * s - (imgs[0].shape[0] * s - H) / 2], [0, 0, 1]])
    if which == "metric":
        depth = depth * Km[0, 0] / 300.0              # 公式の説明どおり実寸へ（焦点距離は処理後の画像の画素で）
        c2ws = []
        for t_, _, _ in pairs:                         # 1 枚ずつなので位置は無い。評価のため実測の位置に置く（大きさの比較だけに使う）
            k = int(np.argmin(np.abs(gt[:, 0] - t_)))
            c = np.eye(4)
            c[:3, :3], c[:3, 3] = quat_to_rot(gt[k, 4:8]), gt[k, 1:4]
            c2ws.append(c)
        c2ws = np.stack(c2ws)
    else:
        e = pr.extrinsics
        e4 = np.tile(np.eye(4), (N, 1, 1))
        e4[:, :3, :] = e[:, :3, :]
        c2ws = np.linalg.inv(e4)
        if pr.intrinsics is not None:
            Km = None
    pts, cols, cf, cam, bits = [], [], [], [], []
    for v in range(N):
        Kv = Km if Km is not None else pr.intrinsics[v]
        u, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        z = depth[v]
        pc = np.stack([(u - Kv[0, 2]) / Kv[0, 0] * z, (vv - Kv[1, 2]) / Kv[1, 1] * z, z], -1)
        pw = pc @ c2ws[v][:3, :3].T + c2ws[v][:3, 3]
        ok = np.isfinite(pw).all(-1) & (z > 0) & (conf[v] >= np.percentile(conf[v], 10))
        pts.append(pw[ok].astype(np.float32)); cols.append(proc[v][ok].astype(np.uint8)); cf.append(conf[v][ok].astype(np.float32))
        cam.append(np.full(int(ok.sum()), v, np.int32)); bits.append(np.packbits(ok.ravel()))
    raw = {"points": np.concatenate(pts), "colors": np.concatenate(cols), "conf": np.concatenate(cf), "cam_index": np.concatenate(cam),
           "cam2world": c2ws.astype(np.float64), "intrinsics": np.stack([Km if Km is not None else pr.intrinsics[v] for v in range(N)]).astype(np.float64),
           "image_hw": np.array([[H, W]] * N), "valid_bits": np.concatenate(bits), "view_images": proc.astype(np.uint8),
           "meta": {"model": REPO[which], "mode": mode, "n_views": N, "sec_inference": sec, "device": "cpu"}}
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in raw.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(out / "raw.npz", **arrays, meta=json.dumps(raw["meta"]))
    print(f"{which}/{mode}: {N} 枚 {H}x{W} 推論 {sec} 秒", flush=True)


if __name__ == "__main__":
    main()
