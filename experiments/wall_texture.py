"""試作: 復元で分かった壁の平面に、元の高画質の写真を投影して貼る（壁 1 面ぶんの画像を作る）。

点の集まりの表示は、AI が写真を 392×518 画素に縮めて計算するため、ぼやけてのっぺり見える（2026-10-16）。
部屋の大部分は平面なので、壁の位置だけ復元から取り、見た目は元の写真（3072×4096）を貼る。壁は平面なので扉の膨らみも出ない。
写真の画素と 3D の対応は、モデル自身のカメラ（位置・内部パラメータ）で投影するので、復元の点と食い違わない。

使い方（server フォルダで）: uv run python ../experiments/wall_texture.py <復元の出力（room.json, raw.npz, images/）> <元の写真のフォルダ> <出力フォルダ>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import load_raw  # noqa: E402

RES = 0.002   # 壁の画像の 1 画素 = 2mm


def main() -> None:
    out_dir, photo_dir, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dst.mkdir(parents=True, exist_ok=True)
    room = json.loads((out_dir / "room.json").read_text(encoding="utf-8"))
    raw = load_raw(out_dir / "raw.npz")
    T = np.array(room["T_room_from_world"]); Tinv = np.linalg.inv(T)
    photos = sorted(photo_dir.glob("*.jpg"))
    pre = sorted((out_dir / "images").glob("*.jpg"))
    hm, wm = raw["image_hw"][0]
    big = [np.asarray(Image.open(p).convert("RGB")) for p in photos]
    Hs, Ws = Image.open(pre[0]).size[1], Image.open(pre[0]).size[0]          # 前処理後（1024 の側）
    Hb, Wb = big[0].shape[:2]
    k_big = Wb / Ws                                                         # 前処理後 → 元の写真
    s = max(wm / Ws, hm / Hs); ox, oy = (Ws * s - wm) / 2, (Hs * s - hm) / 2  # モデルの画素 ← 前処理後（縦横比を保って縮小し中央で切る）
    ceil = room["ceiling_height"] or 2.5
    O = np.array(room["outline"])
    for e, st in enumerate(room["edge_status"]):
        if st != "observed":
            continue
        a, b = O[e], O[(e + 1) % len(O)]
        L = float(np.linalg.norm(b - a)); t = (b - a) / L
        n = np.array([t[1], 0.0, -t[0]])                                    # 壁の法線（部屋の内側を向くように後で符号を決める）
        nu, nv = int(L / RES), int(ceil / RES)
        uu, vv = np.meshgrid((np.arange(nu) + 0.5) * RES, ceil - (np.arange(nv) + 0.5) * RES)
        P = np.stack([a[0] + t[0] * uu, vv, a[1] + t[1] * uu], -1).reshape(-1, 3)
        W = P @ Tinv[:3, :3].T + Tinv[:3, 3]
        # まず写真ごとに、この壁をどれだけ広く・正面から写しているかを見て、主に使う 1 枚を決める
        per = []
        for v in range(len(big)):
            c2w = raw["cam2world"][v]; K = raw["intrinsics"][v]
            pc = (W - c2w[:3, 3]) @ c2w[:3, :3]
            z = pc[:, 2]
            um = K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
            vm = K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
            xb, yb = (um + ox) / s * k_big, (vm + oy) / s * k_big           # 元の写真の画素
            inside = (z > 0.2) & (xb >= 0) & (xb < Wb - 1) & (yb >= 0) & (yb < Hb - 1)
            cam_room = T[:3, :3] @ c2w[:3, 3] + T[:3, 3]
            ray = P - cam_room; ray /= np.linalg.norm(ray, axis=1, keepdims=True)
            nn = n if np.dot(n, cam_room - P.mean(0)) > 0 else -n
            frontal = -(ray @ nn)                                           # 1 = 正面から見ている
            edge = np.minimum.reduce([xb / Wb, 1 - xb / Wb, yb / Hb, 1 - yb / Hb])
            score = np.where(inside, frontal + 0.5 * np.clip(edge, 0, 0.1), -1)
            per.append((v, score, xb, yb))
        order = sorted(per, key=lambda q: -float(np.where(q[1] > 0, q[1], 0).sum()))   # 広く正面から写している順
        img = np.zeros((len(P), 3), np.float64); done = np.zeros(len(P), bool); ref = None
        for v, score, xb, yb in order:
            take = (score > 0.35) & ~done
            if not take.any():
                continue
            col = big[v][yb.astype(int).clip(0, Hb - 1), xb.astype(int).clip(0, Wb - 1)].astype(np.float64)
            if ref is None:
                ref = col
            else:   # 主の写真と両方に写っている所で、色の倍率を合わせる
                both = done & (score > 0.35)
                if both.sum() > 500:
                    col *= np.median(img[both], axis=0) / np.maximum(np.median(col[both], axis=0), 1)
            img[take] = col[take]; done |= take
        im = np.clip(img, 0, 255).astype(np.uint8).reshape(nv, nu, 3)
        im[~done.reshape(nv, nu)] = 235
        Image.fromarray(im).save(dst / f"wall_{e}.jpg", quality=90)
        print(f"壁 {e}: 幅 {L:.2f}m × 高さ {ceil:.2f}m → {nu}×{nv} 画素", flush=True)


if __name__ == "__main__":
    main()
