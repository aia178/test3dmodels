"""MoGe（単眼の実寸の奥行き、MIT）を、scale_cases.py が取り出した写真に 1 枚ずつ当て、モデルの画素格子での点（カメラ座標・m）を保存する。

同じ元の写真は 1 回だけ推定する（複数の復元で同じコマを使うため）。fov を与える版（製品では EXIF・AR の内部パラメータ）と与えない版。
使い方（experiments/ext/mogeenv の python で）: python moge_infer.py <fov|free> [トークン数] [ケース名 ...]
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

here = Path(__file__).resolve().parent
BENCH = here / "ext" / "scale_bench"
MODEL = "Ruicheng/moge-2-vitl-normal"
TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None  # None は MoGe の既定（最も細かい）
torch.set_num_threads(8)


def model_grid_to_source(hm, wm, hs, ws):
    s = max(wm / ws, hm / hs)
    ox, oy = (ws * s - wm) / 2, (hs * s - hm) / 2
    u, v = np.meshgrid(np.arange(wm) + 0.5, np.arange(hm) + 0.5)
    return (u + ox) / s, (v + oy) / s


def main() -> None:
    mode = sys.argv[1]
    names = [a for a in sys.argv[2:] if not a.isdigit()] or [p.stem for p in sorted((BENCH / "cases").glob("*.npz"))]
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(MODEL).eval()
    per_image = BENCH / "moge2_images" / mode
    per_image.mkdir(parents=True, exist_ok=True)
    (BENCH / f"moge2_{mode}").mkdir(exist_ok=True)
    for name in names:
        c = np.load(BENCH / "cases" / f"{name}.npz")
        out = {}
        for v in range(int(c["n"])):
            src, K = Path(str(c["src"][v])), c["K"][v]
            hm, wm = c[f"mask{v}"].shape
            key = hashlib.sha1(src.read_bytes() + f"|{hm}x{wm}|{TOKENS}".encode()).hexdigest()[:16]
            f = per_image / f"{key}.npy"
            if not f.exists():
                img = np.asarray(Image.open(src).convert("RGB"))
                hs, ws = img.shape[:2]
                t0 = time.time()
                with torch.inference_mode():
                    kw = {"fov_x": math.degrees(2 * math.atan(ws / 2 / K[0, 0]))} if mode == "fov" else {}
                    r = model.infer(torch.from_numpy(img).permute(2, 0, 1).float() / 255, use_fp16=False, num_tokens=TOKENS, **kw)
                pts = r["points"].numpy()
                pts[~r["mask"].numpy().astype(bool)] = np.nan
                xs, ys = model_grid_to_source(hm, wm, hs, ws)
                g = pts[np.clip(ys.astype(int), 0, hs - 1), np.clip(xs.astype(int), 0, ws - 1)]
                np.save(f, g.astype(np.float32))
                print(f"{name} {v}: {time.time() - t0:.1f} 秒", flush=True)
            out[f"pts{v}"] = np.load(f)
        np.savez_compressed(BENCH / f"moge2_{mode}" / f"{name}.npz", **out)
        print(f"== {name}", flush=True)
    (BENCH / f"moge2_{mode}" / "model.json").write_text(json.dumps({"model": MODEL}), encoding="utf-8")


if __name__ == "__main__":
    main()
