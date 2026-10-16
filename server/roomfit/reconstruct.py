"""写真から点群とカメラ姿勢を推定する（MapAnything, CPU）。

- 使う重み: facebook/map-anything-apache（Apache 2.0 として公開されているもの）。
  CC BY-NC の facebook/map-anything は使わない。
- GPU は前提にしない。CUDA があれば使うが、無くても CPU で実行する。
- 出力の縮尺はモデルの推定値。実測で確認するまで「未補正」として扱う。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_ID = "facebook/map-anything-apache"
MAX_SIDE = 1024


def preprocess_images(src_paths: list[Path], out_dir: Path) -> list[Path]:
    """EXIF の向きを画素へ反映し、EXIF（位置情報など）を落として保存する。壊れた画像は例外にする。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, sp in enumerate(src_paths):
        with Image.open(sp) as im:
            im.load()
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > MAX_SIDE:
                r = MAX_SIDE / max(im.size)
                im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
            dp = out_dir / f"{i:03d}.jpg"
            im.save(dp, "JPEG", quality=92)
            out.append(dp)
    return out


def load_model():
    """モデルを読み込む。範囲に分けて何回か復元するとき、読み込みを 1 回で済ませるために分けてある。"""
    import torch
    from mapanything.models import MapAnything

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MapAnything.from_pretrained(MODEL_ID).to(device).eval()


def run_mapanything(image_paths: list[Path], on_stage=lambda s: None, focal_35mm: float | None = None, model=None, views=None) -> dict:
    """focal_35mm: 写真の EXIF にある 35mm 換算の焦点距離。渡すと、カメラの画角を推定に任せず既知として与える。
    model: load_model() で読み込み済みのモデル。渡さなければここで読み込む。
    views: 写真に加えてカメラ情報・位置・奥行きなどを付けた入力（mapanything の preprocess_inputs の結果）。渡すと image_paths の代わりに使う。"""
    import torch
    from mapanything.utils.image import load_images

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if model is None:
        on_stage("loading_model")
        model = load_model()
    t_load = time.time() - t0

    if views is not None:
        pass
    elif focal_35mm:
        import math

        from mapanything.utils.image import preprocess_inputs

        raw_views = []
        for p in image_paths:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB"))
            h, w = arr.shape[:2]
            f = focal_35mm / 43.267 * math.hypot(w, h)  # 35mm 換算は対角 43.267mm が基準
            K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=torch.float32)
            raw_views.append({"img": torch.from_numpy(arr.copy()), "intrinsics": K})
        views = preprocess_inputs(raw_views)
    else:
        views = load_images([str(p) for p in image_paths])
    on_stage("inferring")
    t1 = time.time()
    with torch.inference_mode():
        preds = model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=(device == "cuda"),
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )
    t_inf = time.time() - t1

    pts, cols, conf, cam_idx, poses, intr, sizes, masks, view_imgs = [], [], [], [], [], [], [], [], []
    for vi, pr in enumerate(preds):
        p3 = pr["pts3d"][0].float().cpu().numpy()            # (H,W,3) world
        img = pr["img_no_norm"][0].float().cpu().numpy()     # (H,W,3) 0..1
        cf = pr["conf"][0].float().cpu().numpy()             # (H,W)
        mk = pr["mask"][0].cpu().numpy().reshape(cf.shape).astype(bool)
        ok = mk & np.isfinite(p3).all(axis=-1)
        pts.append(p3[ok].astype(np.float32))
        cols.append((np.clip(img[ok], 0, 1) * 255).astype(np.uint8))
        conf.append(cf[ok].astype(np.float32))
        cam_idx.append(np.full(int(ok.sum()), vi, dtype=np.int32))
        poses.append(pr["camera_poses"][0].float().cpu().numpy())
        intr.append(pr["intrinsics"][0].float().cpu().numpy())
        sizes.append(list(cf.shape))
        masks.append(np.packbits(ok.ravel()))
        view_imgs.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
    msf = preds[0].get("metric_scaling_factor")
    return {
        "points": np.concatenate(pts),
        "colors": np.concatenate(cols),
        "conf": np.concatenate(conf),
        "cam_index": np.concatenate(cam_idx),
        "cam2world": np.stack(poses).astype(np.float64),
        "intrinsics": np.stack(intr).astype(np.float64),
        "image_hw": np.array(sizes),
        # view ごとの有効画素（行優先・ビット詰め）。points はこの順に並んでいる
        "valid_bits": np.concatenate(masks),
        # モデルが見た画像（view ごとに同じ大きさ）。写真の上で位置を選ぶ機能で使う
        "view_images": np.stack(view_imgs),
        "meta": {
            "model": MODEL_ID,
            "device": device,
            "n_views": len(image_paths),
            "sec_model_load": round(t_load, 1),
            "sec_inference": round(t_inf, 1),
            "metric_scaling_factor": None if msf is None else float(np.asarray(msf.float().cpu()).ravel()[0]),
            "focal_35mm_given": focal_35mm,
        },
    }


def save_raw(result: dict, path: Path) -> None:
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays, meta=json.dumps(result["meta"]))


def load_raw(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(str(z["meta"]))
    return out
