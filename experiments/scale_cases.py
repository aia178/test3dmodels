"""大きさの推定を比べるための材料を、保存済みの復元（MapAnything）から取り出す。再推論はしない。

1 つの復元（case）ごとに、写真ごとの「モデルが見た画素格子」での次のものを experiments/ext/scale_bench/cases/<名前>.npz に書く。
- ma: MapAnything の点（その写真のカメラ座標・m）、mask: 有効な画素
- src: 元の写真のパス、K: 元の写真の内部パラメータ（製品では EXIF や AR から得られるもの）
- gt: 正解の点（カメラ座標・m。公開データだけ。自宅は無し）
自宅の復元は、measure_home_1012.py の結果（scale_bench/home/<名前>/measure_results.json）を別に使う。

公開データは、保存された写真を元の連続画像と画素で突き合わせて、どのコマかを決める（復元のときの選び方の引数に頼らない）。
使い方（server フォルダで）: uv run python ../experiments/scale_cases.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
sys.path.insert(0, str(here))
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402
from tum_eval import model_grid_to_source  # noqa: E402

OUT = here / "out"
BENCH = here / "ext" / "scale_bench"
HOME_PHOTO_F35 = 23.0  # 自宅の写真の EXIF（35mm 換算、13 枚とも同じ）

PUBLIC = {
    "tum_even": ("tum", OUT / "frame_select" / "tum_even_photo"),
    "tum_overlap": ("tum", OUT / "frame_select" / "tum_overlap_photo"),
    "tum_mixed": ("tum", OUT / "frame_select" / "tum_mixed_photo"),
    "tum_loop12": ("tum", OUT / "tum_room_12"),
    "tum_dense10": ("tum", OUT / "tum_room_dense10"),
    "tum_half16": ("tum", OUT / "tum_room_half16"),
    "tum_loop20": ("tum", OUT / "tum_room_loop20"),
    "arkit_even": ("arkit", OUT / "frame_select" / "arkit_even_photo"),
    "arkit_overlap": ("arkit", OUT / "frame_select" / "arkit_overlap_photo"),
    "arkit_mixed": ("arkit", OUT / "frame_select" / "arkit_mixed_photo"),
    "arkit_w30": ("arkit", OUT / "arkit_42444946_w30_photo"),  # arkit_42444946_16_photo は arkit_even と同じコマなので入れない
}
# 正解のカメラの位置（と内部パラメータ）を MapAnything に渡した復元（§15・§20）。製品で AR の位置を渡す場合に近い。
# *_noisy は位置に誤差（大きさ 3% のずれ・コマごとに 2cm と 1°・1 秒あたり 5mm）を足して渡したもの
POSED = {
    "tum_even_pose": ("tum", OUT / "frame_select" / "tum_even_pose"),
    "tum_overlap_pose": ("tum", OUT / "frame_select" / "tum_overlap_pose"),
    "tum_mixed_pose": ("tum", OUT / "frame_select" / "tum_mixed_pose"),
    "tum_half16_pose": ("tum", OUT / "tum_half16_pose"),
    "tum_loop20_pose": ("tum", OUT / "tum_loop20_pose"),
    "tum_half16_pose_noisy": ("tum", OUT / "tum_half16_pose_noisy"),
    "tum_loop20_pose_noisy": ("tum", OUT / "tum_loop20_pose_noisy"),
    "arkit_even_pose": ("arkit", OUT / "frame_select" / "arkit_even_pose"),
    "arkit_overlap_pose": ("arkit", OUT / "frame_select" / "arkit_overlap_pose"),
    "arkit_mixed_pose": ("arkit", OUT / "frame_select" / "arkit_mixed_pose"),
    "arkit_w30_pose": ("arkit", OUT / "arkit_42444946_w30_pose"),
}
HOME = {
    "home_avg": OUT / "home_1012_avg",
    "home_1012": OUT / "home_1012",
    "home_n4": OUT / "home_1012_n4",
    "home_n6": OUT / "home_1012_n6",
    "home_n8": OUT / "home_1012_n8",
}


def thumb(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((80, 60), Image.BILINEAR), dtype=np.float32)


def source_frames(kind: str) -> list[dict]:
    from frame_select import load_arkit, load_tum

    return load_tum(step=1) if kind == "tum" else load_arkit()


def match_frames(images: list[Path], frames: list[dict]) -> list[dict]:
    th = np.stack([thumb(f["rgb"]) for f in frames])
    out = []
    for p in images:
        d = np.abs(th - thumb(p)).mean(axis=(1, 2))
        k = int(np.argmin(d))
        if d[k] > 1.0:
            raise RuntimeError(f"{p}: 元のコマが見つかりません（差 {d[k]:.2f}）")
        out.append(frames[k])
    return out


def world_grids(raw: dict, images: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
    """古い復元（valid_bits が無い）は、measure_home_1012.py と同じく色の突き合わせで画素格子へ戻す。"""
    if "valid_bits" in raw:
        return view_grids(raw)
    from gridtools import model_input_images, recover_grid

    imgs = model_input_images(images)
    return [(g["valid"], g["xyz"]) for g in (recover_grid(raw, v, imgs[v]) for v in range(len(raw["cam2world"])))]


def cam_grids(raw: dict, images: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for v, (mask, xyz) in enumerate(world_grids(raw, images)):
        T = raw["cam2world"][v]
        out.append((mask, ((xyz.astype(np.float64) - T[:3, 3]) @ T[:3, :3]).astype(np.float32)))
    return out


def gt_grid(fr: dict, kind: str, hm: int, wm: int) -> np.ndarray:
    d = np.asarray(Image.open(fr["depth"])).astype(np.float64) / fr["depth_scale"]
    w, h = fr["wh"]
    xs, ys = model_grid_to_source(hm, wm, h, w)
    sx, sy = d.shape[1] / w, d.shape[0] / h
    z = d[np.clip((ys * sy).astype(int), 0, d.shape[0] - 1), np.clip((xs * sx).astype(int), 0, d.shape[1] - 1)]
    lo, hi = (0.3, 4.0) if kind == "tum" else (0.2, 8.0)
    z[(z < lo) | (z > hi)] = np.nan
    K = fr["K"]
    return np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], -1).astype(np.float32)


def save_case(name: str, raw: dict, srcs: list[Path], Ks: list[np.ndarray], gts: list[np.ndarray] | None, gt_c2w: list[np.ndarray] | None = None) -> None:
    grids = cam_grids(raw, srcs)
    arrs = {"n": np.array(len(grids)), "src": np.array([str(p) for p in srcs]), "K": np.stack(Ks), "ma_c2w": raw["cam2world"]}
    if gt_c2w is not None:
        arrs["gt_c2w"] = np.stack(gt_c2w)
    for v, (mask, pc) in enumerate(grids):
        arrs[f"mask{v}"], arrs[f"ma{v}"] = mask, pc
        if gts is not None:
            arrs[f"gt{v}"] = gts[v]
    (BENCH / "cases").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BENCH / "cases" / f"{name}.npz", **arrs)
    print(f"{name}: {len(grids)} 枚", flush=True)


def main() -> None:
    cache = {}
    for name, (kind, d) in {**PUBLIC, **POSED}.items():
        if (BENCH / "cases" / f"{name}.npz").exists() and "gt_c2w" in np.load(BENCH / "cases" / f"{name}.npz"):
            continue
        if kind not in cache:
            cache[kind] = source_frames(kind)
        raw = load_raw(d / "raw.npz")
        imgs = sorted((d / "images").glob("*.png"))
        frames = match_frames(imgs, cache[kind])
        hm, wm = (int(x) for x in raw["image_hw"][0])
        save_case(name, raw, [f["rgb"] for f in frames], [f["K"] for f in frames], [gt_grid(f, kind, hm, wm) for f in frames],
                  [f["c2w"] for f in frames])
    for name, d in HOME.items():
        if (BENCH / "cases" / f"{name}.npz").exists() and "ma_c2w" in np.load(BENCH / "cases" / f"{name}.npz"):
            continue
        raw = load_raw(d / "raw.npz")
        imgs = sorted((d / "images").glob("*.jpg"))
        Ks = []
        for p in imgs:
            with Image.open(p) as im:
                w, h = im.size
            f = HOME_PHOTO_F35 / 43.267 * math.hypot(w, h)
            Ks.append(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]))
        save_case(name, raw, imgs, Ks, None)
    (BENCH / "cases" / "index.json").write_text(json.dumps({"public": {k: v[0] for k, v in PUBLIC.items()}, "posed": {k: v[0] for k, v in POSED.items()},
                                                            "home": list(HOME)}), encoding="utf-8")


if __name__ == "__main__":
    main()
