"""保存済みの復元と同じ写真を、入力を変えて MapAnything で復元し直す（大きさの比較用）。

変え方:
- k: 写真ごとの内部パラメータ（scale_cases.py が保存した K。自宅は EXIF、公開データはデータセットの値）を与える
- flip: 写真を左右反転して与える（反転した結果の大きさを、元の復元と平均する案のため）
出力は experiments/ext/scale_bench/runs/<ケース名>_<変え方>/raw.npz。scale_cases.py と同じ形の cases ファイルも書く
（flip は画素格子を左右に戻してから保存するので、元の復元と同じ画素で比べられる）。
使い方（server フォルダで）: uv run python ../experiments/run_variant.py <ケース名> <k|flip>
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
sys.path.insert(0, str(here))
from roomfit.reconstruct import run_mapanything, save_raw  # noqa: E402
from scale_cases import cam_grids  # noqa: E402

BENCH = here / "ext" / "scale_bench"


def main() -> None:
    name, how = sys.argv[1], sys.argv[2]
    if (BENCH / "cases" / f"{name}_{how}.npz").exists():
        print(f"{name}_{how}: 済み", flush=True)
        return
    c = np.load(BENCH / "cases" / f"{name}.npz")
    srcs = [Path(str(p)) for p in c["src"]]
    out = BENCH / "runs" / f"{name}_{how}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    paths = []
    for i, p in enumerate(srcs):
        with Image.open(p) as im:
            im = im.convert("RGB")
            if how == "flip":
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            dp = out / "images" / f"{i:03d}.png"
            im.save(dp)
            paths.append(dp)
    views = None
    if how == "k":
        from mapanything.utils.image import preprocess_inputs

        views = preprocess_inputs([{"img": torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).copy()),
                                    "intrinsics": torch.tensor(K, dtype=torch.float32)} for p, K in zip(paths, c["K"])])
    raw = run_mapanything(paths, views=views)
    save_raw(raw, out / "raw.npz")
    arrs = {k: c[k] for k in c.files if not k.startswith(("ma", "mask"))}
    arrs["ma_c2w"] = raw["cam2world"]
    for v, (mask, pc) in enumerate(cam_grids(raw, paths)):
        if how == "flip":
            mask, pc = mask[:, ::-1], pc[:, ::-1] * np.array([-1, 1, 1], np.float32)
        arrs[f"mask{v}"], arrs[f"ma{v}"] = np.ascontiguousarray(mask), np.ascontiguousarray(pc)
    np.savez_compressed(BENCH / "cases" / f"{name}_{how}.npz", **arrs)
    print(f"{name}_{how}: msf {raw['meta']['metric_scaling_factor']:.4f} 推論 {raw['meta']['sec_inference']} 秒", flush=True)


if __name__ == "__main__":
    main()
