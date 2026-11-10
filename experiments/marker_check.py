"""実験: 復元の出力フォルダ（raw.npz と images/）と元の写真から、印刷マーカーでの縮尺（roomfit.marker_scale）を出す。

サーバーのジョブ（写真だけ）では worker.add_marker_scale が自動で出す。これは、AR の記録から復元したもの（run_arcore_posed.py の出力）や、
既にある復元結果に対して確かめるための台本。メジャーの実測があれば、マーカーで合わせた後の長さと比べる。

使い方（server フォルダで）:
  uv run python ../experiments/marker_check.py <出力フォルダ> <元の写真のフォルダ> [--arcore]
    --arcore: 撮影アプリの記録（写真はセンサーの向きのまま保存されている）。run_arcore_posed.py と同じく時計回りに 90 度回して使い、
              framemap.json で元のコマ番号をたどる
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.marker_scale import estimate_scale  # noqa: E402
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.textures import ViewData, photo_mapping  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402


def main() -> None:
    out, src = Path(sys.argv[1]), Path(sys.argv[2])
    arcore = "--arcore" in sys.argv
    raw = load_raw(out / "raw.npz")
    pre = sorted((out / "images").glob("*.jpg"))
    if arcore:
        fm = json.loads((out / "framemap.json").read_text(encoding="utf-8"))
        old_of = {new: int(old) for old, new in fm.items()}
        photos = [src / f"{old_of[i]:03d}.jpg" for i in range(len(pre))]
    else:
        photos = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if "valid_bits" in raw:
        grids = [(m.shape, xyz.astype(np.float64)) for m, xyz in view_grids(raw)]
    else:   # 古い形式: 写真ごとの 3D は views/NNN.xyz（部屋座標、mm）にある（make_surfaces.py と同じ）
        from roomfit.worker import XYZ_INVALID

        grids = []
        for m in json.loads((out / "views" / "views.json").read_text(encoding="utf-8")):
            q = np.frombuffer((out / "views" / f"{m['n']:03d}.xyz").read_bytes(), "<i2").reshape(m["h"], m["w"], 3).astype(np.float64)
            q[(q == XYZ_INVALID).any(-1)] = np.nan
            grids.append(((m["h"], m["w"]), q / 1000.0))
    views = []
    for v, (hw, xyz) in enumerate(grids):
        with Image.open(photos[v]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if arcore:
                im = im.rotate(-90, expand=True)
            photo = np.asarray(im)
        with Image.open(pre[v]) as im:
            pw = im.size
        # 長さだけを見るので、部屋座標へ移さず世界座標のまま使う
        views.append(ViewData(cam2room=raw["cam2world"][v], K=raw["intrinsics"][v], grid_room=xyz, photo=photo,
                              to_photo=photo_mapping(hw, pw, (photo.shape[1], photo.shape[0]))))
    r = estimate_scale(views)
    if r is None:
        print("マーカーは見つからなかった")
        return
    for d in r["detections"]:
        print(f"  写真 {d['view']:2d} 番号 {d['id']}: " + (f"倍率 {d['factor']:.4f}（一辺 {d['side_m'] * 1000:.1f}mm、モデルの画像で {d['side_px']} 画素、当てはめの残り {d['fit_residual_mm']}mm）"
                                                     if "factor" in d else f"使わない（{d['reject']}）"))
    print(f"→ 倍率 {r['factor']}（{r['n']} か所、ばらつき {r.get('spread_pct')}%）")
    (out / "marker_scale.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
