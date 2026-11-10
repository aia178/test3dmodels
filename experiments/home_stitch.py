"""実験: 自宅の一角（13 枚）を 2 つの範囲に分けて復元したものを、アプリ本体の処理（roomfit.stitch）でつなぐ。

高解像度・静止の写真で、つないだ結果がメジャーの実測とどれだけ合うかを見る（measure_home_1012.py で測る）。
比べる相手は、同じ 13 枚を 1 回で復元した結果（out/home_1012）。
使い方（server フォルダで）: uv run python ../experiments/home_stitch.py <範囲 A の出力> <範囲 B の出力> <つないだ出力> [共通の枚数=3]
範囲 A, B は run_subset.py の出力（A = 0..7, B = 5..12 のように、A の最後と B の最初が同じ写真）。
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import load_raw, save_raw  # noqa: E402
from roomfit.stitch import merge_raws  # noqa: E402
from roomfit.worker import capture_quality, structure_from_raw  # noqa: E402

a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
shared = int(sys.argv[4]) if len(sys.argv) > 4 else 3
ma, mb = (json.loads((d / "viewmap.json").read_text()) for d in (a, b))
olds = sorted(int(k) for k in ma) + sorted(int(k) for k in mb)[shared:]      # つないだ後の写真の並び（元の番号）
assert sorted(int(k) for k in ma)[-shared:] == sorted(int(k) for k in mb)[:shared], "A の最後と B の最初が同じ写真になっていません"
merged = merge_raws([load_raw(a / "raw.npz"), load_raw(b / "raw.npz")], shared)
(out / "images").mkdir(parents=True, exist_ok=True)
for new, old in enumerate(olds):
    src = (a / "images" / f"{ma[str(old)]:03d}.jpg") if str(old) in ma else (b / "images" / f"{mb[str(old)]:03d}.jpg")
    shutil.copyfile(src, out / "images" / f"{new:03d}.jpg")
(out / "viewmap.json").write_text(json.dumps({str(old): new for new, old in enumerate(olds)}), encoding="utf-8")
save_raw(merged, out / "raw.npz")
room = structure_from_raw(merged, out)
print("つなぎ:", merged["meta"]["chunks"]["links"])
print("質:", {"A": capture_quality(load_raw(a / "raw.npz")), "B": capture_quality(load_raw(b / "raw.npz")), "つないだ後": room["capture_quality"]})
print("天井高:", room["ceiling_height"], " 壁:", room["edge_status"])
