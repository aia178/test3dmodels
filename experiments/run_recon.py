"""実験: 写真フォルダ → MapAnything(CPU) → 生の復元結果(npz)。時間とメモリを記録する。

使い方（server フォルダで）:
  uv run python ../experiments/run_recon.py <写真フォルダ> <出力フォルダ> [最大枚数]
"""
import json
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import preprocess_images, run_mapanything, save_raw  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
out.mkdir(parents=True, exist_ok=True)
files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
if limit:
    step = max(1, len(files) // limit)
    files = files[::step][:limit]

peak = {"rss": 0}
stop = False


def watch():
    proc = psutil.Process()
    while not stop:
        peak["rss"] = max(peak["rss"], proc.memory_info().rss)
        time.sleep(0.5)


threading.Thread(target=watch, daemon=True).start()
t0 = time.time()
imgs = preprocess_images(files, out / "images")
res = run_mapanything(imgs, on_stage=lambda s: print("stage:", s, flush=True))
stop = True
res["meta"]["sec_total"] = round(time.time() - t0, 1)
res["meta"]["peak_rss_gb"] = round(peak["rss"] / 2**30, 2)
res["meta"]["n_points"] = int(len(res["points"]))
save_raw(res, out / "raw.npz")
(out / "meta.json").write_text(json.dumps(res["meta"], indent=2), encoding="utf-8")
print(json.dumps(res["meta"], indent=2))
