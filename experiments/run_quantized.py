"""実験: モデルの線形層を 8 ビットに量子化（動的量子化）すると、CPU での推論が速くなるか・精度が保てるか。
使い方（server フォルダで）: uv run python ../experiments/run_quantized.py <元の出力フォルダ> <出力フォルダ> <写真番号 例: 0,5,11,12> [threads]
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit import reconstruct  # noqa: E402
from roomfit.structure import StructureError  # noqa: E402
from roomfit.worker import structure_from_raw  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
idx = [int(x) for x in sys.argv[3].split(",")]
if len(sys.argv) > 4:
    torch.set_num_threads(int(sys.argv[4]))
(out / "images").mkdir(parents=True, exist_ok=True)
paths = []
for new, old in enumerate(idx):
    p = out / "images" / f"{new:03d}.jpg"
    shutil.copyfile(src / "images" / f"{old:03d}.jpg", p)
    paths.append(p)
(out / "viewmap.json").write_text(json.dumps({str(old): new for new, old in enumerate(idx)}), encoding="utf-8")

from mapanything.models import MapAnything  # noqa: E402

_orig = MapAnything.from_pretrained.__func__


def _quantized(cls, *a, **k):
    m = _orig(cls, *a, **k)
    n_lin = sum(1 for x in m.modules() if isinstance(x, torch.nn.Linear))
    t = time.time()
    q = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    print(f"quantized {n_lin} linear layers in {time.time() - t:.1f}s", flush=True)
    q.to = lambda *aa, **kk: q   # 量子化後は device の移動をしない（CPU 専用）
    return q


MapAnything.from_pretrained = classmethod(_quantized)
t0 = time.time()
raw = reconstruct.run_mapanything(paths)
raw["meta"]["sec_total"] = round(time.time() - t0, 1)
raw["meta"]["quantized"] = True
raw["meta"]["threads"] = torch.get_num_threads()
reconstruct.save_raw(raw, out / "raw.npz")
(out / "meta.json").write_text(json.dumps(raw["meta"], indent=1), encoding="utf-8")
try:
    room = structure_from_raw(raw, out)
    print(f"{len(idx)} photos quantized: inference {raw['meta']['sec_inference']}s, quality {room['capture_quality']['level']} {room['capture_quality']['disagreement_mm']}mm", flush=True)
except StructureError as e:
    print(f"STRUCTURE FAILED: {e.code} {e.message}", flush=True)
