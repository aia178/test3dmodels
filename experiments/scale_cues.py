"""大きさの手がかりの候補を、保存済みの復元から 1 表にまとめる（再推論なし）。

ケースごとに、正しい倍率（自宅は実測 4 か所、公開データは正解の奥行き）と並べて、次を出す:
- msf: MapAnything が出す metric_scaling_factor
- cam_h: 床からのカメラの高さ（中央値）、ceil: 天井高（観測できたときだけ）。製品の形状抽出（build_room_structure）で求める
出力は experiments/ext/scale_bench/cues.json。scale_bench.py の方法が読む。
使い方（server フォルダで）: uv run python ../experiments/scale_cues.py
"""
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
sys.path.insert(0, str(here))
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.structure import StructureParams, build_room_structure  # noqa: E402
from roomfit.worker import CONF_DROP_PERCENTILE  # noqa: E402
from scale_cases import HOME, PUBLIC  # noqa: E402

BENCH = here / "ext" / "scale_bench"


def cues(raw: dict) -> dict:
    pts, conf = raw["points"].astype(np.float64), raw["conf"]
    keep = conf >= np.percentile(conf, CONF_DROP_PERCENTILE)
    out = {"msf": raw["meta"].get("metric_scaling_factor")}
    try:
        res = build_room_structure(pts[keep], raw["cam2world"], raw["cam_index"][keep], StructureParams())
    except Exception as e:  # 床が見つからない復元もある
        return {**out, "error": str(e)[:80]}
    T = res.T_room_from_world
    cams = raw["cam2world"][:, :3, 3] @ T[:3, :3].T + T[:3, 3]
    out["cam_h"] = float(np.median(cams[:, 1]))
    out["ceil"] = res.ceiling_height if res.ceiling_observed else None
    return out


def main() -> None:
    report = json.loads((BENCH / "last_report.json").read_text(encoding="utf-8"))["none"]
    true_s = {r["case"]: 1 / (1 + r["size_err_pct"] / 100) for r in report["public"]}
    for name in report["home"]:
        res = json.loads((BENCH / "home" / name / "measure_results.json").read_text(encoding="utf-8"))
        true_s[name] = float(np.mean(res["scale_factors"]))
    rows = {}
    for name, d in [(k, v[1]) for k, v in PUBLIC.items()] + [(k, v) for k, v in HOME.items() if k != "home_1012"]:
        rows[name] = {"true_s": round(true_s[name], 4), **cues(load_raw(d / "raw.npz"))}
        print(name, rows[name], flush=True)
    (BENCH / "cues.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
