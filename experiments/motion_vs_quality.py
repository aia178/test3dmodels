"""調査: 写真 1 枚ごとにカメラをどれだけ回したかと、復元の質（重なり・写真どうしの食い違い）の関係。

撮り方の案内に「1 枚ごとに回す角度の目安」を入れられるかを見る。復元はやり直さず、既存の raw.npz を読むだけ。
回した角度は、復元が出したカメラの向き（cam2world の回転）から取る。位置は当てにならない（検証記録 §10）ので使わない。
使い方（server フォルダで）: uv run python ../experiments/motion_vs_quality.py <出力フォルダ> [<出力フォルダ> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.worker import capture_quality  # noqa: E402


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


rows = []
for d in map(Path, sys.argv[1:]):
    if not (d / "raw.npz").exists():
        continue
    raw = load_raw(d / "raw.npz")
    R = raw["cam2world"][:, :3, :3]
    n = len(R)
    steps = [angle_deg(R[i], R[i + 1]) for i in range(n - 1)]                               # 撮った順の、隣どうしの角度
    nearest = [min(angle_deg(R[i], R[j]) for j in range(n) if j != i) for i in range(n)]    # 撮った順によらない、いちばん近い写真との角度
    q = capture_quality(raw)
    rows.append({"name": d.name, "n": n, "step_median": round(float(np.median(steps)), 1), "step_max": round(max(steps), 1),
                 "nearest_median": round(float(np.median(nearest)), 1), "nearest_max": round(max(nearest), 1),
                 "overlap": q["overlap"], "disagreement_mm": q["disagreement_mm"], "level": q["level"]})
rows.sort(key=lambda r: r["step_median"])
print(f"{'name':24s} {'n':>3s} {'step_med':>8s} {'step_max':>8s} {'near_med':>8s} {'near_max':>8s} {'overlap':>7s} {'disag':>6s} level")
for r in rows:
    print(f"{r['name']:24s} {r['n']:3d} {r['step_median']:8.1f} {r['step_max']:8.1f} {r['nearest_median']:8.1f} {r['nearest_max']:8.1f} {r['overlap']:7.3f} {r['disagreement_mm']:6.1f} {r['level']}")
Path(__file__).resolve().parent.joinpath("out", "motion_vs_quality.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
