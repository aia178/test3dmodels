"""撮影アプリが保存した ARCore の深度（NNN.depth16、160×90、mm）と、MapAnything の奥行きを同じ画素で比べる（再推論なし）。

モデルの奥行き ÷ ARCore の深度 の中央値が、モデルの大きさの倍率の見積もりになる（1.04 なら、モデルは 4% 大きい）。
ARCore の深度は写真と同じ時刻・同じ向き（センサーの横長）。生の深度（rawdepth16、信頼度 conf8）は時刻がずれることがあるので、
姿勢の時刻差（rawPoseDiffMs）が小さいコマだけ別に数える。
深度の画素 → 元の写真の画素（中心で 12 倍）→ 縦向きに回した画素 → モデルに入った画像の画素（triangulate_scale.py と同じ写し方）。

使い方（server フォルダで）: uv run python ../experiments/arcore_depth_scale.py <ジョブのフォルダ> <元の記録のフォルダ>
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from roomfit.ar_views import build_views, plan_views  # noqa: E402
from roomfit.reconstruct import load_raw  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402

RAW_POSE_MS = 50


def read_u16(p, w, h):
    return np.frombuffer(p.read_bytes(), "<u2").reshape(h, w).astype(np.float64) / 1000.0


def model_z(raw, grids, a, uv_model):
    mask, xyz = grids[a]
    T = raw["cam2world"][a]
    i, j = np.round(uv_model[:, 1] - 0.5).astype(int), np.round(uv_model[:, 0] - 0.5).astype(int)
    ok = (i >= 0) & (i < mask.shape[0]) & (j >= 0) & (j < mask.shape[1])
    z = np.full(len(uv_model), np.nan)
    ok[ok] = mask[i[ok], j[ok]]
    z[ok] = ((xyz[i[ok], j[ok]] - T[:3, 3]) @ T[:3, :3])[:, 2]
    return z


def main():
    job = Path(sys.argv[1])
    cap = Path(sys.argv[2])   # 元の記録のフォルダ（サーバーは深度を取り出さない）
    raw = load_raw(job / "raw.npz")
    plan = plan_views(cap)
    assert plan["ids"] == raw["meta"]["ar_frames"]["ids"]
    with tempfile.TemporaryDirectory() as d:
        _, views = build_views(cap, Path(d), plan)
    grids = view_grids(raw)
    rows = {"depth": [], "raw": []}
    print("コマ  深度の比(中央値, 画素数)  生の深度の比(中央値, 画素数, 姿勢の時刻差 ms)")
    for a, (fid, K, v) in enumerate(zip(plan["ids"], plan["K"], views)):
        f = json.loads((cap / f"{fid}.json").read_text(encoding="utf-8"))
        Kp = v["intrinsics"][0].numpy().astype(np.float64)
        sx, sy = Kp[0, 0] / K[0, 0], Kp[1, 1] / K[1, 1]
        line = f"{fid}"
        for kind, name, conf_name in (("depth", "depth16", None), ("raw", "rawdepth16", "conf8")):
            w, h = (f["depthW"], f["depthH"]) if kind == "depth" else (f["rawW"], f["rawH"])
            D = read_u16(cap / f"{fid}.{name}", w, h)
            ok = D > 0.1
            if conf_name:
                C = np.frombuffer((cap / f"{fid}.{conf_name}").read_bytes(), np.uint8).reshape(h, w)
                ok &= C >= 128
            dy, dx = np.nonzero(ok)
            u, vv = (dx + 0.5) * f["w"] / w, (dy + 0.5) * f["h"] / h          # センサーの向きの写真の画素
            ur, vr = f["h"] - vv, u                                             # 縦向きに回した画素
            uvm = np.c_[sx * ur + Kp[0, 2] - sx * K[0, 2], sy * vr + Kp[1, 2] - sy * K[1, 2]]
            zm = model_z(raw, grids, a, uvm)
            r = zm / D[dy, dx]
            r = r[np.isfinite(r)]
            if kind == "raw" and f.get("rawPoseDiffMs", 0) > RAW_POSE_MS:
                line += f"   （生の深度は時刻が {f['rawPoseDiffMs']:.0f}ms ずれるので除外）"
                continue
            if len(r):
                rows[kind].append(r)
                line += f"   {np.median(r):.4f} ({len(r)})" + (f" {f.get('rawPoseDiffMs', 0):.0f}ms" if kind == "raw" else "")
        print(line)
    for kind, rs in rows.items():
        if rs:
            allr = np.concatenate(rs)
            print(f"{kind}: 全画素の中央値 {np.median(allr):.4f}（25〜75% {np.percentile(allr, 25):.4f}〜{np.percentile(allr, 75):.4f}）｜ コマごとの中央値の中央値 {np.median([np.median(x) for x in rs]):.4f}")


if __name__ == "__main__":
    main()
