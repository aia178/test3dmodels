"""実験（ARKitScenes）: 本物のスマホの AR（iPad Pro の ARKit）の位置をモデルへ渡すと、どこまで良くなるか。

TUM の実験（docs/検証記録.md §15）は、正確な位置にわざと誤差を足して「スマホ並み」を模した。ここでは本物の AR の位置を使う。
答え合わせ: レーザースキャナーから作った正解の奥行き（highres_depth、1920×1440）。
  - 写真ごとの大きさ（モデルの奥行き ÷ 正解の奥行き）は、AR の位置と無関係に測れる。
  - 写真をまたぐ 2 点の距離は、正解の奥行きを AR の位置で部屋の座標へ置いて作るので、AR の位置の誤差が答えの側にも入る（参考値）。
データ: ARKitScenes（Apple、研究・非商用の利用条件。製品には使わない）。experiments/data/arkitscenes/<video_id>/ に
  vga_wide/（640×480）、vga_wide_intrinsics/、lowres_wide.traj（世界→カメラ、OpenCV の向き。§16 で確認）、highres_depth/ を置く。

使い方（server フォルダで）:
  uv run python ../experiments/arkit_posed.py <データのフォルダ> <出力> <枚数> <photo|pose> [開始秒=0] [終了秒=全体]
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))


def load_frames(src: Path):
    tr = np.loadtxt(src / "lowres_wide.traj")
    hd = sorted((src / "highres_depth").glob("*.png"))
    out = []
    for f in hd:
        ts = float(f.stem.split("_")[1])
        i = int(np.argmin(np.abs(tr[:, 0] - ts)))
        if abs(tr[i, 0] - ts) > 0.06:
            continue
        R, _ = cv2.Rodrigues(tr[i, 1:4])
        w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = tr[i, 4:7]
        k = np.loadtxt(src / "vga_wide_intrinsics" / f"{f.stem}.pincam")
        out.append({"ts": ts, "rgb": src / "vga_wide" / f"{f.stem}.png", "depth": f, "c2w": np.linalg.inv(w2c),
                    "K": np.array([[k[2], 0, k[4]], [0, k[3], k[5]], [0, 0, 1.0]])})
    return out


def pick(frames, n, t0, t1):
    s = frames[0]["ts"]
    sel = [f for f in frames if f["ts"] - s >= t0 and (t1 is None or f["ts"] - s <= t1)]
    idx = np.round(np.linspace(0, len(sel) - 1, n)).astype(int)
    return [sel[i] for i in idx]


def evaluate(raw: dict, frames: list) -> dict:
    from roomfit.worker import view_grids

    rng = np.random.default_rng(0)
    rows, P_all, Q_all, V_all = [], [], [], []
    for v, ((mask, xyz), fr) in enumerate(zip(view_grids(raw), frames)):
        T = raw["cam2world"][v]
        pc = (xyz.astype(np.float64) - T[:3, 3]) @ T[:3, :3]
        hm, wm = mask.shape
        gt_full = np.asarray(Image.open(fr["depth"])).astype(np.float64) / 1000.0     # 1920×1440
        Hs, Ws = 480, 640
        s = max(wm / Ws, hm / Hs)
        ox, oy = (Ws * s - wm) / 2, (Hs * s - hm) / 2
        u, vv = np.meshgrid(np.arange(wm) + 0.5, np.arange(hm) + 0.5)
        xs, ys = (u + ox) / s, (vv + oy) / s                                         # 640×480 の画素
        gt = gt_full[np.clip((ys * 3).astype(int), 0, 1439), np.clip((xs * 3).astype(int), 0, 1919)]
        ok = mask & (gt > 0.2) & (gt < 8) & np.isfinite(pc).all(-1) & (pc[..., 2] > 0.05)
        if ok.sum() < 2000:
            rows.append({"view": v, "n": int(ok.sum())})
            continue
        r = pc[..., 2][ok] / gt[ok]
        rows.append({"view": v, "scale": round(float(np.median(r)), 4), "shape_err_pct": round(float(np.median(np.abs(r / np.median(r) - 1))) * 100, 2),
                     "depth_m": round(float(np.median(gt[ok])), 2)})
        K = fr["K"]
        Qc = np.stack([(xs - K[0, 2]) / K[0, 0] * gt, (ys - K[1, 2]) / K[1, 1] * gt, gt], -1)
        idx = np.argwhere(ok)
        sel = idx[rng.integers(0, len(idx), 3000)]
        P_all.append(xyz[sel[:, 0], sel[:, 1]].astype(np.float64))
        Q_all.append(Qc[sel[:, 0], sel[:, 1]] @ fr["c2w"][:3, :3].T + fr["c2w"][:3, 3])
        V_all.append(np.full(len(sel), v))
    scales = np.array([x["scale"] for x in rows if "scale" in x])
    res = {"views": rows, "scale_median": round(float(np.median(scales)), 4), "scale_spread_pct": round(float((scales.max() - scales.min()) / scales.mean() * 100), 1),
           "scale_iqr_pct": round(float((np.percentile(scales, 75) - np.percentile(scales, 25)) / np.median(scales) * 100), 1)}
    P, Q, V = map(np.concatenate, (P_all, Q_all, V_all))
    i, j = rng.integers(0, len(P), 400000), rng.integers(0, len(P), 400000)
    dp, dq = np.linalg.norm(P[i] - P[j], axis=1), np.linalg.norm(Q[i] - Q[j], axis=1)
    res["room_extent_m"] = [round(float(x), 2) for x in np.percentile(Q, 99, axis=0) - np.percentile(Q, 1, axis=0)]
    for name, m in (("same_view_>0.5m", (V[i] == V[j]) & (dq > 0.5)), ("cross_0.5-1.5m", (V[i] != V[j]) & (dq > 0.5) & (dq <= 1.5)),
                    ("cross_1.5-3m", (V[i] != V[j]) & (dq > 1.5) & (dq <= 3)), ("cross_>3m", (V[i] != V[j]) & (dq > 3))):
        if m.sum() < 500:
            continue
        rr = dp[m] / dq[m]
        res[name] = {"pairs": int(m.sum()), "error_pct": round((float(np.median(rr)) - 1) * 100, 1),
                     "p10_p90_pct": [round((float(np.percentile(rr, 10)) - 1) * 100, 1), round((float(np.percentile(rr, 90)) - 1) * 100, 1)],
                     "abs_error_cm_median": round(float(np.median(np.abs(dp[m] - dq[m]))) * 100, 1)}
    return res


def main() -> None:
    import torch
    from mapanything.utils.image import preprocess_inputs

    from roomfit.reconstruct import run_mapanything, save_raw
    from roomfit.worker import capture_quality

    src, out, n, mode = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    t0 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    t1 = float(sys.argv[6]) if len(sys.argv) > 6 else None
    frames = pick(load_frames(src), n, t0, t1)
    (out / "images").mkdir(parents=True, exist_ok=True)
    views, paths = [], []
    for i, fr in enumerate(frames):
        im = Image.open(fr["rgb"]).convert("RGB")
        p = out / "images" / f"{i:03d}.png"
        im.save(p)
        paths.append(p)
        view = {"img": torch.from_numpy(np.asarray(im).copy())}
        if mode == "pose":
            view["intrinsics"] = torch.from_numpy(fr["K"].astype(np.float32))
            view["camera_poses"] = torch.from_numpy(fr["c2w"].astype(np.float32))
            view["is_metric_scale"] = torch.tensor([True])
        views.append(view)
    print(f"{mode}: {len(frames)} 枚（{frames[-1]['ts'] - frames[0]['ts']:.1f} 秒）", flush=True)
    t = time.time()
    raw = run_mapanything(paths, views=preprocess_inputs(views)) if mode == "pose" else run_mapanything(paths)
    raw["meta"].update({"sec_total": round(time.time() - t, 1), "input_mode": mode, "frames": [f["rgb"].name for f in frames]})
    save_raw(raw, out / "raw.npz")
    res = evaluate(raw, frames)
    res["capture_quality"] = capture_quality(raw)
    res["sec_inference"] = raw["meta"]["sec_inference"]
    (out / "arkit_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "views"}, ensure_ascii=False), flush=True)
    print("写真ごとの大きさ:", [x.get("scale") for x in res["views"]], flush=True)


if __name__ == "__main__":
    main()
