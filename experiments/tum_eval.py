"""評価: 公開データ TUM RGB-D（深度センサーの実測つき）で、写真だけからの復元の縮尺と形を確かめる。

狙い（docs/検証記録.md §12 の「残っている問い」）:
  - 部屋を一周する撮影でも成り立つか
  - 「約 4% 大きい」が、別の部屋・別のカメラでも同じ向き・同じ大きさで出るか
  - 実測なしの質の判定（写真どうしの食い違い）が、別の部屋でも実測との合い方に対応するか

データ: https://cvg.cit.tum.de/data/datasets/rgbd-dataset （CC BY 4.0）。展開したフォルダに rgb.txt, depth.txt, rgb/, depth/ がある。
  深度画像は 16 ビット PNG で、値 ÷ 5000 が m。0 は値なし。カラーと深度は同じ 640×480 に位置合わせ済み。
  深度センサー（Kinect）自体にも 1〜数 % の誤差があるので、結果はその範囲で読むこと。

使い方（server フォルダで）:
  uv run python ../experiments/tum_eval.py <展開したフォルダ> <出力フォルダ> [枚数=12] [開始秒=0] [終了秒=全体]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))


def read_list(path: Path) -> list[tuple[float, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t, name = line.split()[:2]
        out.append((float(t), name))
    return out


def associate(rgb: list[tuple[float, str]], depth: list[tuple[float, str]], max_dt: float = 0.02) -> list[tuple[float, str, str]]:
    """カラーと深度を、撮影時刻が近いもの（既定 20ms 以内）どうしで組にする。"""
    dt = np.array([d[0] for d in depth])
    pairs = []
    for t, name in rgb:
        j = int(np.argmin(np.abs(dt - t)))
        if abs(dt[j] - t) <= max_dt:
            pairs.append((t, name, depth[j][1]))
    return pairs


def pick_evenly(pairs: list, n: int, t0: float, t1: float | None) -> list:
    start = pairs[0][0]
    sel = [p for p in pairs if p[0] - start >= t0 and (t1 is None or p[0] - start <= t1)]
    if len(sel) <= n:
        return sel
    idx = np.round(np.linspace(0, len(sel) - 1, n)).astype(int)
    return [sel[i] for i in idx]


def model_grid_to_source(hm: int, wm: int, hs: int, ws: int):
    """モデル画像の画素中心 → 元画像の画素（小数）。前処理は、縦横比を保って拡大縮小し、はみ出した側を中央で切り落とす。"""
    s = max(wm / ws, hm / hs)
    ox, oy = (ws * s - wm) / 2, (hs * s - hm) / 2
    u, v = np.meshgrid(np.arange(wm) + 0.5, np.arange(hm) + 0.5)
    return (u + ox) / s, (v + oy) / s


def sample_nearest(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xi = np.clip(np.floor(x).astype(int), 0, img.shape[1] - 1)
    yi = np.clip(np.floor(y).astype(int), 0, img.shape[0] - 1)
    return img[yi, xi]


def evaluate(raw: dict, depth_paths: list[Path]) -> dict:
    """写真ごとに、復元の奥行き（カメラの前方向の距離）と深度センサーの実測を画素ごとに比べる。"""
    from roomfit.worker import view_grids

    rows, all_ratio = [], []
    for v, ((mask, xyz), dp) in enumerate(zip(view_grids(raw), depth_paths)):
        T = raw["cam2world"][v]
        pc = (xyz.astype(np.float64) - T[:3, 3]) @ T[:3, :3]
        z = pc[..., 2]
        gt_full = np.asarray(Image.open(dp)).astype(np.float64) / 5000.0
        xs, ys = model_grid_to_source(z.shape[0], z.shape[1], gt_full.shape[0], gt_full.shape[1])
        gt = sample_nearest(gt_full, xs, ys)
        ok = mask & (gt > 0.3) & (gt < 6.0) & np.isfinite(z) & (z > 0.05)
        if ok.sum() < 2000:
            rows.append({"view": v, "n": int(ok.sum()), "note": "比べられる画素が少ない"})
            continue
        r = z[ok] / gt[ok]
        s = float(np.median(r))
        rel = np.abs(z[ok] / s - gt[ok]) / gt[ok]          # 写真ごとに縮尺を合わせたあとの、形の誤差
        rows.append({"view": v, "n": int(ok.sum()), "scale_pred_over_true": round(s, 4), "shape_err_median_pct": round(float(np.median(rel)) * 100, 2),
                     "shape_err_p90_pct": round(float(np.percentile(rel, 90)) * 100, 2), "depth_median_m": round(float(np.median(gt[ok])), 2)})
        all_ratio.append(r[:: max(1, len(r) // 20000)])
    ratio = np.concatenate(all_ratio) if all_ratio else np.array([np.nan])
    per_view = np.array([x["scale_pred_over_true"] for x in rows if "scale_pred_over_true" in x])
    g = float(np.median(ratio))
    return {
        "views": rows,
        "scale_pred_over_true_overall": round(g, 4),
        "scale_error_pct": round((g - 1) * 100, 1),
        "per_view_scale_min_max": [round(float(per_view.min()), 4), round(float(per_view.max()), 4)] if len(per_view) else None,
        "per_view_scale_spread_pct": round(float((per_view.max() - per_view.min()) / per_view.mean() * 100), 1) if len(per_view) else None,
        "global_shape_err_median_pct": round(float(np.median(np.abs(ratio / g - 1))) * 100, 2),
    }


def main() -> None:
    from roomfit.reconstruct import run_mapanything, save_raw
    from roomfit.structure import StructureError
    from roomfit.worker import capture_quality, structure_from_raw

    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    t0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    t1 = float(sys.argv[5]) if len(sys.argv) > 5 else None
    pairs = pick_evenly(associate(read_list(src / "rgb.txt"), read_list(src / "depth.txt")), n, t0, t1)
    if len(pairs) < 3:
        sys.exit("カラーと深度の組が足りません。")
    (out / "images").mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (_, rgb, _) in enumerate(pairs):
        p = out / "images" / f"{i:03d}.png"
        Image.open(src / rgb).convert("RGB").save(p)
        paths.append(p)
    print(f"{len(pairs)} 枚を選択（{pairs[-1][0] - pairs[0][0]:.1f} 秒の範囲）", flush=True)
    t = time.time()
    raw = run_mapanything(paths)
    raw["meta"]["sec_total"] = round(time.time() - t, 1)
    save_raw(raw, out / "raw.npz")
    res = evaluate(raw, [src / d for _, _, d in pairs])
    res["capture_quality"] = capture_quality(raw)
    res["sec_inference"] = raw["meta"]["sec_inference"]
    try:
        room = structure_from_raw(raw, out)
        res["structure"] = {"ceiling_m": room["ceiling_height"], "corner_deg": room["diagnostics"].get("wall_corner_angle_deg"),
                            "rectified": room["diagnostics"].get("rectified"), "edges": room["edge_status"]}
    except StructureError as e:
        res["structure"] = {"failed": e.code}
    (out / "tum_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "views"}, indent=1, ensure_ascii=False))
    for r in res["views"]:
        print("  ", r)


if __name__ == "__main__":
    main()
