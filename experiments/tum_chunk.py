"""実験（TUM RGB-D）: 部屋を「重なりの高い小さな範囲」に分けて別々に復元し、共通の写真でつなぐ。

背景（docs/検証記録.md §14）: 1 回の復元は 20 枚前後が上限で、範囲を広げると重なりが落ちて、写真どうしのつなぎ方が崩れる。
やり方: 範囲 A と範囲 B を別々に復元する。B には A の最後の数枚と同じ写真を入れておく。
        同じ写真の同じ画素は、同じ場所を指すはずなので、その対応から「B の座標 → A の座標」の変換（回転・移動・拡大縮小）を求める。
比べる相手: 同じ時間の範囲を 1 回で復元した結果（例: 最初の 20 秒・16 枚）。

使い方（server フォルダで）:
  復元: uv run python ../experiments/tum_chunk.py run  <データ> <A の出力> <B の出力> <A 開始秒> <A 終了秒> <B 終了秒> [B の新しい枚数=10] [共通の枚数=3]
  評価: uv run python ../experiments/tum_chunk.py eval <データ> <A の出力> <B の出力> ...（同じ引数）
A の出力は、tum_eval.py で作った既存のフォルダ（同じ開始秒・終了秒・枚数）を使う。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
sys.path.insert(0, str(here))
from tum_eval import associate, model_grid_to_source, pick_evenly, read_list, sample_nearest  # noqa: E402

FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """dst ≈ s·R·src + t となる拡大縮小 s・回転 R・移動 t（最小二乗）。"""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean())
    return s, R, md - s * R @ ms


def robust_similarity(src: np.ndarray, dst: np.ndarray, rounds: int = 6) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """はずれを除きながら変換を求める。戻り値の最後は、残った点での残差 [m]。"""
    keep = np.ones(len(src), bool)
    for _ in range(rounds):
        s, R, t = umeyama(src[keep], dst[keep])
        res = np.linalg.norm(src @ R.T * s + t - dst, axis=1)
        keep = res < max(0.01, 2.5 * float(np.median(res[keep])))
    return s, R, t, res[keep]


def select(src: Path, a0: float, a1: float, b1: float, n_new: int, n_shared: int, n_a: int):
    allp = associate(read_list(src / "rgb.txt"), read_list(src / "depth.txt"))
    pa = pick_evenly(allp, n_a, a0, a1)
    start = allp[0][0]
    later = [p for p in allp if p[0] > pa[-1][0] and p[0] - start <= b1]
    idx = np.round(np.linspace(0, len(later) - 1, n_new + 1)).astype(int)[1:]   # A の最後の写真の直後は飛ばす
    return pa, pa[-n_shared:] + [later[i] for i in idx]


def main() -> None:
    from roomfit.arcore_import import quat_to_rot
    from roomfit.reconstruct import load_raw, run_mapanything, save_raw
    from roomfit.worker import capture_quality, view_grids

    mode, src, out_a, out_b = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    a0, a1, b1 = float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
    n_new = int(sys.argv[8]) if len(sys.argv) > 8 else 10
    n_shared = int(sys.argv[9]) if len(sys.argv) > 9 else 3
    raw_a = load_raw(out_a / "raw.npz")
    pa, pb = select(src, a0, a1, b1, n_new, n_shared, len(raw_a["cam2world"]))

    if mode == "run":
        (out_b / "images").mkdir(parents=True, exist_ok=True)
        paths = []
        for i, (_, rgb, _) in enumerate(pb):
            p = out_b / "images" / f"{i:03d}.png"
            Image.open(src / rgb).convert("RGB").save(p)
            paths.append(p)
        print(f"B: {len(pb)} 枚（共通 {n_shared} 枚 + 新しい {n_new} 枚）", flush=True)
        t = time.time()
        raw_b = run_mapanything(paths)
        raw_b["meta"]["sec_total"] = round(time.time() - t, 1)
        save_raw(raw_b, out_b / "raw.npz")
        print("復元", raw_b["meta"]["sec_inference"], "秒", capture_quality(raw_b), flush=True)
        return

    raw_b = load_raw(out_b / "raw.npz")
    ga, gb = list(view_grids(raw_a)), list(view_grids(raw_b))
    # 共通の写真: A の最後の n_shared 枚 = B の最初の n_shared 枚。同じ画素どうしを対応させる。
    S, D, per_view = [], [], []
    for k in range(n_shared):
        (ma, xa), (mb, xb) = ga[len(ga) - n_shared + k], gb[k]
        ok = ma & mb & np.isfinite(xa).all(-1) & np.isfinite(xb).all(-1)
        S.append(xb[ok].astype(np.float64)); D.append(xa[ok].astype(np.float64))
        sk, _, _, rk = robust_similarity(S[-1][::7], D[-1][::7])
        per_view.append({"scale": round(sk, 4), "residual_mm": round(float(np.median(rk)) * 1000, 1), "pixels": int(ok.sum())})
    s, R, t, res = robust_similarity(np.concatenate(S)[::5], np.concatenate(D)[::5])
    link = {"scale_b_to_a": round(s, 4), "residual_median_mm": round(float(np.median(res)) * 1000, 1),
            "residual_p90_mm": round(float(np.percentile(res, 90)) * 1000, 1), "per_shared_photo": per_view}
    print("つなぎ:", link, flush=True)

    gt = np.array([[float(x) for x in line.split()] for line in (src / "groundtruth.txt").read_text().splitlines() if not line.startswith("#")])
    rng = np.random.default_rng(0)
    views = [(m, x.astype(np.float64), p, 0) for (m, x), p in zip(ga, pa)]
    views += [(m, x.astype(np.float64) @ R.T * s + t, p, 1) for (m, x), p in list(zip(gb, pb))[n_shared:]]
    P_all, Q_all, V_all, C_all = [], [], [], []
    for v, (mask, xyz, (tt, _, dname), chunk) in enumerate(views):
        k = int(np.argmin(np.abs(gt[:, 0] - tt)))
        Rg, tr = quat_to_rot(gt[k, 4:8]), gt[k, 1:4]
        d = np.asarray(Image.open(src / dname)).astype(np.float64) / 5000.0
        xs, ys = model_grid_to_source(xyz.shape[0], xyz.shape[1], d.shape[0], d.shape[1])
        z = sample_nearest(d, xs, ys)
        Qc = np.stack([(xs - CX) / FX * z, (ys - CY) / FY * z, z], -1)
        idx = np.argwhere(mask & (z > 0.3) & (z < 4.0) & np.isfinite(xyz).all(-1))
        sel = idx[rng.integers(0, len(idx), 3000)]
        P_all.append(xyz[sel[:, 0], sel[:, 1]]); Q_all.append(Qc[sel[:, 0], sel[:, 1]] @ Rg.T + tr)
        V_all.append(np.full(len(sel), v)); C_all.append(np.full(len(sel), chunk))
    P, Q, V, C = map(np.concatenate, (P_all, Q_all, V_all, C_all))
    i, j = rng.integers(0, len(P), 400000), rng.integers(0, len(P), 400000)
    dp, dq = np.linalg.norm(P[i] - P[j], axis=1), np.linalg.norm(Q[i] - Q[j], axis=1)
    same, cross = V[i] == V[j], C[i] != C[j]
    out = {"n_views": len(views), "link": link,
           "room_extent_true_m": [round(float(x), 2) for x in (np.percentile(Q, 99, axis=0) - np.percentile(Q, 1, axis=0))]}

    def stat(m, name):
        if m.sum() < 500:
            return
        r = dp[m] / dq[m]
        g = float(np.median(r))
        out[name] = {"pairs": int(m.sum()), "error_pct": round((g - 1) * 100, 1),
                     "p10_p90_pct": [round(float(np.percentile(r, 10) - 1) * 100, 1), round(float(np.percentile(r, 90) - 1) * 100, 1)],
                     "abs_error_cm_median": round(float(np.median(np.abs(dp[m] - dq[m]))) * 100, 1)}

    stat(same & (dq > 0.5), "同じ写真の中の 2 点（50cm 以上）")
    stat(~same & (dq > 0.5) & (dq <= 1.5), "別の写真の 2 点（0.5〜1.5m）")
    stat(~same & (dq > 1.5) & (dq <= 3.0), "別の写真の 2 点（1.5〜3m）")
    stat(~same & (dq > 3.0), "別の写真の 2 点（3m 以上）")
    stat(cross & (dq > 1.5) & (dq <= 3.0), "範囲 A と範囲 B にまたがる 2 点（1.5〜3m）")
    stat(cross & (dq > 3.0), "範囲 A と範囲 B にまたがる 2 点（3m 以上）")
    for c, label in ((0, "A"), (1, "B")):   # どちらの範囲の中で崩れているかを見る
        inside = ~same & (C[i] == c) & (C[j] == c)
        stat(inside & (dq > 1.5) & (dq <= 3.0), f"範囲 {label} の中の別の写真（1.5〜3m）")
        stat(inside & (dq > 3.0), f"範囲 {label} の中の別の写真（3m 以上）")
    (out_b / "tum_chunk_eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, val in out.items():
        print("  ", k, val)


if __name__ == "__main__":
    main()
