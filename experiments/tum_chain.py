"""実験（TUM RGB-D）: 部屋一周を、重なりの高い小さな範囲に分けて復元し、共通の写真で鎖のようにつなぐ。

tum_chunk.py（2 範囲）の続き。範囲 0 = 最初の 10 秒・12 枚（tum_room_dense10）、範囲 1 = 10〜15 秒（tum_chunk_b15）は既存の結果を使い、
そのあと 5 秒ずつ、前の範囲の最後の 3 枚 + 新しい 10 枚で復元する。途中で止めても、済んだ範囲は飛ばして再開できる。

使い方（server フォルダで）:
  uv run python ../experiments/tum_chain.py run  <データ> <出力の親フォルダ> [終了秒=49]
  uv run python ../experiments/tum_chain.py eval <データ> <出力の親フォルダ> [終了秒=49] [使う範囲の数]
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
from roomfit.stitch import robust_similarity  # noqa: E402
from tum_chunk import CX, CY, FX, FY  # noqa: E402
from tum_eval import associate, model_grid_to_source, pick_evenly, read_list, sample_nearest  # noqa: E402

N_NEW, N_SHARED, STEP = 10, 3, 5.0


def icp_rigid(src: np.ndarray, dst: np.ndarray, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """点全体どうしの合わせ直し（回転と移動だけ。大きさは変えない）。src は、共通の写真で大まかに合わせた後の座標。
    近い点どうしを組にして（遠い組は捨てる）、ずれが減るように少しずつ動かす。戻り値は、src に掛ける追加の回転と移動。"""
    from scipy.spatial import cKDTree
    tree = cKDTree(dst)
    R, t = np.eye(3), np.zeros(3)
    for it in range(iters):
        cur = src @ R.T + t
        d, nn = tree.query(cur, workers=-1)
        keep = d < max(0.03, 0.15 * (1 - it / iters))       # 15cm から 3cm へ、だんだん厳しく
        if keep.sum() < 2000:
            break
        a, b = cur[keep], dst[nn[keep]]
        ma, mb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((b - mb).T @ (a - ma))
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        dR = U @ D @ Vt
        R, t = dR @ R, dR @ t + (mb - dR @ ma)
    return R, t


def plan(src: Path, base: Path, t_end: float) -> list[tuple[Path, list]]:
    allp = associate(read_list(src / "rgb.txt"), read_list(src / "depth.txt"))
    start = allp[0][0]
    chunks = [(base / "tum_room_dense10", pick_evenly(allp, 12, 0.0, 10.0))]
    t, k = 10.0, 1
    while t < t_end - 1.0:
        t1 = min(t + STEP, t_end)
        prev = chunks[-1][1]
        later = [p for p in allp if p[0] > prev[-1][0] and p[0] - start <= t1]
        if len(later) < N_NEW + 1:
            break
        idx = np.round(np.linspace(0, len(later) - 1, N_NEW + 1)).astype(int)[1:]
        name = "tum_chunk_b15" if k == 1 else f"tum_chain_c{k}"
        chunks.append((base / name, prev[-N_SHARED:] + [later[i] for i in idx]))
        t, k = t1, k + 1
    return chunks


def main() -> None:
    from roomfit.arcore_import import quat_to_rot
    from roomfit.reconstruct import load_raw, run_mapanything, save_raw
    from roomfit.worker import capture_quality, view_grids

    mode, src, base = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    t_end = float(sys.argv[4]) if len(sys.argv) > 4 else 49.0
    chunks = plan(src, base, t_end)
    if mode == "run":
        for k, (out, frames) in enumerate(chunks):
            if (out / "raw.npz").exists():
                print(f"範囲 {k}: 済み（{out.name}）", flush=True)
                continue
            (out / "images").mkdir(parents=True, exist_ok=True)
            paths = []
            for i, (_, rgb, _) in enumerate(frames):
                p = out / "images" / f"{i:03d}.png"
                Image.open(src / rgb).convert("RGB").save(p)
                paths.append(p)
            t = time.time()
            raw = run_mapanything(paths)
            save_raw(raw, out / "raw.npz")
            q = capture_quality(raw)
            (out / "quality.json").write_text(json.dumps({"quality": q, "sec": round(time.time() - t, 1)}, ensure_ascii=False), encoding="utf-8")
            print(f"範囲 {k}: {len(frames)} 枚 {time.time() - t:.0f} 秒 {q}", flush=True)
            del raw
        return

    if len(sys.argv) > 5:
        chunks = chunks[: int(sys.argv[5])]
    use_global = len(sys.argv) > 6 and sys.argv[6] == "global"   # 順につないだ後、全範囲を同時に合わせ直す
    use_icp = len(sys.argv) > 6 and sys.argv[6] in ("icp", "global")
    scale_pow = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0   # つなぎの縮尺をどれだけ信じるか: 1 = 共通の写真どおり、0 = 変えない、0.5 = 半分だけ寄せる
    use_app = len(sys.argv) > 6 and sys.argv[6] == "app"      # アプリ本体の処理（roomfit.stitch.merge_raws）でつなぐ
    chunks = [c for c in chunks if (c[0] / "raw.npz").exists()]
    gt = np.array([[float(x) for x in line.split()] for line in (src / "groundtruth.txt").read_text().splitlines() if not line.startswith("#")])
    rng = np.random.default_rng(0)
    S, R, T = 1.0, np.eye(3), np.zeros(3)       # 範囲 k の座標 → 範囲 0 の座標
    prev_tail, prev_cloud, links, quality = None, None, [], []
    P_all, Q_all, V_all, C_all, v = [], [], [], [], 0

    def add_view(mask, xyz, frame, k):
        """xyz は範囲 0 の座標へ移した後の値。実測と比べる点を 2000 個ずつ取る。"""
        nonlocal v
        tt, _, dname = frame
        g = int(np.argmin(np.abs(gt[:, 0] - tt)))
        Rg, tr = quat_to_rot(gt[g, 4:8]), gt[g, 1:4]
        d = np.asarray(Image.open(src / dname)).astype(np.float64) / 5000.0
        xs, ys = model_grid_to_source(xyz.shape[0], xyz.shape[1], d.shape[0], d.shape[1])
        z = sample_nearest(d, xs, ys)
        Qc = np.stack([(xs - CX) / FX * z, (ys - CY) / FY * z, z], -1)
        idx = np.argwhere(mask & (z > 0.3) & (z < 4.0) & np.isfinite(xyz).all(-1))
        sel = idx[rng.integers(0, len(idx), 2000)]
        P_all.append(xyz[sel[:, 0], sel[:, 1]].astype(np.float64)); Q_all.append(Qc[sel[:, 0], sel[:, 1]] @ Rg.T + tr)
        V_all.append(np.full(len(sel), v)); C_all.append(np.full(len(sel), k)); v += 1

    structure, cum_scales, held = None, [], []
    if use_app:
        from roomfit.reconstruct import save_raw
        from roomfit.stitch import merge_raws
        from roomfit.structure import StructureError
        from roomfit.worker import structure_from_raw

        merged = merge_raws([load_raw(out / "raw.npz") for out, _ in chunks], N_SHARED)
        mdir = base / "tum_chain_merged"
        mdir.mkdir(exist_ok=True)
        save_raw(merged, mdir / "raw.npz")
        frames_all = chunks[0][1] + [f for _, fr in chunks[1:] for f in fr[N_SHARED:]]
        for (m, x), fr, k in zip(view_grids(merged), frames_all, merged["meta"]["chunks"]["view_chunk"]):
            add_view(m, x, fr, k)
        links, quality = merged["meta"]["chunks"]["links"], [capture_quality(merged)]
        try:
            room = structure_from_raw(merged, mdir)
            structure = {"ceiling_m": room["ceiling_height"], "edges_observed": room["edge_status"].count("observed"), "edges": len(room["edge_status"]),
                         "outline_extent_m": [round(float(x), 2) for x in np.ptp(np.array(room["outline"]), axis=0)]}
        except StructureError as e:
            structure = {"failed": e.code}
        chunks_iter = []
    else:
        chunks_iter = chunks
    for k, (out, frames) in enumerate(chunks_iter):
        raw = load_raw(out / "raw.npz")
        q = capture_quality(raw)
        quality.append({"chunk": k, "overlap": q["overlap"], "disagreement_mm": q["disagreement_mm"], "level": q["level"]})
        grids = [(m, x.astype(np.float64)) for m, x in view_grids(raw)]
        del raw
        cloud = np.concatenate([x[m & np.isfinite(x).all(-1)][::23] for m, x in grids])     # この範囲の点全体（間引き）
        if k > 0:
            src_pts, dst_pts = [], []
            for (ma, xa), (mb, xb) in zip(prev_tail, grids[:N_SHARED]):
                ok = ma & mb & np.isfinite(xa).all(-1) & np.isfinite(xb).all(-1)
                src_pts.append(xb[ok]); dst_pts.append(xa[ok])
            s, r, t, res = robust_similarity(np.concatenate(src_pts)[::5], np.concatenate(dst_pts)[::5])
            if scale_pow != 1.0:
                s, r, t, res = robust_similarity(np.concatenate(src_pts)[::5], np.concatenate(dst_pts)[::5], scale=s ** scale_pow)
            links.append({"chunk": k, "scale_to_prev": round(s, 4), "residual_median_mm": round(float(np.median(res)) * 1000, 1)})
            if use_icp:
                pre = cloud @ r.T * s + t
                dR, dt = icp_rigid(pre, prev_cloud)
                c = pre.mean(0)
                links[-1]["icp_turn_deg"] = round(float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))), 2)
                links[-1]["icp_shift_cm"] = round(float(np.linalg.norm(dR @ c + dt - c)) * 100, 1)
                r, t = dR @ r, dR @ t + dt
            # 前の範囲への変換に、前の範囲から範囲 0 への変換を重ねる
            S, R, T = S * s, R @ r, S * (R @ t) + T
        cum_scales.append(round(S, 4))
        prev_tail, prev_cloud = grids[-N_SHARED:], cloud
        held.append({"k": k, "views": list(zip(grids, frames))[(N_SHARED if k > 0 else 0):], "cloud": cloud * S, "S": S, "R": R, "T": T})
    refine_pairs = None
    if use_global and held:
        from roomfit.stitch import global_refine
        Rs, Ts, refine_pairs = global_refine([h["cloud"] for h in held], [h["R"] for h in held], [h["T"] for h in held])
        for h, r_, t_ in zip(held, Rs, Ts):
            h["move_cm"] = round(float(np.linalg.norm((h["cloud"] @ r_.T + t_).mean(0) - (h["cloud"] @ h["R"].T + h["T"]).mean(0))) * 100, 1)
            h["R"], h["T"] = r_, t_
        print("合わせ直しでの移動 [cm]:", [h["move_cm"] for h in held], flush=True)
    for h in held:
        for (mask, xyz), frame in h["views"]:
            add_view(mask, xyz @ h["R"].T * h["S"] + h["T"], frame, h["k"])
    P, Q, V, C = map(np.concatenate, (P_all, Q_all, V_all, C_all))
    i, j = rng.integers(0, len(P), 800000), rng.integers(0, len(P), 800000)
    dp, dq = np.linalg.norm(P[i] - P[j], axis=1), np.linalg.norm(Q[i] - Q[j], axis=1)
    gap = np.abs(C[i] - C[j])
    res_out = {"n_chunks": len(chunks), "n_views": v, "cumulative_scales": cum_scales, "links": links, "quality": quality, "structure": structure, "refine_pairs": refine_pairs,
               "room_extent_true_m": [round(float(x), 2) for x in (np.percentile(Q, 99, axis=0) - np.percentile(Q, 1, axis=0))]}

    def stat(m, name):
        if m.sum() < 500:
            return
        r = dp[m] / dq[m]
        g = float(np.median(r))
        res_out[name] = {"pairs": int(m.sum()), "error_pct": round((g - 1) * 100, 1),
                         "p10_p90_pct": [round(float(np.percentile(r, 10) - 1) * 100, 1), round(float(np.percentile(r, 90) - 1) * 100, 1)],
                         "abs_error_cm_median": round(float(np.median(np.abs(dp[m] - dq[m]))) * 100, 1)}

    diff = V[i] != V[j]
    stat(diff & (dq > 0.5) & (dq <= 1.5), "別の写真の 2 点（0.5〜1.5m）")
    stat(diff & (dq > 1.5) & (dq <= 3.0), "別の写真の 2 点（1.5〜3m）")
    stat(diff & (dq > 3.0), "別の写真の 2 点（3m 以上）")
    for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 99)):
        stat(diff & (gap >= lo) & (gap <= hi) & (dq > 1.5), f"範囲が {lo}〜{hi} 個離れた 2 点（1.5m 以上）")
    # 全体をひとつの拡大縮小・回転・移動で実測に合わせたあとの、点の位置のずれ（形の崩れ・誤差の溜まり方）
    s, r, t, _ = robust_similarity(P[::3], Q[::3], rounds=1)
    err = np.linalg.norm(P @ r.T * s + t - Q, axis=1)
    res_out["位置のずれ（全体を実測に合わせた後）"] = {"scale_pred_over_true": round(1 / s, 4), "median_cm": round(float(np.median(err)) * 100, 1),
                                         "p90_cm": round(float(np.percentile(err, 90)) * 100, 1),
                                         "by_chunk_median_cm": [round(float(np.median(err[C == k])) * 100, 1) for k in range(len(chunks))]}
    (base / f"tum_chain_eval_{len(chunks)}{'_global' if use_global else '_icp' if use_icp else '_app' if use_app else ''}{'' if scale_pow == 1.0 else f'_s{scale_pow}'}.json").write_text(json.dumps(res_out, indent=1, ensure_ascii=False), encoding="utf-8")
    for key, val in res_out.items():
        print("  ", key, val)


if __name__ == "__main__":
    main()
