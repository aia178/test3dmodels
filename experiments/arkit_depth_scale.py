"""実験（ARKitScenes）: AR の位置は実寸なのに、モデルの奥行きだけが小さく出る（docs/検証記録.md §16）のを、写真どうしの対応点で直す。

同じ点を写真 i と j から出した 3D 位置 X_i = R_i (a_i P_i) + t_i、X_j = R_j (a_j P_j) + t_j が一致するよう、倍率 a を決める。
R, t は AR の位置（実寸）、P はモデルのカメラ座標の点。
  G = 全体で 1 つの倍率
  C = 写真ごとの倍率（1 付近への弱い引き戻しつき）
対応点は SIFT（商用可）＋比の検定＋RANSAC、さらに補正前の食い違いが大きいものは誤対応として捨てる。

使い方（server フォルダで）:
  uv run python ../experiments/arkit_depth_scale.py <データのフォルダ> <arkit_posed の出力> <枚数> [開始秒 終了秒]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
sys.path.insert(0, str(here))
from arkit_posed import evaluate, load_frames, pick  # noqa: E402
from roomfit.reconstruct import load_raw, save_raw  # noqa: E402
from roomfit.worker import view_grids  # noqa: E402

MAX_GAP, GATE_REL = 4, 0.35    # 何枚先まで対応点を探すか / 補正前の食い違いが奥行きのこの割合を超える対応は捨てる


def matches(a, b, K, sift):
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    m = [p[0] for p in cv2.BFMatcher().knnMatch(da, db, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(m) < 20:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    E, inl = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 1.0)
    if inl is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    k = inl.ravel().astype(bool)
    return pa[k], pb[k]


def main() -> None:
    src, run, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    t0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    t1 = float(sys.argv[5]) if len(sys.argv) > 5 else None
    frames = pick(load_frames(src), n, t0, t1)
    raw = load_raw(run / "raw.npz")
    grids = view_grids(raw)
    N = len(grids)
    Tout = raw["cam2world"]
    Tin = np.stack([f["c2w"] for f in frames])
    hm, wm = grids[0][0].shape
    s = max(wm / 640, hm / 480)
    ox, oy = (640 * s - wm) / 2, (480 * s - hm) / 2
    cam = [(xyz.astype(np.float64) - Tout[v][:3, 3]) @ Tout[v][:3, :3] for v, (_, xyz) in enumerate(grids)]
    imgs = [cv2.imread(str(run / "images" / f"{i:03d}.png"), cv2.IMREAD_GRAYSCALE) for i in range(N)]
    sift = cv2.SIFT_create(nfeatures=5000)
    rows = []
    for i in range(N):
        for j in range(i + 1, min(N, i + 1 + MAX_GAP)):
            pa, pb = matches(imgs[i], imgs[j], frames[i]["K"], sift)
            if len(pa) < 15:
                continue
            ua, va = np.clip((pa[:, 0] * s - ox).astype(int), 0, wm - 1), np.clip((pa[:, 1] * s - oy).astype(int), 0, hm - 1)
            ub, vb = np.clip((pb[:, 0] * s - ox).astype(int), 0, wm - 1), np.clip((pb[:, 1] * s - oy).astype(int), 0, hm - 1)
            Pi, Pj = cam[i][va, ua], cam[j][vb, ub]
            ok = grids[i][0][va, ua] & grids[j][0][vb, ub] & np.isfinite(Pi).all(1) & np.isfinite(Pj).all(1)
            for p, q in zip(Pi[ok], Pj[ok]):
                ri, rj, dt = Tin[i][:3, :3] @ p, Tin[j][:3, :3] @ q, Tin[i][:3, 3] - Tin[j][:3, 3]
                rows.append((i, j, ri, rj, dt))
    # 全体の倍率の初期値（ゲートなしの頑健な 1 次元探索）で誤対応を見分ける
    def resid(a_vec, rr):
        return np.array([np.linalg.norm(a_vec[i] * ri - a_vec[j] * rj + dt) for i, j, ri, rj, dt in rr])
    grid = np.linspace(0.6, 1.6, 101)
    med = [np.median(resid(np.full(N, g), rows)) for g in grid]
    g0 = float(grid[int(np.argmin(med))])
    r0 = resid(np.full(N, g0), rows)
    depth = np.array([np.linalg.norm(ri) for _, _, ri, _, _ in rows])
    keep = r0 < GATE_REL * g0 * depth
    rows = [r for r, k in zip(rows, keep) if k]
    M = len(rows)
    counts = np.zeros(N, int)
    for i, j, *_ in rows:
        counts[i] += 1; counts[j] += 1
    print(f"対応点 {M}（捨てた {int((~keep).sum())}）、写真ごとの点の数 {counts.tolist()}")

    def solve(per_view: bool):
        w = np.ones(M)
        a = np.full(N, g0)
        for _ in range(10):
            k = N if per_view else 1
            A = np.zeros((3 * M + k, k)); b = np.zeros(3 * M + k)
            for m_, (i, j, ri, rj, dt) in enumerate(rows):
                ci, cj = (i, j) if per_view else (0, 0)
                A[3 * m_:3 * m_ + 3, ci] += w[m_] * ri
                A[3 * m_:3 * m_ + 3, cj] -= w[m_] * rj
                b[3 * m_:3 * m_ + 3] = -w[m_] * dt
            if per_view:   # 全体の倍率 g0 付近へ弱く引き戻す（点の少ない写真が暴れないように）
                reg = np.sqrt(max(1, M / N)) * 0.05
                A[3 * M:, :] = np.eye(N) * reg
                b[3 * M:] = reg * g0
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            a = sol if per_view else np.full(N, sol[0])
            r = resid(a, rows)
            c = 1.345 * max(0.01, float(np.median(r)))
            w = np.where(r <= c, 1.0, np.sqrt(c / np.maximum(r, 1e-9)))
        return a, float(np.median(r))

    for tag, per_view in (("G", False), ("C", True)):
        a, rmed = solve(per_view)
        print(f"{tag}: 倍率 {np.round(a, 3).tolist() if per_view else round(float(a[0]), 4)}  対応点の食い違い（中央値）{rmed * 100:.1f}cm")
        pts = []
        for v in range(N):
            mask = grids[v][0]
            pts.append(((cam[v][mask] * a[v]) @ Tin[v][:3, :3].T + Tin[v][:3, 3]).astype(np.float32))
        new = dict(raw)
        new["points"] = np.concatenate(pts)
        new["cam2world"] = Tin.astype(np.float64)
        new["meta"] = {**raw["meta"], "depth_scale_fix": tag, "scale": a.round(4).tolist()}
        out = Path(str(run) + "_" + tag)
        out.mkdir(exist_ok=True)
        save_raw(new, out / "raw.npz")
        res = evaluate(new, frames)
        (out / "arkit_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        brief = {k: (v["abs_error_cm_median"], v["error_pct"]) for k, v in res.items() if k.startswith(("cross", "same"))}
        print(f"  評価: 写真ごとの大きさ 中央値 {res['scale_median']} 四分位幅 {res['scale_iqr_pct']}%  距離の誤差(cm, %) {brief}", flush=True)


if __name__ == "__main__":
    main()
