"""モデル自身のカメラの位置と向きで写真どうしの対応点を三角測量し、モデルの奥行きと比べる（再推論なし）。

仮説: MapAnything の出力では、カメラの位置（間隔）と、写真ごとの奥行きの大きさが食い違っていることがある
（10/21 の自宅の記録で、カメラの間隔は AR の 0.985 倍なのに物は 4% 大きく、写真どうしの床が 8〜11cm 食い違った）。
三角測量の奥行きはカメラの位置と向きだけで決まるので、「モデルの奥行き ÷ 三角測量の奥行き」の中央値が、点の膨らみの見積もりになる。
これで点を割った倍率（1 ÷ 膨らみ）を全体に当てたときの、大きさのずれと写真をまたぐ距離の誤差を、何もしない場合と比べる。

対応点: SIFT ＋ 比の検定。モデルの姿勢から求めたエピ極線から 2 画素以内、再投影の誤差 2 画素以内、視線の角度 2° 以上（元の写真の画素）。
内部パラメータは、そのコマのモデルの点（カメラ座標）に当てたもの（plate_ruler.fit_K）を元の写真の画素に戻して使う。

使い方（server フォルダで）: uv run python ../experiments/self_triangulate.py [ケース名 ...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent / "server"))
from plate_ruler import fit_K, preprocess_map  # noqa: E402
from scale_bench import BANDS, BENCH, MEASURED, home_metrics, inview_ratio, load_case, public_metrics  # noqa: E402
from triangulate_scale import epi_dist, fundamental, projection  # noqa: E402

EPI_PX, REPROJ_PX, MIN_ANGLE_DEG = 2.0, 2.0, 2.0


def views_of(case):
    """コマごとの (元の写真, 元の写真 → モデル画像の写し方, 元の写真の画素での内部パラメータ, モデルのカメラ座標の点, マスク)。"""
    c = np.load(BENCH / "cases" / f"{case['name']}.npz")
    out = []
    for v in range(case["n"]):
        with Image.open(str(c["src"][v])) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        h, w = rgb.shape[:2]
        sx, tx, sy, ty, shape = preprocess_map(w, h)
        assert shape == case["mask"][v].shape, (shape, case["mask"][v].shape)
        Km = fit_K(case["ma"][v].astype(np.float64), case["mask"][v])
        Ko = np.array([[Km[0, 0] / sx, 0, (Km[0, 2] - tx) / sx], [0, Km[1, 1] / sy, (Km[1, 2] - ty) / sy], [0, 0, 1]])
        out.append((rgb, (sx, tx, sy, ty), Ko, case["ma"][v], case["mask"][v]))
    return out


def mixed_poses(case):
    """向きはモデルの出力、位置は正解（製品なら AR）。正解の位置をモデルの座標へ剛体で合わせる（倍率は 1 のまま）。"""
    A, B = case["ma_c2w"][:, :3, 3], case["gt_c2w"][:, :3, 3]
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    T = case["ma_c2w"].copy()
    T[:, :3, 3] = (B - cb) @ R.T + ca
    return T


def inflation(case, views, T):
    sift, bf = cv2.SIFT_create(4000), cv2.BFMatcher(cv2.NORM_L2)
    feats = [sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None) for rgb, *_ in views]
    ratios, per_pair = [], []
    for a in range(len(views)):
        for b in range(a + 1, len(views)):
            (ka, da), (kb, db) = feats[a], feats[b]
            if da is None or db is None or len(ka) < 20 or len(kb) < 20:
                continue
            ms = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
            if len(ms) < 20:
                continue
            pa = np.float64([ka[m.queryIdx].pt for m in ms])
            pb = np.float64([kb[m.trainIdx].pt for m in ms])
            Ka, Kb = views[a][2], views[b][2]
            ok = epi_dist(fundamental(Ka, T[a], Kb, T[b]), pa, pb) < EPI_PX
            pa, pb = pa[ok], pb[ok]
            if len(pa) < 8:
                continue
            X = cv2.triangulatePoints(projection(Ka, T[a]), projection(Kb, T[b]), pa.T, pb.T)
            X = (X[:3] / X[3]).T
            good = np.ones(len(X), bool)
            for K, Tv, p in ((Ka, T[a], pa), (Kb, T[b], pb)):
                pc = (X - Tv[:3, 3]) @ Tv[:3, :3]
                uv = pc @ K.T
                good &= (pc[:, 2] > 0.1) & (np.linalg.norm(uv[:, :2] / uv[:, 2:] - p, axis=1) < REPROJ_PX)
            ra, rb = X - T[a][:3, 3], X - T[b][:3, 3]
            ang = np.degrees(np.arccos(np.clip((ra * rb).sum(1) / np.linalg.norm(ra, axis=1) / np.linalg.norm(rb, axis=1), -1, 1)))
            good &= ang > MIN_ANGLE_DEG
            pair = []
            for v, p in ((a, pa), (b, pb)):
                _, (sx, tx, sy, ty), _, xyz, mask = views[v]
                zt = ((X[good] - T[v][:3, 3]) @ T[v][:3, :3])[:, 2]
                i = np.round(p[good, 1] * sy + ty - 0.5).astype(int)
                j = np.round(p[good, 0] * sx + tx - 0.5).astype(int)
                inb = (i >= 0) & (i < mask.shape[0]) & (j >= 0) & (j < mask.shape[1])
                inb[inb] = mask[i[inb], j[inb]]
                zm = xyz[i[inb], j[inb], 2]
                pair += list(zm / zt[inb])
            if pair:
                ratios += pair
                per_pair.append((a, b, len(pair), float(np.median(pair))))
    return np.array(ratios), per_pair


def main():
    idx = json.loads((BENCH / "cases" / "index.json").read_text(encoding="utf-8"))
    kinds = {**idx["public"], **idx.get("posed", {})}
    names = sys.argv[1:] or ["home_avg"] + list(kinds)
    rows = []
    for name in names:
        case = load_case(name)
        views = views_of(case)
        variants = [("モデルの姿勢", case["ma_c2w"])] + ([("向き=モデル・位置=正解", mixed_poses(case))] if "gt_c2w" in case else [])
        for label, T in variants:
            r, pairs = inflation(case, views, T)
            if len(r) < 30:
                print(f"{name:22s} {label}: 三角測量の点が少なすぎる（{len(r)}）", flush=True)
                continue
            infl = float(np.median(r))
            s = 1 / infl
            row = {"case": name, "variant": label, "n_pts": int(len(r)), "n_pairs": len(pairs), "inflation": infl,
                   "iqr": [float(np.percentile(r, 25)), float(np.percentile(r, 75))]}
            head = f"{name:22s} {label}: 膨らみ {infl:.4f}（25〜75% {row['iqr'][0]:.3f}〜{row['iqr'][1]:.3f}、点 {len(r)}、組 {len(pairs)}）"
            if name.startswith("home"):
                h0, h1 = home_metrics(name, 1.0), home_metrics(name, s)
                row.update(home_none=h0["mean_abs_cm"], home_corr=h1["mean_abs_cm"])
                print(f"{head} ｜ 自宅の差 {h0['mean_abs_cm']:.2f} → {h1['mean_abs_cm']:.2f}cm", flush=True)
            else:
                m0, m1 = public_metrics(case, 1.0), public_metrics(case, s)
                row.update({f"none {k}": v for k, v in m0.items()}, **{f"corr {k}": v for k, v in m1.items()})
                print(f"{head} ｜ 大きさ {m0['size_err_pct']:+.1f}% → {m1['size_err_pct']:+.1f}% ｜ "
                      + "  ".join(f"{b} {m0[b]:.1f}→{m1[b]:.1f}" for b, _, _ in BANDS), flush=True)
            rows.append(row)
    for group, members in (("写真だけ", idx["public"]), ("位置つき", idx.get("posed", {}))):
        for ds in ("tum", "arkit"):
            for label in ("モデルの姿勢", "向き=モデル・位置=正解"):
                sub = [r for r in rows if members.get(r["case"]) == ds and r["variant"] == label]
                if sub:
                    print(f"{group} {ds} {label}（{len(sub)} 本）: 大きさ|.| {np.mean([abs(r['none size_err_pct']) for r in sub]):.1f}% → {np.mean([abs(r['corr size_err_pct']) for r in sub]):.1f}% ｜ "
                          + "  ".join(f"{b} {np.nanmean([r[f'none {b}'] for r in sub]):.1f}→{np.nanmean([r[f'corr {b}'] for r in sub]):.1f}" for b, _, _ in BANDS))
    (BENCH / "self_triangulate.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
