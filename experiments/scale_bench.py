"""大きさ（全体の倍率）を何も測らずに決める方法を、保存済みの復元で比べる。再推論はしない。

材料: scale_cases.py（復元と正解）、moge_infer.py（MoGe の点）。どれも experiments/ext/scale_bench/ にある。
指標:
- 自宅（実測 4 か所）: 倍率を当てたあとの、実測との差の絶対値の平均（cm）。measure_home_1012.py の値と、デモの部屋ファイルの値の 2 通り。
- 公開データ（TUM・ARKitScenes、写真だけの復元）: 倍率を当てたあとの、写真をまたぐ 2 点の距離の誤差（中央値・cm）を
  0.5〜1.5m / 1.5〜3m / 3m 以上で。これが悪くならないことが下限。あわせて、写真の中の大きさの倍率（正解に対する比）。
使い方（server フォルダで）: uv run python ../experiments/scale_bench.py [方法 ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
BENCH = here / "ext" / "scale_bench"
MEASURED = {"door_width": 85.0, "door_height": 219.0, "closet_width": 102.0, "closet_panel": 226.0}
# デモの部屋ファイル（demo/home_corner_measured.roomfit.json）の 4 か所の長さ。復元は home_1012_avg と同じもの
DEMO_CM = {"closet_width": 106.0, "closet_panel": 234.7, "door_width": 89.8, "door_height": 226.5}
BANDS = (("0.5-1.5m", 0.5, 1.5), ("1.5-3m", 1.5, 3.0), (">3m", 3.0, np.inf))


def load_case(name: str) -> dict:
    c = np.load(BENCH / "cases" / f"{name}.npz")
    n = int(c["n"])
    case = {"name": name, "n": n, "K": c["K"], "ma_c2w": c["ma_c2w"], "mask": [c[f"mask{v}"] for v in range(n)], "ma": [c[f"ma{v}"] for v in range(n)]}
    if "gt_c2w" in c:
        case["gt_c2w"] = c["gt_c2w"]
        case["gt"] = [c[f"gt{v}"] for v in range(n)]
    for mode in ("fov", "free"):
        p = BENCH / f"moge2_{mode}" / f"{name}.npz"
        if p.exists():
            m = np.load(p)
            case[f"moge_{mode}"] = [m[f"pts{v}"] for v in range(n)]
    return case


def inview_ratio(A: np.ndarray, B: np.ndarray, ok: np.ndarray, rng, min_d=0.3, n_pairs=20000) -> float | None:
    """同じ写真の中の 2 点の距離の比（A ÷ B）の中央値。焦点距離の見積もりが違っても、物の大きさどうしで比べられる。"""
    idx = np.argwhere(ok)
    if len(idx) < 500:
        return None
    i, j = idx[rng.integers(0, len(idx), n_pairs)], idx[rng.integers(0, len(idx), n_pairs)]
    da = np.linalg.norm(A[i[:, 0], i[:, 1]] - A[j[:, 0], j[:, 1]], axis=1)
    db = np.linalg.norm(B[i[:, 0], i[:, 1]] - B[j[:, 0], j[:, 1]], axis=1)
    m = db > min_d
    return float(np.median(da[m] / db[m])) if m.sum() > 200 else None


def per_view(case: dict, key: str) -> np.ndarray:
    """写真ごとの「key の大きさ ÷ MapAnything の大きさ」。"""
    rng = np.random.default_rng(0)
    out = []
    for v in range(case["n"]):
        A, B = case[key][v], case["ma"][v]
        ok = case["mask"][v] & np.isfinite(A).all(-1) & np.isfinite(B).all(-1)
        r = inview_ratio(A, B, ok, rng)
        out.append(np.nan if r is None else r)
    return np.array(out)


def variant_ratio(c: dict, how: str) -> float:
    """run_variant.py で入力を変えて復元し直した結果の大きさ ÷ 元の復元の大きさ（写真の中の距離の比の中央値）。"""
    p = BENCH / "cases" / f"{c['name']}_{how}.npz"
    if not p.exists():
        raise KeyError(how)
    z = np.load(p)
    c = {**c, how: [z[f"ma{v}"] for v in range(c["n"])]}
    return float(np.nanmedian(per_view(c, how)))


METHODS = {
    "none": lambda c: 1.0,
    "moge_fov_median": lambda c: float(np.nanmedian(per_view(c, "moge_fov"))),
    "moge_free_median": lambda c: float(np.nanmedian(per_view(c, "moge_free"))),
    "moge_fov_geomean": lambda c: float(np.sqrt(np.nanmedian(per_view(c, "moge_fov")))),
    "moge_free_geomean": lambda c: float(np.sqrt(np.nanmedian(per_view(c, "moge_free")))),
    "intrinsics_given": lambda c: variant_ratio(c, "k"),
    "flip_mean": lambda c: float(np.sqrt(variant_ratio(c, "flip"))),
}


REPLACES = {"intrinsics_given": "k"}


def public_metrics(case: dict, s: float) -> dict:
    rng = np.random.default_rng(0)
    P_all, Q_all, V_all, g = [], [], [], []
    for v in range(case["n"]):
        P, Q = case["ma"][v], case["gt"][v]
        ok = case["mask"][v] & np.isfinite(P).all(-1) & np.isfinite(Q).all(-1) & (P[..., 2] > 0.05)
        idx = np.argwhere(ok)
        if len(idx) < 2000:
            continue
        r = inview_ratio(P, Q, ok, rng, min_d=0.5)
        if r is not None:
            g.append(r)
        sel = idx[rng.integers(0, len(idx), 3000)]
        Tm, Tg = case["ma_c2w"][v], case["gt_c2w"][v]
        P_all.append(P[sel[:, 0], sel[:, 1]].astype(np.float64) @ Tm[:3, :3].T + Tm[:3, 3])
        Q_all.append(Q[sel[:, 0], sel[:, 1]].astype(np.float64) @ Tg[:3, :3].T + Tg[:3, 3])
        V_all.append(np.full(len(sel), v))
    P, Q, V = map(np.concatenate, (P_all, Q_all, V_all))
    i, j = rng.integers(0, len(P), 400000), rng.integers(0, len(P), 400000)
    dp, dq = np.linalg.norm(P[i] - P[j], axis=1) * s, np.linalg.norm(Q[i] - Q[j], axis=1)
    out = {"size_err_pct": (s * float(np.median(g)) - 1) * 100}
    for name, lo, hi in BANDS:
        m = (V[i] != V[j]) & (dq > lo) & (dq <= hi)
        out[name] = float(np.median(np.abs(dp[m] - dq[m]))) * 100 if m.sum() >= 500 else np.nan
    return out


def home_metrics(name: str, s: float) -> dict:
    res = json.loads((BENCH / "home" / name / "measure_results.json").read_text(encoding="utf-8"))
    errs = {k: s * res[k]["recon_cm"] - m for k, m in MEASURED.items() if res.get(k)}
    out = {"mean_abs_cm": float(np.mean(np.abs(list(errs.values())))), **{k: round(v, 1) for k, v in errs.items()}}
    if name == "home_avg":
        out["demo_mean_abs_cm"] = float(np.mean([abs(s * DEMO_CM[k] - MEASURED[k]) for k in MEASURED]))
    return out


def main() -> None:
    methods = sys.argv[1:] or list(METHODS)
    idx = json.loads((BENCH / "cases" / "index.json").read_text(encoding="utf-8"))
    report = {}
    for meth in methods:
        rows, home = [], {}
        for name in idx["home"]:
            if name == "home_1012":   # home_avg と同じ復元（色の処理だけ違う）
                continue
            c = load_case(name)
            try:
                s = METHODS[meth](c)
            except KeyError:
                continue
            home[name] = {"s": round(s, 4), **home_metrics(name, s)}
        for name in idx["public"]:
            c = load_case(name)
            try:
                s = METHODS[meth](c)
            except KeyError:
                continue
            if meth in REPLACES:   # 復元そのものが変わる方法は、変わった後の復元で写真をまたぐ距離を測る
                c, s = load_case(f"{name}_{REPLACES[meth]}"), 1.0
            rows.append({"case": name, "s": round(s, 4), **{k: round(v, 2) for k, v in public_metrics(c, s).items()}})
        report[meth] = {"home": home, "public": rows}
        print(f"\n### {meth}")
        for k, v in home.items():
            extra = f" デモ {v['demo_mean_abs_cm']:.2f}" if "demo_mean_abs_cm" in v else ""
            print(f"  {k:10s} s={v['s']:.4f}  平均 {v['mean_abs_cm']:.2f}cm{extra}  " + " ".join(f"{kk}={vv:+.1f}" for kk, vv in v.items() if kk in MEASURED))
        for ds in ("tum", "arkit"):
            sub = [r for r in rows if idx["public"][r["case"]] == ds]
            if not sub:
                continue
            for r in sub:
                print(f"  {r['case']:14s} s={r['s']:.4f}  大きさ {r['size_err_pct']:+6.1f}%  " + "  ".join(f"{b} {r[b]:6.1f}" for b, _, _ in BANDS))
            print(f"  {ds} 平均{' ' * 9}       大きさ|.| {np.mean([abs(r['size_err_pct']) for r in sub]):5.1f}%  " + "  ".join(f"{b} {np.nanmean([r[b] for r in sub]):6.1f}" for b, _, _ in BANDS))
    (BENCH / "last_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
