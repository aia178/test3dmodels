"""実験（P2）: 連続画像から同じ枚数を選ぶとき、選び方で復元の精度がどれだけ変わるか。

撮影アプリで「開始・終了の 2 操作だけで、AR の連続記録からコマを自動で選ぶ」案（astra の報告書）の前提を確かめる。
選び方（どれも、アプリでも手に入る「カメラの位置・向き」と内部パラメータだけを使う。ここでは公開データの正解の位置で代用）:
  even   : 等間隔（時間で）
  overlap: 重なり優先。前に選んだコマとの視野の重なりがしきい値を下回る直前のコマを次に選ぶ（動きが速い所は密に、遅い所は疎に）
  mixed  : 重なり＋離れた視点＋正面性。overlap で選んだ N 枚を骨組みにし、各コマを、前後のコマとの重なりを 2 割以上は落とさない範囲で、
           前のコマから離れていて（視差・大きさの手がかり）、水平に近い向き（壁を正面から見やすい）のコマに置き換える
           （重なりの範囲の中から直接選ぶ方式は、先へ進まず 16 枚では部屋を回りきれなかった）
  overlap のしきい値は、ちょうど N 枚になるように探す。
視野の重なり: コマ i の画面の格子点を「2m 先」に置いて、コマ j の画面に入る割合（i→j と j→i の小さい方）。深度を使わない（スマホの多くは無い）。

答え合わせ（evaluate）: TUM は Kinect の深度とモーションキャプチャの位置、ARKitScenes はレーザー由来の深度と ARKit の位置
（ARKit の位置は答えの側にも入るので、写真をまたぐ距離は参考値。写真ごとの大きさは位置と無関係）。

使い方（server フォルダで）:
  uv run python ../experiments/frame_select.py select                      → 選んだコマを experiments/out/frame_select/ に書く（速い）
  uv run python ../experiments/frame_select.py run <tum|arkit> <even|overlap|mixed> [photo|pose] [ずらし=0]   → 復元して評価（CPU で 10 分前後）
    ずらし: 候補のコマの先頭をこの数だけ飛ばして選ぶ（少し違うコマを選んだときの結果の揺れを測るため）
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

N = 16
ASSUMED_DEPTH_M = 2.0
OUT = here / "out" / "frame_select"
TUM = here / "data" / "tum" / "rgbd_dataset_freiburg1_room"
ARKIT = here / "data" / "arkitscenes" / "42444946"


def load_tum(step: int = 5) -> list[dict]:
    from roomfit.arcore_import import quat_to_rot
    from tum_eval import associate, read_list

    pairs = associate(read_list(TUM / "rgb.txt"), read_list(TUM / "depth.txt"))
    gt = np.array([[float(x) for x in line.split()] for line in (TUM / "groundtruth.txt").read_text().splitlines() if not line.startswith("#")])
    K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1.0]])
    out = []
    for t, rgb, dep in pairs[::step]:
        k = int(np.argmin(np.abs(gt[:, 0] - t)))
        if abs(gt[k, 0] - t) > 0.02:
            continue
        c2w = np.eye(4); c2w[:3, :3] = quat_to_rot(gt[k, 4:8]); c2w[:3, 3] = gt[k, 1:4]
        out.append({"ts": t, "rgb": TUM / rgb, "depth": TUM / dep, "depth_scale": 5000.0, "c2w": c2w, "K": K, "wh": (640, 480)})
    return out


def load_arkit() -> list[dict]:
    from arkit_posed import load_frames

    return [{**f, "depth_scale": 1000.0, "wh": (640, 480)} for f in load_frames(ARKIT)]


def overlap_matrix(frames: list[dict]) -> np.ndarray:
    n = len(frames)
    w, h = frames[0]["wh"]
    u, v = np.meshgrid(np.linspace(0.05, 0.95, 12) * w, np.linspace(0.05, 0.95, 9) * h)
    uv = np.c_[u.ravel(), v.ravel()]
    world = []
    for f in frames:
        K = f["K"]
        pc = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))] * ASSUMED_DEPTH_M
        world.append(pc @ f["c2w"][:3, :3].T + f["c2w"][:3, 3])
    M = np.ones((n, n))
    for j, g in enumerate(frames):
        R, t, K = g["c2w"][:3, :3], g["c2w"][:3, 3], g["K"]
        for i in range(n):
            if i == j:
                continue
            pc = (world[i] - t) @ R
            z = pc[:, 2]
            x, y = pc[:, 0] / np.maximum(z, 1e-6) * K[0, 0] + K[0, 2], pc[:, 1] / np.maximum(z, 1e-6) * K[1, 1] + K[1, 2]
            M[i, j] = float(((z > 0.1) & (x >= 0) & (x < w) & (y >= 0) & (y < h)).mean())
    return np.minimum(M, M.T)


def up_axis(frames: list[dict]) -> np.ndarray:
    """カメラはおおむね立てて持つので、カメラの x・y 軸のうち世界で向きがそろっている方が上下（ARKitScenes は画像が横倒し）。"""
    ax = [np.mean([f["c2w"][:3, k] for f in frames], axis=0) for k in (0, 1)]
    a = ax[int(np.argmax([np.linalg.norm(x) for x in ax]))]
    return a / np.linalg.norm(a)


def chain(M: np.ndarray, theta: float) -> list[int]:
    """先頭から順に、前に選んだコマとの重なりが theta 以上で連続している範囲のいちばん先を選ぶ（範囲が無ければ次のコマ）。
    範囲を連続に限るのは、部屋を一周すると終わりのコマが始めのコマと同じ所を見るため（限らないと、途中を飛ばして終わりへ跳ぶ）。"""
    n, sel = len(M), [0]
    while sel[-1] < n - 1:
        last = sel[-1]
        k = last + 1
        while k + 1 < n and M[last, k + 1] >= theta:
            k += 1
        window = list(range(last + 1, k + 1))
        sel.append(window[-1])
    return sel


def fit_count(M: np.ndarray, n: int) -> tuple[list[int], float]:
    """n 枚以内で選べる最大のしきい値（＝隣どうしの重なりの最小をできるだけ大きくする）を探し、足りない枚数は、
    隣どうしの重なりがいちばん小さい間へ 1 枚ずつ足す。"""
    best = None
    # 部屋一周を 16 枚にすると、隣どうしの重なりはかなり小さくなる（TUM で等間隔だと 0 の所もある）ので、小さい値から探す
    for theta in np.r_[np.linspace(0.002, 0.05, 49), np.linspace(0.055, 0.95, 180)]:
        s = chain(M, theta)
        if len(s) <= n:
            best = (s, float(theta))
    s = list(best[0])
    while len(s) < n:
        gaps = [(M[a, b], i) for i, (a, b) in enumerate(zip(s, s[1:])) if b - a > 1]
        _, i = min(gaps)
        s.insert(i + 1, (s[i] + s[i + 1]) // 2)
    return s, best[1]


def select(frames: list[dict], how: str, n: int = N) -> dict:
    M = overlap_matrix(frames)
    if how == "even":
        idx, theta = [int(x) for x in np.round(np.linspace(0, len(frames) - 1, n))], None
    elif how == "overlap":
        idx, theta = fit_count(M, n)
    elif how == "mixed":
        up = up_axis(frames)
        P = np.array([f["c2w"][:3, 3] for f in frames])
        frontal = np.array([1 - abs(float(f["c2w"][:3, 2] @ up)) for f in frames])
        idx, theta = fit_count(M, n)
        for i in range(1, n - 1):
            a, b = idx[i - 1], idx[i + 1]
            tau = 0.8 * min(M[a, idx[i]], M[idx[i], b])

            def score(c):
                # 離れた視点（前のコマからの距離。0.5m で頭打ち）と、水平に近い向きを半々で
                return 0.5 * min(np.linalg.norm(P[c] - P[a]) / 0.5, 1.0) + 0.5 * frontal[c]
            cands = [c for c in range(a + 1, b) if M[a, c] >= tau and M[c, b] >= tau] or [idx[i]]
            idx[i] = max(cands, key=score)
    else:
        raise ValueError(how)
    P = np.array([frames[i]["c2w"][:3, 3] for i in idx])
    up = up_axis(frames)
    cons = [float(M[a, b]) for a, b in zip(idx, idx[1:])]
    # どのコマとも重ならないコマがないか（いちばん重なる相手との重なり）
    Ms = M[np.ix_(idx, idx)].copy(); np.fill_diagonal(Ms, 0)
    return {"how": how, "idx": idx, "theta": theta, "seconds": [round(frames[i]["ts"] - frames[0]["ts"], 2) for i in idx],
            "consecutive_overlap_min": round(min(cons), 3), "consecutive_overlap_median": round(float(np.median(cons)), 3),
            "best_partner_overlap_min": round(float(Ms.max(1).min()), 3),
            "step_m_median": round(float(np.median(np.linalg.norm(np.diff(P, axis=0), axis=1))), 3),
            "extent_m": round(float(np.linalg.norm(P[:, None] - P[None], axis=-1).max()), 2),
            "frontal_mean": round(float(np.mean([1 - abs(float(frames[i]["c2w"][:3, 2] @ up)) for i in idx])), 3)}


def evaluate(raw: dict, frames: list[dict]) -> dict:
    """写真ごとの大きさ（復元の奥行き ÷ 正解の奥行き）と、写真をまたぐ 2 点の距離の誤差。"""
    from roomfit.worker import view_grids
    from tum_eval import model_grid_to_source

    rng = np.random.default_rng(0)
    rows, P_all, Q_all, V_all = [], [], [], []
    for v, ((mask, xyz), fr) in enumerate(zip(view_grids(raw), frames)):
        T = raw["cam2world"][v]
        pc = (xyz.astype(np.float64) - T[:3, 3]) @ T[:3, :3]
        gt_full = np.asarray(Image.open(fr["depth"])).astype(np.float64) / fr["depth_scale"]
        w, h = fr["wh"]
        xs, ys = model_grid_to_source(mask.shape[0], mask.shape[1], h, w)
        sx, sy = gt_full.shape[1] / w, gt_full.shape[0] / h
        gt = gt_full[np.clip((ys * sy).astype(int), 0, gt_full.shape[0] - 1), np.clip((xs * sx).astype(int), 0, gt_full.shape[1] - 1)]
        ok = mask & (gt > 0.3) & (gt < 6.0) & np.isfinite(pc).all(-1) & (pc[..., 2] > 0.05)
        if ok.sum() < 2000:
            rows.append({"view": v, "n": int(ok.sum())})
            continue
        r = pc[..., 2][ok] / gt[ok]
        rows.append({"view": v, "scale": round(float(np.median(r)), 4), "depth_m": round(float(np.median(gt[ok])), 2)})
        K = fr["K"]
        Qc = np.stack([(xs - K[0, 2]) / K[0, 0] * gt, (ys - K[1, 2]) / K[1, 1] * gt, gt], -1)
        idx = np.argwhere(ok)
        sel = idx[rng.integers(0, len(idx), 3000)]
        P_all.append(xyz[sel[:, 0], sel[:, 1]].astype(np.float64))
        Q_all.append(Qc[sel[:, 0], sel[:, 1]] @ fr["c2w"][:3, :3].T + fr["c2w"][:3, 3])
        V_all.append(np.full(len(sel), v))
    sc = np.array([x["scale"] for x in rows if "scale" in x])
    res = {"views": rows, "scale_median": round(float(np.median(sc)), 4),
           "scale_iqr_pct": round(float((np.percentile(sc, 75) - np.percentile(sc, 25)) / np.median(sc) * 100), 1),
           "scale_spread_pct": round(float((sc.max() - sc.min()) / sc.mean() * 100), 1)}
    P, Q, V = map(np.concatenate, (P_all, Q_all, V_all))
    i, j = rng.integers(0, len(P), 400000), rng.integers(0, len(P), 400000)
    dp, dq = np.linalg.norm(P[i] - P[j], axis=1), np.linalg.norm(Q[i] - Q[j], axis=1)
    # 写真をまたぐ距離は、全体の大きさ（1 つの倍率）を合わせたあとの誤差も出す。全体の大きさは 1 か所測れば合わせられる（§10）
    g = float(np.median(dp[(V[i] != V[j]) & (dq > 0.5)] / dq[(V[i] != V[j]) & (dq > 0.5)]))
    res["global_scale"] = round(g, 4)
    for name, m in (("cross_0.5-1.5m", (V[i] != V[j]) & (dq > 0.5) & (dq <= 1.5)), ("cross_1.5-3m", (V[i] != V[j]) & (dq > 1.5) & (dq <= 3)),
                    ("cross_>3m", (V[i] != V[j]) & (dq > 3))):
        if m.sum() < 500:
            continue
        res[name] = {"pairs": int(m.sum()), "abs_error_cm_median": round(float(np.median(np.abs(dp[m] - dq[m]))) * 100, 1),
                     "abs_error_cm_after_scale": round(float(np.median(np.abs(dp[m] / g - dq[m]))) * 100, 1)}
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "select":
        allsel = {}
        for name, loader in (("tum", load_tum), ("arkit", load_arkit)):
            frames = loader()
            allsel[name] = {"candidates": len(frames), "seconds": round(frames[-1]["ts"] - frames[0]["ts"], 1)}
            for how in ("even", "overlap", "mixed"):
                s = select(frames, how)
                allsel[name][how] = s
                print(name, how, json.dumps({k: v for k, v in s.items() if k not in ("idx", "seconds")}, ensure_ascii=False), flush=True)
        (OUT / "selections.json").write_text(json.dumps(allsel, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    import torch
    from mapanything.utils.image import preprocess_inputs

    from roomfit.reconstruct import run_mapanything, save_raw
    from roomfit.worker import capture_quality

    data, how = sys.argv[2], sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else "photo"
    shift = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    frames_all = load_tum() if data == "tum" else load_arkit()
    if shift == 0:
        sel = json.loads((OUT / "selections.json").read_text(encoding="utf-8"))[data][how]
    else:
        sel = select(frames_all[shift:], how)
        sel["idx"] = [i + shift for i in sel["idx"]]
    sel["shift"] = shift
    frames = [frames_all[i] for i in sel["idx"]]
    out = OUT / (f"{data}_{how}_{mode}" + (f"_s{shift}" if shift else ""))
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
    t = time.time()
    raw = run_mapanything(paths, views=preprocess_inputs(views)) if mode == "pose" else run_mapanything(paths)
    raw["meta"].update({"sec_total": round(time.time() - t, 1), "input_mode": mode, "selection": how})
    save_raw(raw, out / "raw.npz")
    res = evaluate(raw, frames)
    res["capture_quality"] = capture_quality(raw)
    res["sec_inference"] = raw["meta"]["sec_inference"]
    res["selection"] = {k: v for k, v in sel.items() if k != "idx"}
    (out / "eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(data, how, mode, shift, json.dumps({k: v for k, v in res.items() if k not in ("views", "selection")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
