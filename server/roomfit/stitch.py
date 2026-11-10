"""写真が多いときに、いくつかの範囲に分けて復元し、共通の写真でつなぐ。

1 回の復元は、この構成（CPU・メモリ 16GB）では 20 枚前後が上限で、広い範囲を少ない枚数で撮ると写真どうしのつなぎ方が崩れる。
そこで、撮った順に並んだ写真を、前の範囲の最後の数枚を重ねながら区切り、範囲ごとに復元する。
同じ写真の同じ画素は同じ場所を指すので、その対応から「後の範囲の座標 → 前の範囲の座標」の変換（回転・移動・拡大縮小）を求め、
そのあと、2 つの範囲の点全体どうしで回転と移動だけを合わせ直す（大きさは共通の写真で決めた値のまま）。
公開データでの確かめは docs/検証記録.md §14（experiments/tum_chunk.py, tum_chain.py）。

前提: 写真が撮った順に並んでいること。順番がばらばらだと、共通の写真が隣の範囲とつながらない。
"""
from __future__ import annotations

import numpy as np

CHUNK_SIZE = 13      # 1 範囲の枚数（共通の写真を含む）。この PC で 1 範囲 約 8 分・メモリ約 8GB
N_SHARED = 3         # 隣の範囲と共通にする枚数
SINGLE_MAX = 16      # これ以下の枚数なら分けずに 1 回で復元する
LINK_WARN_MM = 60.0  # つなぎ目の残差（中央値）がこれを超えたら、つなぎが怪しいと伝える。大きさを変えずにつなぐので、大きさの差も残差に入る（公開データで、合った組 41mm・合わない組 55〜114mm）
LINK_SIZE_GAP_WARN_PCT = 10.0  # 隣の範囲と、共通の写真で見た大きさが 10% を超えて違ったら、同じく伝える（公開データ 1 本から置いた仮の値）


def plan_chunks(n: int, size: int = CHUNK_SIZE, shared: int = N_SHARED, single_max: int = SINGLE_MAX) -> list[list[int]]:
    """写真の番号（撮った順）を範囲に分ける。隣どうしは shared 枚を共有する。最後の範囲が小さくなりすぎないよう、ほぼ均等に割る。"""
    if n <= single_max:
        return [list(range(n))]
    step = size - shared
    k = max(2, int(np.ceil((n - shared) / step)))
    # k 個の範囲で n 枚を覆う: 各範囲の新しい枚数をほぼ均等に配る
    new = [(n - shared) // k + (1 if i < (n - shared) % k else 0) for i in range(k)]
    chunks, start = [], 0
    for i, m in enumerate(new):
        end = start + shared + m
        chunks.append(list(range(start, end)))
        start = end - shared
    return chunks


def umeyama(src: np.ndarray, dst: np.ndarray, scale: float | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """dst ≈ s·R·src + t となる拡大縮小 s・回転 R・移動 t（最小二乗）。scale を渡すと、s をその値に固定して R と t だけ求める。"""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()) if scale is None else float(scale)
    return s, R, md - s * R @ ms


def robust_similarity(src: np.ndarray, dst: np.ndarray, rounds: int = 6, scale: float | None = None) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """はずれを除きながら変換を求める。戻り値の最後は、残った対応での残差 [m]。scale を渡すと拡大縮小はその値に固定する。"""
    keep = np.ones(len(src), bool)
    for _ in range(rounds):
        s, R, t = umeyama(src[keep], dst[keep], scale)
        res = np.linalg.norm(src @ R.T * s + t - dst, axis=1)
        keep = res < max(0.01, 2.5 * float(np.median(res[keep])))
    return s, R, t, res[keep]


def icp_rigid(src: np.ndarray, dst: np.ndarray, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """点全体どうしの合わせ直し（回転と移動だけ）。src は大まかに合わせた後の座標。戻り値は src に掛ける追加の回転と移動。"""
    from scipy.spatial import cKDTree

    tree = cKDTree(dst)
    R, t = np.eye(3), np.zeros(3)
    for it in range(iters):
        cur = src @ R.T + t
        d, nn = tree.query(cur)
        keep = d < max(0.03, 0.15 * (1 - it / iters))       # 15cm から 3cm へ、だんだん厳しく
        if keep.sum() < 2000:
            break
        a, b = cur[keep], dst[nn[keep]]
        ma, mb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((b - mb).T @ (a - ma))
        D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
        dR = U @ D @ Vt
        R, t = dR @ R, dR @ t + (mb - dR @ ma)
    return R, t


def _skew(v: np.ndarray) -> np.ndarray:
    """(N,3) のベクトルそれぞれの外積行列 (N,3,3)。"""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], -1), np.stack([v[:, 2], z, -v[:, 0]], -1), np.stack([-v[:, 1], v[:, 0], z], -1)], -2)


def _rodrigues(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = _skew((w / th)[None])[0]
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def global_refine(clouds: list[np.ndarray], Rs: list[np.ndarray], ts: list[np.ndarray], rounds: int = 12,
                  reach: tuple[float, float] = (0.30, 0.05), min_pairs: int = 1500) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """すべての範囲を同時に合わせ直す（回転と移動だけ。最初の範囲は動かさない）。

    隣どうしを順につなぐだけだと、ずれが先へ行くほど溜まる。部屋を一周して戻ってくると、最初と最後が同じ場所を写しているのに合わない。
    ここでは、重なっている範囲の組すべてで近い点どうしを組にし、全部の組のずれの合計が減るように、各範囲を少しずつ動かす。
    clouds[k] は範囲 k の点（その範囲の座標、大きさは合わせた後）、Rs[k], ts[k] は範囲 k → 全体の座標への回転と移動（順につないだ結果）。
    """
    from scipy.spatial import cKDTree

    K = len(clouds)
    Rs, ts = [r.copy() for r in Rs], [t.copy() for t in ts]
    used: list[dict] = []
    if K < 3:
        return Rs, ts, used
    for it in range(rounds):
        thr = reach[0] + (reach[1] - reach[0]) * it / max(1, rounds - 1)
        world = [c @ r.T + t for c, r, t in zip(clouds, Rs, ts)]
        trees = [cKDTree(w) for w in world]
        H, g = np.zeros((6 * K, 6 * K)), np.zeros(6 * K)
        used = []
        for a in range(K):
            for b in range(a + 1, K):
                q = world[b][::3]
                d, nn = trees[a].query(q, distance_upper_bound=thr)
                ok = np.isfinite(d)
                if ok.sum() < min_pairs:
                    continue
                pa, pb = world[a][nn[ok]], q[ok]
                used.append({"a": a, "b": b, "pairs": int(ok.sum()), "median_mm": round(float(np.median(d[ok])) * 1000, 1)})
                # 残差 r = (pa + ωa×pa + δa) − (pb + ωb×pb + δb)。ω×p = −[p]× ω
                r = pa - pb
                Ja = np.concatenate([-_skew(pa), np.broadcast_to(np.eye(3), (len(pa), 3, 3))], -1)   # (N,3,6)
                Jb = np.concatenate([-_skew(pb), np.broadcast_to(np.eye(3), (len(pb), 3, 3))], -1)
                sa, sb = slice(6 * a, 6 * a + 6), slice(6 * b, 6 * b + 6)
                H[sa, sa] += np.einsum("nij,nik->jk", Ja, Ja); H[sb, sb] += np.einsum("nij,nik->jk", Jb, Jb)
                hab = np.einsum("nij,nik->jk", Ja, Jb)
                H[sa, sb] -= hab; H[sb, sa] -= hab.T
                g[sa] += np.einsum("nij,ni->j", Ja, r); g[sb] -= np.einsum("nij,ni->j", Jb, r)
        if not used:
            break
        # 最初の範囲は固定。どの組にも入らなかった範囲は動かさない（対角に小さな値を足して解を 0 にする）
        free = np.arange(6, 6 * K)
        Hf = H[np.ix_(free, free)] + np.eye(len(free)) * 1e-6 * max(1.0, float(np.trace(H)) / (6 * K))
        x = np.zeros(6 * K)
        x[free] = np.linalg.solve(Hf, -g[free])
        for k in range(1, K):
            dR = _rodrigues(x[6 * k:6 * k + 3])
            Rs[k], ts[k] = dR @ Rs[k], dR @ ts[k] + x[6 * k + 3:6 * k + 6]
    return Rs, ts, used


def _split_views(raw: dict) -> list[dict]:
    """raw を写真ごとに分ける（点・色・信頼度・有効画素のビット列）。"""
    out, bit0, pt0 = [], 0, 0
    for v, (h, w) in enumerate(raw["image_hw"]):
        nbytes = (int(h) * int(w) + 7) // 8
        bits = raw["valid_bits"][bit0:bit0 + nbytes]
        mask = np.unpackbits(bits)[: int(h) * int(w)].astype(bool)
        k = int(mask.sum())
        out.append({"bits": bits, "mask": mask, "points": raw["points"][pt0:pt0 + k], "colors": raw["colors"][pt0:pt0 + k],
                    "conf": raw["conf"][pt0:pt0 + k], "cam2world": raw["cam2world"][v], "intrinsics": raw["intrinsics"][v],
                    "hw": raw["image_hw"][v], "image": raw["view_images"][v]})
        bit0, pt0 = bit0 + nbytes, pt0 + k
    return out


def _shared_pairs(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """同じ写真を 2 つの範囲で復元したもの a, b から、両方で有効な画素の 3D 位置の組（b 側, a 側）を返す。"""
    both = a["mask"] & b["mask"]
    ia = np.cumsum(a["mask"]) - 1
    ib = np.cumsum(b["mask"]) - 1
    return b["points"][ib[both]].astype(np.float64), a["points"][ia[both]].astype(np.float64)


def merge_raws(raws: list[dict], shared: int = N_SHARED, match_scale: bool = False) -> dict:
    """範囲ごとの復元結果を、最初の範囲の座標へそろえて 1 つにまとめる。共通の写真は、前の範囲のものを残す。

    match_scale=False（既定）: 各範囲の大きさは変えず、回転と移動だけでつなぐ。
      公開データでは、範囲ごとの大きさは実測に対して 0.95〜1.10 とばらつくが、共通の 3 枚から出した大きさの比はそれ以上に当てにならず
      （同じ範囲の中でも写真ごとに 0.88〜1.20）、順に掛け合わせると 6 範囲で 22% 縮んだ。変えない方が、端から端の誤差が半分以下だった。
    """
    if len(raws) == 1:
        return raws[0]
    S, R, T = 1.0, np.eye(3), np.zeros(3)         # 今の範囲の座標 → 最初の範囲の座標
    held: list[dict] = []
    links, prev_views, prev_cloud = [], None, None
    for k, raw in enumerate(raws):
        views = _split_views(raw)
        cloud = np.concatenate([v["points"][::23] for v in views]).astype(np.float64)
        if k > 0:
            src, dst = [], []
            for a, b in zip(prev_views[-shared:], views[:shared]):
                if tuple(a["hw"]) != tuple(b["hw"]):
                    raise ValueError("共通の写真の大きさが範囲によって違います")
                sb, sa = _shared_pairs(a, b)
                src.append(sb); dst.append(sa)
            src, dst = np.concatenate(src), np.concatenate(dst)
            if len(src) < 5000:
                raise ValueError("共通の写真に、両方の範囲で復元できた画素がほとんどありません")
            step = max(1, len(src) // 150000)
            seen, r, t, res = robust_similarity(src[::step], dst[::step])
            s = seen
            if not match_scale:
                s, r, t, res = robust_similarity(src[::step], dst[::step], scale=1.0)
            dR, dt = icp_rigid(cloud @ r.T * s + t, prev_cloud)
            r, t = dR @ r, dR @ t + dt
            links.append({"chunk": k, "size_ratio_seen": round(float(seen), 4), "scale_to_prev": round(float(s), 4),
                          "residual_median_mm": round(float(np.median(res)) * 1000, 1),
                          "icp_turn_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))), 2)})
            S, R, T = S * s, R @ r, S * (R @ t) + T
        prev_views, prev_cloud = views, cloud
        held.append({"k": k, "views": views[(shared if k > 0 else 0):], "cloud": cloud * S, "S": S, "R": R, "T": T})
    Rs, Ts, pairs = global_refine([h["cloud"] for h in held], [h["R"] for h in held], [h["T"] for h in held])
    merged: list[dict] = []
    for h, R, T in zip(held, Rs, Ts):
        for v in h["views"]:
            c2w = v["cam2world"].copy()
            c2w[:3, :3] = R @ c2w[:3, :3]
            c2w[:3, 3] = h["S"] * (R @ v["cam2world"][:3, 3]) + T
            merged.append({**v, "points": (v["points"].astype(np.float64) @ R.T * h["S"] + T).astype(np.float32), "cam2world": c2w, "chunk": h["k"]})
    meta = dict(raws[0]["meta"])
    meta.update({
        "n_views": len(merged),
        "sec_inference": round(sum(r["meta"].get("sec_inference", 0) for r in raws), 1),
        "chunks": {"count": len(raws), "shared": shared, "sizes": [int(len(r["cam2world"])) for r in raws], "links": links, "match_scale": match_scale,
                   "refine_pairs": [[p["a"], p["b"]] for p in pairs], "view_chunk": [v["chunk"] for v in merged]},
    })
    return {
        "points": np.concatenate([v["points"] for v in merged]),
        "colors": np.concatenate([v["colors"] for v in merged]),
        "conf": np.concatenate([v["conf"] for v in merged]),
        "cam_index": np.concatenate([np.full(len(v["points"]), i, np.int32) for i, v in enumerate(merged)]),
        "cam2world": np.stack([v["cam2world"] for v in merged]).astype(np.float64),
        "intrinsics": np.stack([v["intrinsics"] for v in merged]).astype(np.float64),
        "image_hw": np.stack([v["hw"] for v in merged]),
        "valid_bits": np.concatenate([v["bits"] for v in merged]),
        "view_images": np.stack([v["image"] for v in merged]),
        "meta": meta,
    }
