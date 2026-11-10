"""AR の連続記録から、復元に使うコマを選ぶ（カメラの位置・向きと内部パラメータだけを使う。写真の中身は見ない）。

方針（docs/検証記録.md §20）: 隣どうしの視野の重なりが、どの間でも min_overlap 以上になるように、重なりが下回る直前のコマを次々に選ぶ
（重なり優先）。時間で等間隔に選ぶと、速く向きを変えた所で「つながらない間（重なり 0）」ができることがあった。
精度は等間隔と同程度で、凝った選び方（離れた視点・正面性）は良くならなかった。
枚数が max_frames を超えるときは、max_frames 枚以内で隣どうしの重なりの最小がいちばん大きくなるように選ぶ。

視野の重なり: コマ i の画面の格子点を「assumed_depth 先」に置いて、コマ j の画面に入る割合（i→j と j→i の小さい方）。
深度は使わない（スマホの多くは無い）。min_overlap の既定値は仮（TUM で精度の良かった選び方の、隣どうしの重なりの中央値 0.29 から）。
"""
from __future__ import annotations

import numpy as np

ASSUMED_DEPTH_M = 2.0
MIN_OVERLAP = 0.3
MAX_FRAMES = 40


def overlap_matrix(c2w: np.ndarray, K: np.ndarray, wh: tuple[int, int], assumed_depth: float = ASSUMED_DEPTH_M) -> np.ndarray:
    """c2w: n×4×4（カメラ→世界、OpenCV の向き）、K: n×3×3 または 3×3、wh: 画像の (幅, 高さ)。戻り値 n×n（対角は 1）。"""
    c2w = np.asarray(c2w, float)
    n = len(c2w)
    Ks = np.broadcast_to(np.asarray(K, float), (n, 3, 3))
    w, h = wh
    u, v = np.meshgrid(np.linspace(0.05, 0.95, 12) * w, np.linspace(0.05, 0.95, 9) * h)
    uv = np.c_[u.ravel(), v.ravel()]
    world = []
    for T, k in zip(c2w, Ks):
        pc = np.c_[(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(uv))] * assumed_depth
        world.append(pc @ T[:3, :3].T + T[:3, 3])
    M = np.ones((n, n))
    for j in range(n):
        R, t, k = c2w[j, :3, :3], c2w[j, :3, 3], Ks[j]
        for i in range(n):
            if i == j:
                continue
            pc = (world[i] - t) @ R
            z = pc[:, 2]
            x = pc[:, 0] / np.maximum(z, 1e-6) * k[0, 0] + k[0, 2]
            y = pc[:, 1] / np.maximum(z, 1e-6) * k[1, 1] + k[1, 2]
            M[i, j] = float(((z > 0.1) & (x >= 0) & (x < w) & (y >= 0) & (y < h)).mean())
    return np.minimum(M, M.T)


def chain(M: np.ndarray, theta: float) -> list[int]:
    """先頭から順に、前に選んだコマとの重なりが theta 以上で連続している範囲のいちばん先を選ぶ（範囲が無ければ次のコマ）。
    範囲を連続に限るのは、部屋を一周すると終わりのコマが始めのコマと同じ所を見るため（限らないと、途中を飛ばして終わりへ跳ぶ）。"""
    n, sel = len(M), [0]
    while sel[-1] < n - 1:
        last = sel[-1]
        k = last + 1
        while k + 1 < n and M[last, k + 1] >= theta:
            k += 1
        sel.append(k)
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
    if best is None:                      # どのしきい値でも n 枚を超える（候補が n より多く、ほとんど重ならない）: 等間隔
        return [int(x) for x in np.round(np.linspace(0, len(M) - 1, n))], 0.0
    s = list(best[0])
    while len(s) < n:
        gaps = [(M[a, b], i) for i, (a, b) in enumerate(zip(s, s[1:])) if b - a > 1]
        if not gaps:
            break
        _, i = min(gaps)
        s.insert(i + 1, (s[i] + s[i + 1]) // 2)
    return s, best[1]


def select_frames(c2w: np.ndarray, K: np.ndarray, wh: tuple[int, int], min_overlap: float = MIN_OVERLAP,
                  max_frames: int = MAX_FRAMES) -> dict:
    """隣どうしの重なりが min_overlap 以上になるようにコマを選ぶ。max_frames を超えるなら、その枚数で重なりの最小を最大にする。
    戻り値: {"idx": 選んだコマの番号, "overlap_min": 隣どうしの重なりの最小, "capped": 枚数の上限で重なりを下げたか}"""
    n = len(c2w)
    if n <= 2:
        return {"idx": list(range(n)), "overlap_min": None, "capped": False}
    M = overlap_matrix(c2w, K, wh)
    idx = chain(M, min_overlap)
    capped = len(idx) > max_frames
    if capped:
        idx, _ = fit_count(M, max_frames)
    cons = [float(M[a, b]) for a, b in zip(idx, idx[1:])]
    return {"idx": idx, "overlap_min": round(min(cons), 3), "capped": capped}
