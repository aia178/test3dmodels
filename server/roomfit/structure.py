"""点群とカメラ姿勢から、家具配置に使える部屋形状（床・壁の輪郭・柱・未確認箇所）を作る。

座標系（docs/座標系と単位.md と一致させること）
- 入力 world: 復元モデルの世界座標。カメラ姿勢は OpenCV 規約 (+X右, +Y下, +Z前) の cam2world。
- 出力 room : 右手系・Y上・床が y=0・単位は m。床平面は X-Z。壁の主方向を X/Z 軸へ合わせる。
- world→room は回転+平行移動のみ（det=+1）。鏡映は入れない。左右は反転しない。

重要な方針
- 壁として観測できた境界と、撮影されておらず単に観測が途切れた境界を区別する (observed / unobserved)。
- 取得できていない凹凸を「存在しない」とは扱わない。unobserved の辺は判定側で要確認になる。
- 簡略化は snap_tol 未満の段差だけをまとめる。それ以上の段差（柱・出っ張り）は残す。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


class StructureError(Exception):
    """利用者に伝えられる理由つきの失敗。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class StructureParams:
    cell: float = 0.025          # 2D グリッドの 1 マス [m]
    height_bin: float = 0.10     # 壁判定用の高さ方向ビン [m]
    wall_coverage: float = 0.5   # 高さ方向にこの割合以上点があるマスを縦の構造物とみなす
    snap_tol: float = 0.04       # これ未満の段差は同一の壁面としてまとめる [m]
    min_edge: float = 0.05       # これより短い辺は隣へ吸収する [m]
    max_rays: int = 150_000


@dataclass
class StructureResult:
    T_room_from_world: np.ndarray            # 4x4
    outline: list[list[float]]               # [[x,z],...] 反時計回り（上から見て）, m
    edge_status: list[str]                   # 辺 i は outline[i]→outline[i+1]。 "observed" | "unobserved"
    edge_support: list[int]                  # その辺を支える点の数
    ceiling_height: float | None
    ceiling_observed: bool
    obstacles: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 上方向・床

def estimate_up_from_cameras(cam2world: np.ndarray) -> np.ndarray:
    """カメラの画像上方向（OpenCV では -Y）の平均を上方向の初期値にする。"""
    ups = -cam2world[:, :3, 1]
    up = ups.mean(axis=0)
    n = np.linalg.norm(up)
    if n < 0.3:
        raise StructureError(
            "up_ambiguous",
            "写真の向きがばらばらで、上下方向を決められませんでした。スマホを縦か横にそろえて撮影してください。",
        )
    return up / n


def _fit_plane(points: np.ndarray, up: np.ndarray, iters: int = 4, d0: float | None = None,
               th_start: float = 0.08) -> tuple[np.ndarray, float, int]:
    """up に近い法線の平面を最小二乗で当てる。戻り値 (法線, 高さ d : n·p=d, 内点数)。
    d0: 平面の高さの初期値（高さの分布の峰）。与えないと点の中央値から始めるが、天井のように
        すぐ下に壁の点が大量にある場合は中央値が壁側へ引かれるので、峰の位置を渡すこと。"""
    n = up.copy()
    d = float(np.median(points @ n)) if d0 is None else float(d0)
    inl = points
    for th in np.linspace(th_start, 0.025, iters):
        r = points @ n - d
        inl = points[np.abs(r) < th]
        if len(inl) < 50:
            break
        c = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
        nn = vt[-1]
        if nn @ up < 0:
            nn = -nn
        n, d = nn, float(nn @ c)
    return n, d, len(inl)


def _height_peaks(h: np.ndarray, bin_size: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.percentile(h, [0.5, 99.5])
    bins = max(10, int((hi - lo) / bin_size))
    hist, edges = np.histogram(h, bins=bins, range=(lo, hi))
    hist = ndimage.uniform_filter1d(hist.astype(float), 3)
    return hist, 0.5 * (edges[:-1] + edges[1:])


def find_floor_and_ceiling(points: np.ndarray, up0: np.ndarray):
    h = points @ up0
    hist, centers = _height_peaks(h)
    strong = hist >= 0.15 * hist.max()
    if not strong.any():
        raise StructureError("no_floor", "床を見つけられませんでした。床が写るように撮影してください。")
    idx = np.flatnonzero(strong)
    # 最も低い「強い」峰の極大を床とする
    lo_i = idx[0]
    j = lo_i
    while j + 1 < len(hist) and hist[j + 1] >= hist[j]:
        j += 1
    floor_h = centers[j]
    near = points[np.abs(h - floor_h) < 0.10]
    n, d, n_in = _fit_plane(near, up0)
    ang = np.degrees(np.arccos(np.clip(n @ up0, -1, 1)))
    if n_in < 300 or ang > 25:
        raise StructureError(
            "no_floor",
            "床が十分に写っていないため、部屋の水平を決められませんでした。床と壁の境目が写る写真を追加してください。",
        )
    h2 = points @ n - d
    hist2, centers2 = _height_peaks(h2)
    ceiling = None
    ceiling_observed = False
    top = np.flatnonzero((hist2 >= 0.10 * hist2.max()) & (centers2 > 1.5))
    if len(top):
        k = top[-1]
        while k - 1 >= 0 and hist2[k - 1] >= hist2[k]:
            k -= 1
        cand = points[np.abs(h2 - centers2[k]) < 0.10]
        if len(cand) >= 300:
            # 峰の高さから始め、最初から狭い幅で当てる（壁の上端の点を天井に混ぜない）
            cn, cd, cin = _fit_plane(cand, n, d0=centers2[k] + d, th_start=0.04)
            cang = np.degrees(np.arccos(np.clip(cn @ n, -1, 1)))
            # 棚の天板などを天井と取り違えないよう、水平方向に広く写っていることを求める
            a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 0, 1.0])
            e1 = a - (a @ n) * n
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)

            def _cells(q: np.ndarray) -> int:
                return len(np.unique(np.floor(np.stack([q @ e1, q @ e2], axis=1) / 0.1).astype(np.int64), axis=0))

            top_inl = cand[np.abs(cand @ cn - cd) < 0.04]
            coverage = _cells(top_inl) / max(1, _cells(points))
            if cin >= 300 and cang < 8 and coverage >= 0.2:
                # 高さは「天井の点が床の平面からどれだけ上か」で測る。天井の平面の法線は床とわずかに違うので、
                # 2 つの平面の係数の差（cd - d）は高さにならない。
                ceiling = float(np.median(top_inl @ n - d))
                ceiling_observed = True
    return n, d, ceiling, ceiling_observed, {"floor_inliers": int(n_in), "floor_tilt_from_camera_up_deg": float(ang)}


def rotation_up_to_y(up: np.ndarray) -> np.ndarray:
    """up を +Y へ写す固有回転 (det=+1)。行が room の x,y,z 軸（world 表現）。"""
    y = up / np.linalg.norm(up)
    a = np.array([1.0, 0, 0]) if abs(y[0]) < 0.9 else np.array([0, 0, 1.0])
    x = a - (a @ y) * y
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    R = np.stack([x, y, z])
    assert np.linalg.det(R) > 0.999
    return R


def dominant_yaw(xz: np.ndarray) -> float:
    """壁点の X/Z ヒストグラムが最も鋭くなる回転角 [rad] (0〜90°) を探す。"""
    if len(xz) > 60_000:
        xz = xz[np.random.default_rng(0).choice(len(xz), 60_000, replace=False)]
    best, best_a = None, 0.0
    for stage, (lo, hi, step) in enumerate([(0, 90, 1.0), (None, None, 0.1)]):
        if stage == 1:
            lo, hi = np.degrees(best_a) - 1.0, np.degrees(best_a) + 1.0
            best = None
        for deg in np.arange(lo, hi, step):
            a = np.radians(deg)
            c, s = np.cos(a), np.sin(a)
            u = xz[:, 0] * c - xz[:, 1] * s
            v = xz[:, 0] * s + xz[:, 1] * c
            score = 0.0
            for w in (u, v):
                hist, _ = np.histogram(w, bins=max(8, int(np.ptp(w) / 0.03)))
                p = hist / max(1, hist.sum())
                score += float((p * p).sum())
            if best is None or score > best:
                best, best_a = score, a
    return best_a


def _largest_vertical_plane(xz: np.ndarray, rng, iters: int = 250, th: float = 0.015):
    """床に垂直な面（上から見ると直線）のうち、最も多くの点が乗るものを探す。戻り値 (上から見た法線, 点, 乗った点の数)。"""
    best, bn, bp = 0, None, None
    for _ in range(iters):
        a, b = xz[rng.choice(len(xz), 2, replace=False)]
        t = b - a
        ln = np.linalg.norm(t)
        if ln < 0.3:
            continue
        n = np.array([-t[1], t[0]]) / ln
        cnt = int((np.abs((xz - a) @ n) < th).sum())
        if cnt > best:
            best, bn, bp = cnt, n, a
    if bn is None:
        return None
    inl = xz[np.abs((xz - bp) @ bn) < th]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    return vt[1], c, len(inl)


def wall_corner_shear(xz_wall: np.ndarray, max_dev_deg: float = 10.0) -> tuple[np.ndarray, dict]:
    """2 つの主な壁のなす角が 90° からずれていたら、直角に直す 2x2 の変換を返す。

    実データ（docs/検証記録.md §10）で、壁の上の長さは 1% 以内で合うのに、壁どうしの角は実行ごとに 85〜89° とぶれた。
    そこで、どちらの壁についても「壁に沿った長さ」は変えず、角だけを 90° にする（せん断）。
    ずれが max_dev_deg を超える場合は、もともと直角でない部屋の可能性があるので何もしない。
    """
    rng = np.random.default_rng(0)
    if len(xz_wall) > 40_000:
        xz_wall = xz_wall[rng.choice(len(xz_wall), 40_000, replace=False)]
    info = {"wall_corner_angle_deg": None, "rectified": False}
    first = _largest_vertical_plane(xz_wall, rng)
    if first is None:
        return np.eye(2), info
    n1, c1, cnt1 = first
    rest = xz_wall[np.abs((xz_wall - c1) @ n1) > 0.08]
    second = None
    for _ in range(4):  # 1 枚目とほぼ平行な面（同じ壁の扉や出っ張り）は飛ばして、向きの違う壁を探す
        if len(rest) < 500:
            break
        cand = _largest_vertical_plane(rest, rng)
        if cand is None:
            break
        if abs(cand[0] @ n1) < 0.6:
            second = cand
            break
        rest = rest[np.abs((rest - cand[1]) @ cand[0]) > 0.08]
    if second is None or second[2] < max(800, 0.1 * cnt1):
        return np.eye(2), info
    n2 = second[0]
    t1 = np.array([-n1[1], n1[0]])
    t2 = np.array([-n2[1], n2[0]])
    angle = float(np.degrees(np.arccos(np.clip(abs(n1 @ n2), 0, 1))))  # 壁どうしのなす角（90 が直角）
    info["wall_corner_angle_deg"] = angle
    if abs(90 - angle) < 0.3 or abs(90 - angle) > max_dev_deg:
        return np.eye(2), info
    e2 = np.array([-t1[1], t1[0]])
    if e2 @ t2 < 0:
        e2 = -e2
    A = np.stack([t1, e2], axis=1) @ np.linalg.inv(np.stack([t1, t2], axis=1))  # A t1 = t1, A t2 = e2（どちらも長さ 1 のまま）
    info["rectified"] = True
    return A, info


# ---------------------------------------------------------------- グリッド

def _carve_free(cams_xz: np.ndarray, pts_xz: np.ndarray, cam_idx: np.ndarray, origin: np.ndarray,
                cell: float, shape: tuple[int, int]) -> np.ndarray:
    """カメラから各点への視線が通ったマスを「見通せた＝空いている」とする。"""
    free = np.zeros(shape, dtype=bool)
    src = cams_xz[cam_idx]
    vec = pts_xz - src
    dist = np.linalg.norm(vec, axis=1)
    keep = dist > 3 * cell
    src, vec, dist = src[keep], vec[keep], dist[keep]
    stop = (dist - 2.0 * cell) / dist  # 点の手前 2 マスで止める（壁自体は削らない）
    for t in np.linspace(0.0, 1.0, int(np.ceil(dist.max() / (cell * 0.7))) + 1):
        tt = np.minimum(t, stop)
        p = src + vec * tt[:, None]
        ij = np.floor((p - origin) / cell).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
        free[ij[ok, 0], ij[ok, 1]] = True
    return free


def _trace_outer_boundary(mask: np.ndarray) -> list[tuple[int, int]]:
    """True の連結領域（穴なし前提）の外周を、マス目の角の座標 (i,j) の閉路として返す。
    向き: (i=x, j=z) 平面で反時計回り（内部が進行方向の左）。"""
    nxt: dict[tuple[int, int], list[tuple[int, int]]] = {}
    I, J = mask.shape
    m = np.pad(mask, 1)
    ii, jj = np.nonzero(mask)
    for i, j in zip(ii.tolist(), jj.tolist()):
        pi, pj = i + 1, j + 1
        if not m[pi, pj - 1]:
            nxt.setdefault((i, j), []).append((i + 1, j))        # 下辺: +x 方向
        if not m[pi + 1, pj]:
            nxt.setdefault((i + 1, j), []).append((i + 1, j + 1))  # 右辺: +z
        if not m[pi, pj + 1]:
            nxt.setdefault((i + 1, j + 1), []).append((i, j + 1))  # 上辺: -x
        if not m[pi - 1, pj]:
            nxt.setdefault((i, j + 1), []).append((i, j))          # 左辺: -z
    start = min(nxt.keys())
    loop = [start]
    prev_dir = None
    cur = start
    while True:
        cands = nxt[cur]
        if len(cands) == 1 or prev_dir is None:
            choice = cands[0]
        else:
            # 角で接する場合は左折を優先（内部を左に保つ）
            left = (-prev_dir[1], prev_dir[0])
            choice = next((c for c in cands if (c[0] - cur[0], c[1] - cur[1]) == left), cands[0])
        cands.remove(choice)
        prev_dir = (choice[0] - cur[0], choice[1] - cur[1])
        cur = choice
        if cur == start:
            break
        loop.append(cur)
    # 同一直線上の頂点を除く
    out = []
    n = len(loop)
    for k in range(n):
        a, b, c = loop[k - 1], loop[k], loop[(k + 1) % n]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            out.append(b)
    return out


def _cluster_1d(values: np.ndarray, weights: np.ndarray, tol: float) -> np.ndarray:
    """値を tol 以内でまとめ、各値をクラスタの重み付き平均へ置き換える。"""
    order = np.argsort(values)
    out = values.astype(float).copy()
    start = 0
    for k in range(1, len(order) + 1):
        if k == len(order) or values[order[k]] - values[order[k - 1]] > tol:
            idx = order[start:k]
            w = weights[idx]
            out[idx] = float((values[idx] * w).sum() / max(w.sum(), 1e-9))
            start = k
    return out


def simplify_rectilinear(verts: np.ndarray, tol: float, min_edge: float) -> np.ndarray:
    """直交多角形の小さな段差（tol 未満）をまとめる。tol 以上の段差は残す。"""
    v = verts.astype(float).copy()
    for _ in range(6):
        n = len(v)
        if n < 4:
            break
        nxt = np.roll(v, -1, axis=0)
        horiz = np.abs(nxt[:, 1] - v[:, 1]) < 1e-9      # z 一定の辺
        length = np.abs(nxt - v).sum(axis=1)
        # z 一定の辺の z、x 一定の辺の x をそれぞれクラスタリング
        newv = v.copy()
        for is_h, axis in ((True, 1), (False, 0)):
            sel = np.flatnonzero(horiz == is_h)
            if len(sel) == 0:
                continue
            snapped = _cluster_1d(v[sel, axis], length[sel] + 1e-6, tol)
            for e, val in zip(sel, snapped):
                newv[e, axis] = val
                newv[(e + 1) % n, axis] = val
        v = newv
        # 長さ 0 の辺・同一直線の頂点を削除
        keep = []
        n = len(v)
        for k in range(n):
            a, b, c = v[k - 1], v[k], v[(k + 1) % n]
            if np.abs(b - a).sum() < 1e-6:
                continue
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if abs(cross) < 1e-9 and np.abs(c - b).sum() > 1e-6:
                continue
            keep.append(k)
        if len(keep) == n:
            break
        v = v[keep]
    return v


# ---------------------------------------------------------------- 本体

def build_room_structure(points_world: np.ndarray, cam2world: np.ndarray, point_cam_index: np.ndarray,
                         params: StructureParams | None = None, up_hint: np.ndarray | None = None) -> StructureResult:
    """up_hint: 上方向が分かっている場合（ARCore の世界座標は +Y が重力の反対）に渡す。
    スマホを縦に持つとセンサー画像は横倒しになるため、その場合カメラの向きから上方向は推定できない。"""
    p = params or StructureParams()
    if len(points_world) < 5000:
        raise StructureError("too_few_points", "復元できた点が少なすぎます。写真の枚数と重なりを増やしてください。")

    up0 = estimate_up_from_cameras(cam2world) if up_hint is None else np.asarray(up_hint, float) / np.linalg.norm(up_hint)
    n, d, ceiling, ceiling_obs, diag = find_floor_and_ceiling(points_world, up0)
    R = rotation_up_to_y(n)
    pts = points_world @ R.T
    pts[:, 1] -= d
    cams = cam2world[:, :3, 3] @ R.T
    cams[:, 1] -= d

    top = ceiling if ceiling is not None else float(np.percentile(pts[:, 1], 99))
    if top < 1.2:
        raise StructureError("too_low", "床から上の範囲がほとんど復元できていません。壁が写る写真を追加してください。")
    band = (pts[:, 1] > 0.15) & (pts[:, 1] < top - 0.10)

    # 壁どうしの角のずれを直す（床平面内のせん断。高さは変えない）
    A2, rect_info = wall_corner_shear(pts[band][:, [0, 2]])
    A3 = np.eye(3)
    A3[np.ix_([0, 2], [0, 2])] = A2
    pts = pts @ A3.T
    cams = cams @ A3.T

    # 壁の主方向を軸へ合わせる（床平面内の回転のみ）
    yaw = dominant_yaw(pts[band][:, [0, 2]])
    c, s = np.cos(yaw), np.sin(yaw)
    Ryaw = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    assert np.linalg.det(Ryaw) > 0.999
    pts = pts @ Ryaw.T
    cams = cams @ Ryaw.T
    Rfull = Ryaw @ A3 @ R  # 回転と、必要なら小さなせん断（鏡映は含まない）

    xz = pts[:, [0, 2]]
    # 離れて撮ると足もとの床は写らないので、カメラも範囲に入れる（入れないと、カメラのまわりの空間を探せない）
    lo = np.minimum(np.percentile(xz, 0.2, axis=0), cams[:, [0, 2]].min(axis=0)) - 0.3
    hi = np.maximum(np.percentile(xz, 99.8, axis=0), cams[:, [0, 2]].max(axis=0)) + 0.3
    shape = tuple(np.ceil((hi - lo) / p.cell).astype(int))
    if shape[0] * shape[1] > 4_000_000:
        raise StructureError("too_large", "復元範囲が広すぎます。1部屋ずつ撮影してください。")
    ij = np.floor((xz - lo) / p.cell).astype(np.int64)
    inb = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])

    # 縦の構造物: 高さ方向に広く点があるマス
    nb = max(3, int((top - 0.25) / p.height_bin))
    hb = np.floor((pts[:, 1] - 0.15) / p.height_bin).astype(np.int64)
    sel = inb & band & (hb >= 0) & (hb < nb)
    occ = np.zeros((shape[0], shape[1], nb), dtype=bool)
    occ[ij[sel, 0], ij[sel, 1], hb[sel]] = True
    occ2 = ndimage.binary_dilation(occ, structure=np.ones((3, 3, 1), bool))  # 面内の点の疎さを補う
    coverage = occ2.sum(axis=2) / nb
    wall = coverage >= p.wall_coverage
    wall = ndimage.binary_closing(wall, structure=np.ones((3, 3), bool))

    # 見通せた範囲
    rng = np.random.default_rng(0)
    cand = np.flatnonzero(inb)
    if len(cand) > p.max_rays:
        cand = rng.choice(cand, p.max_rays, replace=False)
    free = _carve_free(cams[:, [0, 2]], xz[cand], point_cam_index[cand], lo, p.cell, shape)
    free = ndimage.binary_closing(free, structure=np.ones((5, 5), bool))
    interior = free & ~wall
    lab, _ = ndimage.label(interior)
    cam_ij = np.floor((cams[:, [0, 2]] - lo) / p.cell).astype(int)
    cam_ij = cam_ij[(cam_ij[:, 0] >= 0) & (cam_ij[:, 0] < shape[0]) & (cam_ij[:, 1] >= 0) & (cam_ij[:, 1] < shape[1])]
    labs = lab[cam_ij[:, 0], cam_ij[:, 1]]
    labs = labs[labs > 0]
    if len(labs) == 0:
        raise StructureError("no_interior", "撮影位置のまわりに部屋の空間を見つけられませんでした。部屋の中から壁を撮影してください。")
    main = np.bincount(labs).argmax()
    interior = lab == main
    filled = ndimage.binary_fill_holes(interior)

    # 部屋の中に独立して立つ縦の構造物（独立柱など）
    obstacles = []
    holes = filled & ~interior & wall
    hl, hn = ndimage.label(holes)
    for k in range(1, hn + 1):
        ii, jj = np.nonzero(hl == k)
        if len(ii) < 9:
            continue
        x0, z0 = lo + np.array([ii.min(), jj.min()]) * p.cell
        x1, z1 = lo + np.array([ii.max() + 1, jj.max() + 1]) * p.cell
        colmax = occ[ii, jj].any(axis=0)
        ytop = 0.15 + (np.flatnonzero(colmax).max() + 1) * p.height_bin if colmax.any() else top
        obstacles.append({
            "kind": "unknown_vertical",
            "rect": [float(x0), float(z0), float(x1), float(z1)],
            "yMin": 0.0, "yMax": float(min(top, ytop)),
            "status": "auto",
        })

    loop = np.array(_trace_outer_boundary(filled), dtype=float)
    verts = lo + loop * p.cell
    verts = simplify_rectilinear(verts, p.snap_tol, p.min_edge)
    if len(verts) < 4:
        raise StructureError("no_outline", "部屋の輪郭を作れませんでした。写真を追加して再度お試しください。")

    # 各辺: 近くの壁点で位置を合わせ、壁として観測できたかを決める
    wall_pts = xz[band]
    nV = len(verts)
    status, support = [], []
    refined = verts.copy()
    area2 = float(np.sum(verts[:, 0] * np.roll(verts[:, 1], -1) - np.roll(verts[:, 0], -1) * verts[:, 1]))
    ccw = area2 > 0
    for k in range(nV):
        a, b = verts[k], verts[(k + 1) % nV]
        along = 0 if abs(b[1] - a[1]) < 1e-9 else 1   # 辺が伸びる軸
        norm_ax = 1 - along
        t0, t1 = sorted((a[along], b[along]))
        cpos = a[norm_ax]
        # 外向き法線の符号（反時計回りなら進行方向の右が外）
        dirv = np.sign(b[along] - a[along])
        if along == 0:
            out_sign = -dirv if ccw else dirv
        else:
            out_sign = dirv if ccw else -dirv
        rel = (wall_pts[:, norm_ax] - cpos) * out_sign
        m = (wall_pts[:, along] > t0 + 0.02) & (wall_pts[:, along] < t1 - 0.02) & (rel > -0.06) & (rel < 0.12)
        cnt = int(m.sum())
        length = max(t1 - t0, 1e-6)
        # 辺に沿って点がどれだけ途切れず並んでいるか
        if cnt:
            bins = max(1, int(length / 0.05))
            filled_ratio = len(np.unique(np.floor((wall_pts[m, along] - t0) / length * bins))) / bins
        else:
            filled_ratio = 0.0
        observed = cnt >= 30 and filled_ratio >= 0.6
        status.append("observed" if observed else "unobserved")
        support.append(cnt)
        if observed:
            val = float(np.median(wall_pts[m, norm_ax]))
            refined[k, norm_ax] = val
            refined[(k + 1) % nV, norm_ax] = val

    # 位置合わせで向きが崩れていないことを確認（崩れたら合わせる前を使う）
    def _rectilinear(v):
        nx = np.roll(v, -1, axis=0)
        return bool(np.all((np.abs(nx[:, 0] - v[:, 0]) < 1e-6) | (np.abs(nx[:, 1] - v[:, 1]) < 1e-6)))
    if _rectilinear(refined):
        verts = refined
    if not ccw:
        verts = verts[::-1]
        status = status[::-1]
        status = status[1:] + status[:1]
        support = support[::-1]
        support = support[1:] + support[:1]

    T = np.eye(4)
    T[:3, :3] = Rfull
    T[:3, 3] = Ryaw @ np.array([0.0, -d, 0.0])
    diag.update(rect_info)
    diag.update({
        "grid_shape": [int(shape[0]), int(shape[1])],
        "cell_m": p.cell,
        "yaw_deg": float(np.degrees(yaw)),
        "n_points": int(len(points_world)),
        "observed_edge_ratio": float(np.mean([s == "observed" for s in status])) if status else 0.0,
        "snap_tol_m": p.snap_tol,
    })
    return StructureResult(
        T_room_from_world=T,
        outline=[[float(x), float(z)] for x, z in verts],
        edge_status=status,
        edge_support=support,
        ceiling_height=ceiling,
        ceiling_observed=ceiling_obs,
        obstacles=obstacles,
        diagnostics=diag,
    )
