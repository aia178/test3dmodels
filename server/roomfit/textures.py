"""壁と床に、元の写真を投影して貼った画像を作る（見た目のため。寸法には使わない）。

点の集まりの表示は、AI が写真を 392×518 画素に縮めて計算するため、ぼやけてのっぺり見える。部屋の大部分は平面なので、
壁と床の位置だけ復元から取り、色は元の写真（数千画素）から取る。壁は平面として扱うので、扉の膨らみも出ない。
写真の画素と 3D の対応は、モデル自身のカメラ（位置・内部パラメータ）で投影するので、復元の点と食い違わない。

壁の手前にある物（服・家具など）が壁の模様として貼られないよう、写真ごとに「その画素でモデルが見ている物が、
壁と同じ距離にあるか」を確かめ、手前に物がある写真や、位置のずれで壁より奥を見ている写真は、その場所に使わない。
各面では「広く・正面から写している」1 枚を主に使い、写っていない所だけ他の写真で埋め、重なる所で明るさを合わせる。

スイッチ・コンセントのような小さな物の上を写真の境目が通ると、細く見えたり途中で切れたりする（2026-10-16、自宅の角）。
そこで「保護領域」（壁の上の長方形）を指定すると、その領域全体を手前の物なし・端に余裕ありで写した 1 枚だけで塗り、
写真の境目は保護領域の外の狭い帯に置く。全体を写した 1 枚がなければ、その領域は普段どおり（無理につながない）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_SIDE = 2048          # 1 面の画像の長い辺の上限（表示の負担を抑える）
MIN_RES = 0.002          # 1 画素の大きさの下限 [m]
OCCLUDE_M = 0.06         # 写真の中で、壁より手前にこれ以上近い物が見えていたら「手前に物がある」とする
MIN_FRONTAL = 0.35       # 面を斜め 70 度より浅く見ている写真は使わない
GRAY = 228
PROTECT_PAD_M = 0.01     # 保護領域を、指定より少し広げる（指定の枠の誤差のぶん）
SEAM_BAND_M = 0.03       # 保護領域の外側で、写真の境目をなじませる帯の幅
ALIGN_RING_M = 0.06      # 保護領域を塗る写真を周りに合わせるとき、位置・明るさを比べる輪の幅
PROTECT_EDGE = 0.03      # 保護領域を塗る写真は、元の写真の端から（長い辺の）この割合以上離れて写していること
OBJECT_SEARCH_M = 0.08   # 写真ごとの小物の位置のずれを、この範囲で探す
CONSENSUS_TOL_M = 0.015  # 小物の位置が、写真の多数派からこれ以内の写真を「位置が合っている」とする


@dataclass
class ViewData:
    """写真 1 枚ぶん。座標はすべて部屋座標（y が床からの高さ、m）。"""
    cam2room: np.ndarray        # 4×4、OpenCV の向き（+z が前）
    K: np.ndarray               # 3×3、モデルの画素
    grid_room: np.ndarray       # H×W×3、モデルが各画素で見ている点（無い所は NaN）
    photo: np.ndarray           # 元の写真（向きは見たまま）H0×W0×3 uint8
    to_photo: tuple[float, float, float, float]   # (s, ox, oy, k): 元の写真の画素 = ((モデルの画素 + o) / s) * k


@dataclass
class _OnPlane:
    """写真 1 枚を平面に写した結果。配列は平面の点ごと（画像の画素を並べたもの）。"""
    view: int                   # views の何番目か
    ok: np.ndarray              # その写真から色を取ってよいか
    frontal: np.ndarray         # 正面から見ている度合い（1 が真正面）
    edge: np.ndarray            # 元の写真の端からの距離（長い辺に対する割合）
    xb: np.ndarray              # 元の写真の画素（横）
    yb: np.ndarray              # 元の写真の画素（縦）
    photo: np.ndarray

    def texture(self, shape: tuple[int, int]) -> np.ndarray:
        H0, W0 = self.photo.shape[:2]
        return self.photo[np.clip(self.yb.astype(int), 0, H0 - 1), np.clip(self.xb.astype(int), 0, W0 - 1)].reshape(*shape, 3).astype(np.float64)


def _project(origin: np.ndarray, e1: np.ndarray, e2: np.ndarray, normal: np.ndarray, plane_pts: np.ndarray, vd: ViewData, idx: int) -> _OnPlane | None:
    """平面の点（plane_pts、平面上の座標）を、写真 vd のどの画素で見ているかを求める。平面がほとんど写っていなければ None。

    平面の点を写真へ写す変換は、カメラの位置から計算せず、その写真自身が「各画素で見ている 3D の点」のうち平面の上にある点と、
    その画素の対応から、射影変換（平面→画像）として直接求める。AI の推定したカメラの位置や奥行きの大きさは写真ごとに少しずれており、
    カメラから投影すると、写真ごとに貼る位置が数 cm〜十数 cm ずれた（2026-10-16）。点の位置は写真どうしで 1cm 程度にそろっている。
    """
    import cv2

    hm, wm = vd.grid_room.shape[:2]
    G = vd.grid_room.reshape(-1, 3)
    fin = np.isfinite(G).all(1)
    d = (G - origin) @ normal
    on = fin & (np.abs(d) < 0.03)
    if on.sum() < 300:
        return None
    vv_, uu_ = np.divmod(np.flatnonzero(on), wm)
    src = np.stack([(G[on] - origin) @ e1, (G[on] - origin) @ e2], -1)
    dst = np.stack([uu_ + 0.5, vv_ + 0.5], -1).astype(np.float64)
    if len(src) > 20000:
        pick = np.random.default_rng(0).choice(len(src), 20000, replace=False)
        src, dst = src[pick], dst[pick]
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if H is None or inl.sum() < 200:
        return None
    q = np.c_[plane_pts, np.ones(len(plane_pts))] @ H.T
    um, vm = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
    ok = (q[:, 2] > 0) & (um >= 0) & (um < wm - 1) & (vm >= 0) & (vm < hm - 1)
    # その画素で写真が見ている点が、平面の上にあるか（手前に物がある、別の物を見ている所は使わない）
    ui, vi = np.clip(um.astype(int), 0, wm - 1), np.clip(vm.astype(int), 0, hm - 1)
    Q = vd.grid_room[vi, ui]
    dq = np.abs((Q - origin) @ normal)
    ok &= ~(np.isfinite(dq) & (dq > OCCLUDE_M))
    # 正面から見ているか（並べる順と、浅すぎる角度の除外に使う）
    cam = vd.cam2room[:3, 3]
    P = origin + plane_pts[:, :1] * e1 + plane_pts[:, 1:] * e2
    ray = P - cam; ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-9)
    frontal = np.abs(ray @ normal)
    ok &= frontal > MIN_FRONTAL
    s, ox, oy, k = vd.to_photo
    xb, yb = (um + ox) / s * k, (vm + oy) / s * k
    H0, W0 = vd.photo.shape[:2]
    ok &= (xb >= 0) & (xb < W0 - 1) & (yb >= 0) & (yb < H0 - 1)
    edge = np.minimum(np.minimum(xb, W0 - 1 - xb), np.minimum(yb, H0 - 1 - yb)) / max(H0, W0)
    return _OnPlane(idx, ok, frontal, edge, xb, yb, vd.photo)


def _render_plane(origin: np.ndarray, e1: np.ndarray, e2: np.ndarray, S1: np.ndarray, S2: np.ndarray, views: list[ViewData],
                  shape: tuple[int, int], res_m: float = MIN_RES, protect: list | None = None, seam_band: bool = True) -> dict:
    """平面 origin + s1·e1 + s2·e2 の上の点（S1, S2）の色を、写真から取る。

    protect: 保護領域（この平面の画像の画素で (左, 上, 右, 下)）の並び。seam_band=False は比べるため（境目を保護領域の縁にそのまま置く）。
    戻り値: {"image": H×W×3, "done": 色が取れたか H×W, "src": 各画素を塗った写真の番号（views の番号、無ければ -1）H×W,
             "protected": [{"box", "view"（塗った写真の番号、None なら全体を写した写真がなかった）}]}
    """
    import cv2

    normal = np.cross(e1, e2); normal /= np.linalg.norm(normal)
    plane_pts = np.stack([S1, S2], -1).astype(np.float64)
    per = [p for i, vd in enumerate(views) if (p := _project(origin, e1, e2, normal, plane_pts, vd, i)) is not None]
    # 正面から広く見ている写真を優先する
    per.sort(key=lambda p: -float((np.clip(p.frontal, 0, 1) ** 4 * p.ok).sum()))
    h, w = shape
    img = np.zeros((h, w, 3)); done = np.zeros((h, w), bool); src = np.full((h, w), -1, int)
    for p in per:
        if not p.ok.any():
            continue
        tex = p.texture(shape)
        m = p.ok.reshape(h, w)
        if done.any() and (done & m).sum() > 2000:
            # 先に貼った写真の模様（スイッチ・扉の縁など）に合わせて、少しだけ動かす。写真ごとに 1〜2cm ずれていて、
            # そのまま貼ると、写真の境目にある小さな物が切れたり二重になったりした（2026-10-16、壁のスイッチ）
            Hc = _align_to(tex, m, img, done, max_shift_px=max(4.0, 0.05 / max(res_m, 1e-6)))
            if Hc is not None:
                tex = cv2.warpPerspective(tex, Hc, (w, h), flags=cv2.INTER_LINEAR)
                m = cv2.warpPerspective(m.astype(np.uint8), Hc, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
            both = done & m
            if both.sum() > 500:   # 重なる所で、明るさを合わせる
                tex *= np.median(img[both], axis=0) / np.maximum(np.median(tex[both], axis=0), 1.0)
        take = m & ~done
        img[take] = tex[take]; done |= take; src[take] = p.view
    prot = [_paint_protected(box, per, img, done, src, res_m, seam_band) for box in (protect or [])]
    im = np.clip(img, 0, 255).astype(np.uint8)
    im[~done] = GRAY
    return {"image": im, "done": done, "src": src, "protected": prot}


def _paint_protected(box, per: list[_OnPlane], img: np.ndarray, done: np.ndarray, src: np.ndarray, res_m: float, seam_band: bool) -> dict:
    """保護領域 box を、その全体（＋なじませる帯）を写した 1 枚だけで塗り直す（img, done, src を書き換える）。

    写真ごとに、壁に写したときの小物の位置が数 cm 食い違う（自宅の角のコンセントでは、写真の端で写した 1 枚だけが約 4.5cm 左）。
    そこで、まず小物そのものの模様で写真ごとの位置のずれを測り、多数派の位置に合う写真のうち、手前の物なし・端に余裕あり・
    いちばん正面から見ている 1 枚を選ぶ。保護領域は、周りを塗った写真と選んだ写真での小物の位置がすべて入るように広げる（二重・欠けを防ぐ）。
    塗る前に周りの輪で位置（平行移動）と明るさを合わせ、帯の中で周りと少しずつ混ぜる。境目は保護領域の外に来る。"""
    import cv2

    h, w = done.shape
    pad = max(1, round(PROTECT_PAD_M / res_m))
    band = max(1, round(SEAM_BAND_M / res_m)) if seam_band else 0
    ring = max(2, round(ALIGN_RING_M / res_m))
    srch = max(2, round(OBJECT_SEARCH_M / res_m))
    info = {"box": [int(round(v)) for v in box], "view": None}
    bx0, by0, bx1, by1 = max(0, info["box"][0]), max(0, info["box"][1]), min(w, info["box"][2]), min(h, info["box"][3])
    if bx1 <= bx0 or by1 <= by0:
        return info
    # 計算は保護領域の周りの窓だけで行う（窓の中の座標で扱う）
    m_out = srch + pad + band + ring
    wx0, wy0, wx1, wy1 = max(0, bx0 - m_out), max(0, by0 - m_out), min(w, bx1 + m_out), min(h, by1 + m_out)
    hh, ww = wy1 - wy0, wx1 - wx0
    yy, xx = np.mgrid[wy0:wy1, wx0:wx1]
    flat = (yy * w + xx).ravel()

    def win(a):
        return a[flat].reshape(hh, ww)

    def tex_of(p):
        H0, W0 = p.photo.shape[:2]
        return p.photo[np.clip(win(p.yb).astype(int), 0, H0 - 1), np.clip(win(p.xb).astype(int), 0, W0 - 1)].astype(np.float64)

    def gray(x):
        return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    def sees(p, r):
        x0, y0, x1, y1 = r
        return bool(win(p.ok)[y0:y1, x0:x1].all()) and float(win(p.edge)[y0:y1, x0:x1].min()) >= PROTECT_EDGE

    lb = (bx0 - wx0, by0 - wy0, bx1 - wx0, by1 - wy0)
    inbox = src[by0:by1, bx0:bx1]
    if (inbox >= 0).all() and (inbox == inbox.flat[0]).all():
        info.update(view=int(inbox.flat[0]), kept=True)   # 境目が通っていない: そのまま
        return info
    full = [p for p in per if sees(p, lb)]
    if not full:
        return info                           # 全体を写した写真がない: 普段どおり（無理につながない）
    front = {p.view: float(win(p.frontal)[lb[1]:lb[3], lb[0]:lb[2]].mean()) for p in per}
    ref = max(full, key=lambda p: front[p.view])
    # 写真ごとの小物の位置（ref に対するずれ、画素）を、小物を含む箱の模様の一致で測る
    tmpl = gray(tex_of(ref))[lb[1]:lb[3], lb[0]:lb[2]]
    shifts = {}
    for p in per:
        sx0, sy0, sx1, sy1 = max(0, lb[0] - srch), max(0, lb[1] - srch), min(ww, lb[2] + srch), min(hh, lb[3] + srch)
        okw = win(p.ok)[sy0:sy1, sx0:sx1]
        if okw.mean() < 0.5 or sx1 - sx0 <= tmpl.shape[1] or sy1 - sy0 <= tmpl.shape[0]:
            continue
        g = gray(tex_of(p))[sy0:sy1, sx0:sx1]
        g[~okw] = np.median(g[okw])
        _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(g, tmpl, cv2.TM_CCOEFF_NORMED))
        if score >= 0.5:
            shifts[p.view] = (loc[0] + sx0 - lb[0], loc[1] + sy0 - lb[1])
    cand = [p for p in full if p.view in shifts]
    if not cand:
        return info
    cx, cy = (float(np.median([shifts[p.view][k] for p in cand])) for k in (0, 1))
    info["shift_cm"] = {int(v): [round((d[0] - cx) * res_m * 100, 1), round((d[1] - cy) * res_m * 100, 1)] for v, d in shifts.items()}
    # 目標の位置: 周り（箱の外）を塗った写真での小物の位置（画素数の重み付き平均）。見た目のための処理なので、
    # どれが正しい位置かより、周り（ドア枠など）とのつながりを優先する。周りの写真の位置が測れなければ多数派の位置
    ring_src = src[wy0:wy1, wx0:wx1].copy(); ring_src[lb[1]:lb[3], lb[0]:lb[2]] = -1
    cnt = np.bincount(ring_src[ring_src >= 0].ravel(), minlength=max(shifts) + 1)
    wsum = sum(cnt[v] for v in shifts if v < len(cnt))
    if wsum > 0:
        tx = sum(cnt[v] * shifts[v][0] for v in shifts if v < len(cnt)) / wsum
        ty = sum(cnt[v] * shifts[v][1] for v in shifts if v < len(cnt)) / wsum
    else:
        tx, ty = cx, cy
    off = {v: float(np.hypot(d[0] - tx, d[1] - ty)) for v, d in shifts.items()}
    tol = CONSENSUS_TOL_M / res_m
    # 目標の位置に合う写真を正面の順に、合わない写真はずれの小さい順に試す
    cand.sort(key=lambda p: (off[p.view] > tol, -front[p.view] if off[p.view] <= tol else off[p.view]))
    # 箱は、いまの壁の画像（その場所を塗った写真での位置）の上で指定されたものとして、各写真での小物の位置へ直す
    owner = int(np.bincount(inbox[inbox >= 0]).argmax()) if (inbox >= 0).any() else ref.view
    o = shifts.get(owner, (cx, cy))
    info["box_view"] = owner
    # いま見えている小物の写し（その写真が、その写真での小物の位置を塗っている所）を、すべて領域に入れる
    shown = []
    for v in set(np.unique(src[wy0:wy1, wx0:wx1]).tolist()) - {-1}:
        if v not in shifts:
            continue
        dx, dy = shifts[v][0] - o[0], shifts[v][1] - o[1]
        r = (max(0, lb[0] + dx), max(0, lb[1] + dy), min(ww, lb[2] + dx), min(hh, lb[3] + dy))
        if r[2] > r[0] and r[3] > r[1] and (src[wy0:wy1, wx0:wx1][r[1]:r[3], r[0]:r[2]] == v).mean() > 0.05:
            shown.append((dx, dy))
    for best in cand:
        moves = shown + [(shifts[best.view][0] - o[0], shifts[best.view][1] - o[1])]
        x0 = max(0, lb[0] + min(d[0] for d in moves) - pad); y0 = max(0, lb[1] + min(d[1] for d in moves) - pad)
        x1 = min(ww, lb[2] + max(d[0] for d in moves) + pad); y1 = min(hh, lb[3] + max(d[1] for d in moves) + pad)
        if sees(best, (max(0, x0 - band), max(0, y0 - band), min(ww, x1 + band), min(hh, y1 + band))):
            break
    else:
        return info
    info["view"] = best.view
    info["region"] = [x0 + wx0, y0 + wy0, x1 + wx0, y1 + wy0]
    yy, xx = np.mgrid[0:hh, 0:ww]
    # 保護領域からの距離（画素、領域の中は 0）
    dist = np.hypot(np.maximum(np.maximum(x0 - xx, xx - (x1 - 1)), 0), np.maximum(np.maximum(y0 - yy, yy - (y1 - 1)), 0))
    inner = dist <= band                      # 選んだ 1 枚で塗る所（領域＋帯）
    sl = (slice(wy0, wy1), slice(wx0, wx1))
    tex, ok = tex_of(best), win(best.ok)
    base, base_m = img[sl], done[sl]
    outer = (dist > band) & ok & base_m       # 周りの輪（位置・明るさを比べる所）
    if seam_band and outer.sum() > 200:
        Hc = _align_to(tex, ok & (dist > band), base, base_m & (dist > band), max_shift_px=max(4.0, 0.03 / res_m), min_overlap=200)
        if Hc is not None:
            tex = cv2.warpPerspective(tex, Hc, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ok = cv2.warpPerspective(ok.astype(np.uint8), Hc, (ww, hh), flags=cv2.INTER_NEAREST).astype(bool)
    # 明るさを周りに合わせる。比べるのは、どちらも色の薄い所（白い壁）だけ。ドア枠の木が写真ごとに数 cm ずれて入ると、
    # 色の比が大きく狂った（コンセントの周りが茶色になった）
    plain = (np.ptp(base, axis=2) < 40) & (np.ptp(tex, axis=2) < 40)
    cmp = outer & ok & plain if (outer & ok & plain).sum() > 200 else (base_m & ok & plain & (dist > 0))
    if cmp.sum() > 50:
        tex = tex * (np.median(base[cmp], axis=0) / np.maximum(np.median(tex[cmp], axis=0), 1.0))
    # 帯の中で、保護領域の縁（選んだ 1 枚だけ）→ 帯の外側（周り）へ少しずつ混ぜる。周りが無い所は選んだ 1 枚のまま
    alpha = np.clip(1.0 - dist / max(band, 1), 0, 1) if band else (dist == 0).astype(float)
    alpha = np.where(inner & ~base_m, 1.0, alpha) * (ok | (dist == 0))
    base[:] = base * (1 - alpha[..., None]) + tex * alpha[..., None]
    src[sl][alpha >= 0.5] = best.view
    done[sl] |= alpha > 0
    return info


def _align_to(tex: np.ndarray, m: np.ndarray, base: np.ndarray, base_m: np.ndarray, max_shift_px: float, min_overlap: int = 2000) -> np.ndarray | None:
    """tex を base に重ねる小さな変換（tex の画素 → base の画素、3×3）を、重なる部分の濃淡全体から求める（ECC 法、粗い→細かい）。
    特徴点の対応は、木目や白い壁では取れないか誤対応が多かった。合わない・動かす量が大きすぎるときは None（動かさない）。"""
    import cv2

    def gray(x):
        return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    ov = (m & base_m).astype(np.uint8)
    if ov.sum() < min_overlap:
        return None
    g_tex, g_base = gray(tex), gray(base)
    W = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        for f in (0.25, 0.5, 1.0):
            sz = (max(16, int(g_base.shape[1] * f)), max(16, int(g_base.shape[0] * f)))
            tb, ti = cv2.resize(g_base, sz, interpolation=cv2.INTER_AREA), cv2.resize(g_tex, sz, interpolation=cv2.INTER_AREA)
            mk = cv2.resize(ov, sz, interpolation=cv2.INTER_NEAREST)
            Ws = W.copy(); Ws[:, 2] *= f
            _, Ws = cv2.findTransformECC(tb, ti, Ws, cv2.MOTION_TRANSLATION, crit, mk, 5)
            W = Ws.copy(); W[:, 2] /= f
    except cv2.error:
        return None
    # ECC は「base の画素 x に、tex の W·x を持ってくる」変換。tex → base の向きにして返す
    W3 = np.vstack([W, [0, 0, 1]]).astype(np.float64)
    H = np.linalg.inv(W3)
    h, w = m.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    moved = np.linalg.norm(cv2.perspectiveTransform(corners, H) - corners, axis=2).max()
    return H if moved <= max_shift_px * 2 else None


def make_surfaces(outline: list, edge_status: list, ceiling: float | None, views: list[ViewData],
                  protect: dict | None = None, seam_band: bool = True, details: bool = False) -> dict:
    """観測できた壁と床の画像を作る。戻り値: {"walls": [{"edge": i, "image": 配列, "coverage": 割合}], "floor": {...} | None}。
    壁の画像は、辺の始点→終点を左→右、天井（ceiling、無ければ 2.4m）→床を上→下に並べたもの。
    床の画像は、輪郭を囲む長方形を、x を左→右、z を上→下に並べたもの。

    protect: 壁ごとの保護領域 {辺の番号: [(辺の始点からの距離 左, 床からの高さ 上, 右, 下), ...]}（m）。スイッチ・コンセントなど。
    details=True で、各面に "src"（各画素を塗った写真の番号）と "res"（1 画素の大きさ m）を付ける（実験用）。"""
    O = np.asarray(outline, float)
    hgt = ceiling if ceiling else 2.4
    out = {"walls": [], "floor": None}
    for e, st in enumerate(edge_status):
        if st != "observed":
            continue
        a, b = O[e], O[(e + 1) % len(O)]
        L = float(np.linalg.norm(b - a))
        if L < 0.2:
            continue
        res = max(MIN_RES, max(L, hgt) / MAX_SIDE)
        nu, nv = max(2, int(L / res)), max(2, int(hgt / res))
        t = (b - a) / L
        uu, vv = np.meshgrid((np.arange(nu) + 0.5) * L / nu, hgt - (np.arange(nv) + 0.5) * hgt / nv)
        # 保護領域を、この壁の画像の画素へ（左→右が辺の向き、上→下が天井→床）
        boxes = [(u0 / L * nu, (hgt - h0) / hgt * nv, u1 / L * nu, (hgt - h1) / hgt * nv) for u0, h0, u1, h1 in (protect or {}).get(e, [])]
        r = _render_plane(np.array([a[0], 0.0, a[1]]), np.array([t[0], 0.0, t[1]]), np.array([0.0, 1.0, 0.0]),
                          uu.ravel(), vv.ravel(), views, (nv, nu), res, protect=boxes, seam_band=seam_band)
        wall = {"edge": e, "image": r["image"], "coverage": round(float(r["done"].mean()), 3)}
        if r["protected"]:
            wall["protected"] = r["protected"]
        if details:
            wall.update(src=r["src"], res=res)
        out["walls"].append(wall)
    x0, z0 = O.min(0); x1, z1 = O.max(0)
    W, D = float(x1 - x0), float(z1 - z0)
    res = max(MIN_RES, max(W, D) / MAX_SIDE)
    nu, nv = max(2, int(W / res)), max(2, int(D / res))
    uu, vv = np.meshgrid(x0 + (np.arange(nu) + 0.5) * W / nu, z0 + (np.arange(nv) + 0.5) * D / nv)
    r = _render_plane(np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), uu.ravel(), vv.ravel(), views, (nv, nu), res)
    out["floor"] = {"image": r["image"], "coverage": round(float(r["done"].mean()), 3), "bbox": [float(x0), float(z0), float(x1), float(z1)]}
    if details:
        out["floor"].update(src=r["src"], res=res)
    return out


def photo_mapping(model_hw: tuple[int, int], pre_wh: tuple[int, int], photo_wh: tuple[int, int]) -> tuple[float, float, float, float]:
    """モデルの画素 → 元の写真の画素 の対応。前処理（長い辺 1024 へ縮小）→ モデル（縦横比を保って縮小し、はみ出しを中央で切る）。"""
    hm, wm = model_hw
    Ws, Hs = pre_wh
    s = max(wm / Ws, hm / Hs)
    return s, (Ws * s - wm) / 2, (Hs * s - hm) / 2, photo_wh[0] / Ws
