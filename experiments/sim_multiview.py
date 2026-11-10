"""実験: 1 枚の写真でタップした 2 点を、ほかの写真でも自動で探して測り、中央値をとると精度が上がるか。
「同じ場所」は、タップした点の 3D 位置に最も近い 3D 位置を持つ画素として、ほかの写真から探す。
使い方（server フォルダで）: uv run python ../experiments/sim_multiview.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
src = (Path(__file__).resolve().parent / "sim_tap_snap.py").read_text(encoding="utf-8")
ns = {"__file__": str(Path(__file__).resolve().parent / "sim_tap_snap.py")}
exec(compile(src[: src.index("# 写真 12（引き戸の正面）")], "sim_tap_snap", "exec"), ns)
load, lookup, snap, OUT, INVALID = ns["load"], ns["lookup"], ns["snap"], ns["OUT"], ns["INVALID"]

N = len(list((OUT / "views").glob("*.xyz")))
VIEWS = {n: load(n) for n in range(N)}
FLAT = {}
for n, (L, xyz) in VIEWS.items():
    h, w = xyz.shape[:2]
    ok = xyz[..., 0] != INVALID
    vv, uu = np.nonzero(ok)
    FLAT[n] = (xyz[ok].astype(np.float32) / 1000.0, uu, vv)


def find_in(n, P, max_d=0.03):
    """写真 n の中で、3D 位置 P に最も近い画素（中心の座標）を返す。近いものが無ければ None。"""
    pts, uu, vv = FLAT[n]
    d = np.linalg.norm(pts - P.astype(np.float32), axis=1)
    i = int(np.argmin(d))
    return None if d[i] > max_d else np.array([uu[i] + 0.5, vv[i] + 0.5])


def measure_all(n0, ta, tb):
    """写真 n0 でタップ → 端へ吸い付け → ほかの写真でも同じ 2 点を探して測る。戻り値: 写真ごとの長さ [cm]。"""
    L, xyz = VIEWS[n0]
    sa, sb = snap(L, xyz, ta, tb), snap(L, xyz, tb, ta)
    if sa is None or sb is None:
        return None, []
    base = float(np.linalg.norm(sa[0] - sb[0]) * 100)
    others = []
    for n in VIEWS:
        if n == n0:
            continue
        qa, qb = find_in(n, sa[0]), find_in(n, sb[0])
        if qa is None or qb is None:
            continue
        Ln, xn = VIEWS[n]
        ra, rb = snap(Ln, xn, qa, qb, reach=6.0), snap(Ln, xn, qb, qa, reach=6.0)
        if ra is None or rb is None:
            continue
        w = float(np.linalg.norm(ra[0] - rb[0]) * 100)
        if abs(w - base) < 0.15 * base:      # 別の線を拾った明らかなはずれは除く
            others.append(w)
    return base, others


def run(n0, left, right, label, trials=150, jitter=10.0):
    rng = np.random.default_rng(1)
    single, multi, nv = [], [], []
    for _ in range(trials):
        f = rng.uniform(0.15, 0.85)
        a = np.array(left[0]) + f * (np.array(left[1]) - np.array(left[0]))
        b = np.array(right[0]) + f * (np.array(right[1]) - np.array(right[0]))
        base, others = measure_all(n0, a + rng.uniform(-jitter, jitter, 2), b + rng.uniform(-jitter, jitter, 2))
        if base is None:
            continue
        single.append(base)
        multi.append(float(np.median([base] + others)))
        nv.append(1 + len(others))
    single, multi = np.array(single), np.array(multi)
    print(f"{label}: 1 枚だけ 中央値 {np.median(single):.1f} cm（標準偏差 {single.std():.2f}） → 複数の写真の中央値 {np.median(multi):.1f} cm（標準偏差 {multi.std():.2f}）｜使えた写真の数 中央値 {int(np.median(nv))}（{min(nv)}〜{max(nv)}）")
    return np.median(single), np.median(multi)


print("実測: 引き戸の幅 85cm、クローゼットの幅 102cm。丁寧に測った復元値は 引き戸 88.4（写真 12 で 89.5、写真 5 で 87.3）、クローゼット 106.3")
r = []
r.append(("door", 85.0) + run(12, ((14, 40), (39, 480)), ((217, 35), (212.5, 495)), "引き戸の幅（写真 12 でタップ）"))
r.append(("door", 85.0) + run(5, ((59, 60), (56, 490)), ((256.5, 40), (227.5, 505)), "引き戸の幅（写真 5 でタップ）"))
r.append(("closet", 102.0) + run(0, ((20, 50), (23.5, 500)), ((248.5, 60), (231, 475)), "クローゼットの幅（写真 0 でタップ）"))
print("実測÷復元（この値が 3 つでそろうほど、1 か所で合わせたときにほかが合う）")
for name, col in (("1 枚だけ", 2), ("複数の写真", 3)):
    s = [x[1] / x[col] for x in r]
    print(f"  {name}: {[round(v, 3) for v in s]} ｜ ばらつき {(max(s) - min(s)) / np.mean(s) * 100:.2f}%")
