"""評価: 同じ角を撮影アプリ（ARCore）の連続撮影で撮った写真 13 枚からの復元を、同じ実測と比べる。

標準カメラの 13 枚（measure_home_1012.py）とは別の撮影。壁から約 1m と近く、扉もクローゼットも 1 枚に収まっていないので、
幅は「左端が写った写真」と「右端が写った写真」の 3D 位置をまたいで測る（すべて同じ座標系にある）。
端を探す関数は measure_home_1012.py のものを使う。使い方（server フォルダで）:
  uv run python ../experiments/measure_home_arcore.py [出力フォルダ]
"""
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "out" / "home_arcore13"
src = (here / "measure_home_1012.py").read_text(encoding="utf-8")
lib = src[: src.index('print(f"フォルダ:')]            # 関数の定義まで（その部屋専用の計測の手前）
sys.argv = ["measure", str(OUT)]
ns = {"__file__": str(here / "measure_home_1012.py"), "__name__": "lib"}
exec(compile(lib, "measure_home_1012.py", "exec"), ns)
edge_points, panel_top, panel_bottom, line_fit, at_height = (ns[k] for k in ("edge_points", "panel_top", "panel_bottom", "line_fit", "at_height"))
MEASURED = ns["MEASURED"]


def width_across(PL, PR):
    cl, dl = line_fit(PL)
    cr, dr = line_fit(PR)
    return [float(np.linalg.norm((at_height(cl, dl, y) - at_height(cr, dr, y))[[0, 2]])) * 100 for y in (0.6, 1.2, 1.8)]


res = {}
# 引き戸: 上枠の上端（写真 5, 6）、左の縦枠の外側（写真 6）、戸の右端（写真 4）
tops = []
for v, p0, p1 in ((5, (5, 142), (285, 122)), (6, (45, 138), (290, 95))):
    P, _ = edge_points(v, "col", range(10, 286, 5), p0, p1, 7, True, +1)
    if len(P) > 8:
        tops.append(float(np.median(P[:, 1])) * 100)
        print(f"引き戸 上枠の上端（写真 {v}）: 高さ {tops[-1]:.1f} cm（点数 {len(P)}）")
if tops:
    res["door_height"] = float(np.mean(tops))
L, _ = edge_points(6, "row", range(170, 506, 5), (47, 165), (40, 505), 7, True, +1, (1.0, 5.0))
R, _ = edge_points(4, "row", range(50, 506, 5), (239, 45), (248.5, 510), 7, False, -1)
print(f"引き戸 左端の点 {len(L)}（写真 6）／右端の点 {len(R)}（写真 4）")
if len(L) > 10 and len(R) > 10:
    w = width_across(L, R)
    res["door_width"] = w[1]
    print(f"引き戸の幅: {w[1]:.1f} cm（高さ 60cm で {w[0]:.1f} / 180cm で {w[2]:.1f}）")

# クローゼット: 板の上端（写真 8）、板の下端（写真 10）、枠の左端（写真 9, 10）、枠の右端（写真 8, 11）
Tp, _ = panel_top(8, range(20, 251, 3), lambda c: 25 + c * (80 - 25) / 259)
Bp, _ = panel_bottom(10, [c for c in range(80, 281, 4) if not (185 <= c <= 205)], lambda c: 436 + (c - 70) * (421 - 436) / (290 - 70))
print(f"クローゼット 板の上端 {len(Tp)} 点 / 下端 {len(Bp)} 点")
if len(Tp) > 5 and len(Bp) > 5:
    t, b = float(np.median(Tp[:, 1])) * 100, float(np.median(Bp[:, 1])) * 100
    res["closet_panel"] = t - b
    print(f"クローゼットの板: 上端 {t:.1f} / 下端 {b:.1f} → 長さ {t - b:.1f} cm")
Ls = [edge_points(9, "row", range(30, 500, 5), (49, 20), (64, 505), 7, True, +1, (1.0, 5.0))[0],
      edge_points(10, "row", range(40, 430, 5), (20, 27), (63, 436), 7, True, +1, (1.0, 5.0))[0]]
Rs = [edge_points(8, "row", range(95, 500, 5), (261, 82), (286, 509), 7, False, -1, (1.0, 5.0))[0],
      edge_points(11, "row", range(30, 370, 5), (230, 18), (165, 377), 7, False, -1, (1.0, 5.0))[0]]
print("クローゼット 左端の点", [len(x) for x in Ls], "／右端の点", [len(x) for x in Rs])
ws = []
for a, PL in zip((9, 10), Ls):
    for b, PR in zip((8, 11), Rs):
        if len(PL) > 10 and len(PR) > 10:
            w = width_across(PL, PR)
            ws.append(w[1])
            print(f"  クローゼットの幅（左=写真 {a}, 右=写真 {b}）: {w[1]:.1f} cm（60cm で {w[0]:.1f} / 180cm で {w[2]:.1f}）")
if ws:
    res["closet_width"] = float(np.mean(ws))

print("== まとめ（復元 / 実測 / 差）")
sf = []
for k in ("door_width", "door_height", "closet_width", "closet_panel"):
    if k in res:
        m = MEASURED[k]
        sf.append(m / res[k])
        print(f"  {k}: {res[k]:.1f} / {m:.0f} / {res[k] - m:+.1f} cm ({(res[k] / m - 1) * 100:+.1f}%)")
    else:
        print(f"  {k}: 測れず")
if len(sf) >= 2:
    print(f"  実測÷復元: {[round(x, 3) for x in sf]} ｜ ばらつき {(max(sf) - min(sf)) / np.mean(sf) * 100:.1f}%")
