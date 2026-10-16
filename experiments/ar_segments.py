"""実験（P1）: サーバーの検算（roomfit.ar_check.check_capture）を、保存した撮影アプリの記録すべてに通し、区間の決まり方を一覧にする。

使い方（server フォルダで）: uv run python ../experiments/ar_segments.py
出力: 標準出力に表、experiments/out/ar_segments.json に詳細（組ごとの比較は含めない）
"""
import json
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "server"))
from roomfit.ar_check import check_capture  # noqa: E402


def main() -> None:
    report = {}
    for d in sorted([*(here / "data" / "arcore").iterdir(), *(here / "data" / "arcore_iqoo").iterdir()]):
        if not d.is_dir() or len(list(d.glob("[0-9][0-9][0-9].json"))) < 3:
            continue
        sess = json.loads((d / "session.json").read_text(encoding="utf-8")) if (d / "session.json").exists() else {}
        t = time.time()
        r = check_capture(d)
        sec = time.time() - t
        segs = " ".join(f"{g['first']}-{g['last']}{'*' if g.get('chosen') else ''}" for g in r["segments"] if g["usable"])
        n = len(r["use_pose"])
        key = f"{d.parent.name}/{d.name}"
        print(f"{key:28s} {sess.get('device', '?')[:7]:7s} 飛び={sess.get('poseJumps')} コマ={n:2d} {r['status']:9s} 使う={sum(r['use_pose']):2d} "
              f"判定={r['pairs_judged']:3d} 食い違い={r['pairs_bad']:3d} 動き={r['extent_m']} 大きさ={r['scale_cue']} | 使える区間 {segs} ({sec:.0f} 秒)", flush=True)
        report[key] = {"device": sess.get("device"), "poseJumps": sess.get("poseJumps"), "sec": round(sec, 1),
                       **{k: v for k, v in r.items() if k != "pairs"}}
    (here / "out" / "ar_segments.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
