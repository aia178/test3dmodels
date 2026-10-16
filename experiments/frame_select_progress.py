"""P2 の進み具合を experiments/out/frame_select/進捗.md に書き出す（計算 1 本ごとに台本から呼ぶ。Claude を起こさずに進捗を見るため）。

使い方（server フォルダで）: uv run python ../experiments/frame_select_progress.py [いま計算中のもの] [全部の本数]
"""
import json
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out" / "frame_select"
NAME = {"even": "等間隔", "overlap": "重なり優先", "mixed": "重なり＋視点＋正面"}


def cm(r, k, key="abs_error_cm_median"):
    return f"{r[k][key]}" if k in r else "—"


def main() -> None:
    running = sys.argv[1] if len(sys.argv) > 1 else ""
    total = sys.argv[2] if len(sys.argv) > 2 else "?"
    rows = []
    for f in sorted(OUT.glob("*/eval.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        parts = f.parent.name.split("_")
        data, how, mode = parts[0], parts[1], parts[2]
        shift = parts[3][1:] if len(parts) > 3 else "0"
        rows.append((data, mode, how, int(shift), r))
    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    lines = [f"# P2 コマ選びの比較：進み具合", "", f"更新: {time.strftime('%m/%d %H:%M')}　",
             f"いま計算中: {running or 'なし（終わり）'}　終わった数（揺れの確認）: {sum(1 for x in rows if x[3] > 0)} / {total}", "",
             "写真をまたぐ距離の誤差（中央値、cm）と、写真ごとの大きさのばらつき（四分位幅）。ずらし＝選ぶ開始位置をずらした版（揺れの確認）。", "",
             "| データ | 入力 | 選び方 | ずらし | 0.5〜1.5m | 1.5〜3m | 3m 以上 | 大きさのばらつき | 重なり |",
             "|---|---|---|---|---|---|---|---|---|"]
    for data, mode, how, shift, r in rows:
        lines.append(f"| {data} | {'位置つき' if mode == 'pose' else '写真だけ'} | {NAME.get(how, how)} | {shift} | {cm(r, 'cross_0.5-1.5m')} | "
                     f"{cm(r, 'cross_1.5-3m')} | {cm(r, 'cross_>3m')} | {r['scale_iqr_pct']}% | {round(r['capture_quality']['overlap'] * 100)}% |")
    (OUT / "進捗.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
