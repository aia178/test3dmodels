// 比較と設定。
import { useRef, useState } from 'preact/hooks';
import type { Layout, Placement, Room } from '../core/types';
import { judgeLayout, roomTolerance } from '../core/judge';
import { floorArea } from '../core/room';
import { bbox, pointInPolygon, polyCentroid } from '../core/geom';
import { fmtCm } from '../core/units';
import * as db from '../store/db';
import * as api from '../store/api';
import { Empty, Notice, Page, TabBar, navigate, toast, useLive } from './common';
import { Plan2D } from '../view/Plan2D';

const FX = { outlet: 'コンセント', door: '扉', window: '窓', lan: 'LAN端子', tv: 'テレビ端子', aircon: 'エアコン' } as const;

export function Compare() {
  const { data } = useLive(async () => {
    const [rooms, furniture, layouts, settings] = await Promise.all([db.getAll('rooms'), db.getAll('furniture'), db.getAll('layouts'), db.getSettings()]);
    return { rooms: rooms.filter((r) => r.outline.length), furniture, layouts, settings };
  });
  const [aId, setA] = useState(''), [bId, setB] = useState('');
  const [la, setLa] = useState(''), [lb, setLb] = useState('');
  if (!data) return <><Page title="比較"><p class="muted">読み込み中…</p></Page><TabBar current="/compare" /></>;
  const { rooms, furniture, layouts, settings } = data;
  if (rooms.length < 2) return <><Page title="比較"><Empty title="比べる部屋が足りません"><p>今の部屋と内見した部屋など、2 つ以上の部屋を登録すると、同じ家具を置いて比べられます。</p></Empty></Page><TabBar current="/compare" /></>;
  const A = rooms.find((r) => r.id === aId) ?? rooms[0], B = rooms.find((r) => r.id === bId) ?? rooms.find((r) => r.id !== A.id)!;
  const lays = (r: Room) => layouts.filter((l) => l.roomId === r.id).sort((x, y) => y.updatedAt.localeCompare(x.updatedAt));
  const LA = lays(A).find((l) => l.id === la) ?? lays(A)[0], LB = lays(B).find((l) => l.id === lb) ?? lays(B)[0];

  const copyTo = async (src: Layout, dest: Room) => {
    // 同じ家具セットを別の部屋へ。部屋の形が違うので位置は引き継がず、中央付近に並べる
    const c = polyCentroid(dest.outline), bb = bbox(dest.outline);
    const start = pointInPolygon(c, dest.outline) ? c : dest.outline[0];
    const items: Placement[] = src.items.map((p, i) => ({ ...p, id: db.uid(), x: Math.min(bb.x1, start.x + ((i % 3) - 1) * 0.3), z: Math.min(bb.z1, start.z + (Math.floor(i / 3) - 1) * 0.3), rotY: 0 }));
    const l: Layout = { id: db.uid(), roomId: dest.id, name: `${src.name}（${rooms.find((r) => r.id === src.roomId)?.name ?? ''}から）`, items, updatedAt: db.now() };
    await db.put('layouts', l);
    toast('家具セットを移しました。位置は置き直してください。');
    navigate(`/rooms/${dest.id}/layout/${l.id}`);
  };

  const col = (r: Room, l: Layout | undefined) => {
    const res = l ? judgeLayout(r, furniture, l, settings) : null;
    const items = l ? l.items.flatMap((p) => { const f = furniture.find((x) => x.id === p.furnitureId); return f ? [{ p, f }] : []; }) : [];
    return { r, l, res, items, tol: roomTolerance(r) };
  };
  const ca = col(A, LA), cb = col(B, LB);
  const row = (label: string, f: (c: ReturnType<typeof col>) => string) => <tr><th>{label}</th><td>{f(ca)}</td><td>{f(cb)}</td></tr>;

  return (
    <>
      <Page title="部屋・配置案の比較" wide>
        <div class="row2">
          {[{ c: ca, set: setA, setL: setLa, other: cb }, { c: cb, set: setB, setL: setLb, other: ca }].map(({ c, set, setL, other }) => (
            <div class="compare-col">
              <label class="field"><span>部屋</span><select value={c.r.id} onChange={(e) => set((e.target as HTMLSelectElement).value)}>{rooms.map((r) => <option value={r.id}>{r.name}</option>)}</select></label>
              <label class="field"><span>配置案</span>
                <select value={c.l?.id ?? ''} onChange={(e) => setL((e.target as HTMLSelectElement).value)} disabled={!lays(c.r).length}>
                  {lays(c.r).length ? lays(c.r).map((l) => <option value={l.id}>{l.name}</option>) : <option value="">配置案なし</option>}
                </select>
              </label>
              <div class="compare-plan"><Plan2D room={c.r} items={c.items} /></div>
              {other.l && c.r.id !== other.r.id && <button onClick={() => copyTo(other.l!, c.r)}>「{other.r.name}」の家具セットをこの部屋へ移す</button>}
            </div>
          ))}
        </div>
        <table class="tbl compare">
          <thead><tr><th /><th>{A.name}</th><th>{B.name}</th></tr></thead>
          <tbody>
            {row('広さ', (c) => `約 ${floorArea(c.r).toFixed(1)}㎡`)}
            {row('天井高', (c) => (c.r.ceiling.height ? `${fmtCm(c.r.ceiling.height)}${c.r.ceiling.status === 'observed' ? '（未確認）' : ''}` : '未確認'))}
            {row('作り方', (c) => (c.r.source === 'photo' ? '写真から復元' : '寸法入力'))}
            {row('縮尺', (c) => ({ model_estimate: '未確認（推定のまま）', calibrated: '実測で合わせた', marker: '印刷マーカーで合わせた', manual: '寸法入力' }[c.r.scale.status]))}
            {row('見込んでいる誤差', (c) => fmtCm(c.tol.tol))}
            {row('形の分からない壁', (c) => `${c.r.edges.filter((e) => e.status === 'unobserved').length} か所`)}
            {row('柱・梁・出っ張り', (c) => `${c.r.obstacles.length} 個`)}
            {(Object.keys(FX) as (keyof typeof FX)[]).map((t) => row(FX[t], (c) => { const n = c.r.fixtures.filter((f) => f.type === t).length; return n ? `${n} 個` : '未登録'; }))}
            {row('置いた家具', (c) => (c.l ? `${c.items.length} 点` : '—'))}
            {row('✕ 問題あり', (c) => (c.res ? `${c.res.issues.filter((i) => i.status === 'problem').length} 件` : '—'))}
            {row('？ 要確認', (c) => (c.res ? `${c.res.issues.filter((i) => i.status === 'needs_check').length} 件` : '—'))}
          </tbody>
        </table>
        <p class="muted small">「未登録」は、設備が無いという意味ではありません。件数は、登録された形・寸法と、いまの判定基準での結果です。部屋どうしで見込んでいる誤差が違うと、同じ配置でも「要確認」の出方が変わります。</p>
        {[ca, cb].map((c) => c.res && c.res.issues.length > 0 && (
          <details><summary>{c.r.name} の確認結果（{c.res.issues.length} 件）</summary><ul class="list">{c.res.issues.map((i) => <li>{i.status === 'problem' ? '✕ 問題あり' : '？ 要確認'}: {i.title}</li>)}</ul></details>
        ))}
      </Page>
      <TabBar current="/compare" />
    </>
  );
}

export function Settings() {
  const { data: h, error: hErr } = useLive(() => api.health(), []);
  const { data: counts } = useLive(async () => ({ rooms: (await db.getAll('rooms')).length, furniture: (await db.getAll('furniture')).length, layouts: (await db.getAll('layouts')).length }));
  const fileRef = useRef<HTMLInputElement>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [showLog, setShowLog] = useState(false);
  const backup = async () => {
    const blob = new Blob([await db.exportBackup()], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `roomfit-backup-${new Date().toISOString().slice(0, 10)}.json`; a.click(); URL.revokeObjectURL(a.href);
  };
  const restore = async (file: File | undefined) => {
    if (!file) return;
    try { const r = await db.importBackup(await file.text()); setMsg(`読み込みました（部屋 ${r.rooms}、家具 ${r.furniture}、配置案 ${r.layouts}）。同じものは上書きされます。`); } catch (e) { setMsg(`読み込めませんでした: ${(e as Error).message}`); }
  };
  const wipe = async () => {
    if (!confirm('この端末の部屋・家具・配置案と、サーバーに残っている写真・復元結果をすべて削除します。元に戻せません。')) return;
    let serverNote = '';
    try { const n = await api.deleteAllJobs(); serverNote = `サーバーの写真・復元結果 ${n} 件も削除しました。`; } catch { serverNote = 'サーバーに接続できず、サーバー側の写真は削除できていません。接続できるときにもう一度実行してください。'; }
    await db.deleteEverything();
    setMsg(`この端末のデータを削除しました。${serverNote}`);
  };
  return (
    <>
      <Page title="設定">
        <h2>データの保存先</h2>
        <p>部屋・家具・配置案は<b>この端末のブラウザの中</b>に保存されます。アカウントは不要です。別の端末へは自動では移りません（下のバックアップを使います）。</p>
        <p>写真から部屋を作るときだけ、写真を RoomFit サーバーへ送ります。{h ? h.photo_destination : ''}</p>
        {hErr && <Notice kind="warn">{hErr}（写真からの復元以外は、このまま使えます）</Notice>}
        {counts && <p class="muted">いまの保存数: 部屋 {counts.rooms}、家具 {counts.furniture}、配置案 {counts.layouts}</p>}
        <h2>バックアップ</h2>
        <div class="btn-row">
          <button onClick={backup}>バックアップを保存</button>
          <button onClick={() => fileRef.current?.click()}>バックアップから読み込む</button>
          <input ref={fileRef} type="file" accept="application/json,.json" hidden onChange={(e) => { void restore((e.target as HTMLInputElement).files?.[0]); (e.target as HTMLInputElement).value = ''; }} />
        </div>
        <p class="muted small">バックアップには部屋の形・家具・配置案が入ります。写真と、写真から復元した点は入りません。</p>
        <h2>削除</h2>
        <p class="small">部屋ごとの削除は各部屋の画面から、家具ごとの削除は家具の編集画面からできます。</p>
        <button class="danger" onClick={wipe}>すべてのデータを削除</button>
        {msg && <Notice>{msg}</Notice>}
        <h2>このアプリの判定について</h2>
        <ul class="small">
          <li>判定は保証ではなく、登録された形と寸法の範囲で分かったことの報告です。</li>
          <li>家具は直方体として扱います。脚のすき間や凹みは考慮しません。</li>
          <li>写真から作った部屋は、縮尺を実測で合わせ、別の場所で確かめるまでは誤差が分かりません。</li>
          <li>設備（コンセントなど）の自動認識、写真からの家具の立体生成、VR 表示には対応していません。</li>
        </ul>
        <h2>開発者向け</h2>
        <p class="small muted">復元の方式: {h?.pipeline ?? '不明'}</p>
        <button onClick={() => setShowLog(!showLog)}>{showLog ? '診断ログを閉じる' : '診断ログを表示'}</button>
        {showLog && <pre class="log">{db.diagLog().join('\n') || '（記録なし）'}</pre>}
      </Page>
      <TabBar current="/settings" />
    </>
  );
}
