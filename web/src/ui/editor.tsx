// 配置エディタ。2D と 3D は同じ状態（配置・選択・選んだ問題）を表示する。
import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import type { Furniture, Issue, JudgeSettings, Layout, Placement, Room } from '../core/types';
import { judgeLayout } from '../core/judge';
import { pointInPolygon, polyCentroid } from '../core/geom';
import { fmtCm, fmtCmExact } from '../core/units';
import * as db from '../store/db';
import { CmField, Notice, StatusChip, navigate, toast, useLive } from './common';
import { Plan2D } from '../view/Plan2D';
import { Scene3D } from '../view/Scene3D';
import { useCloud, useSurfaces } from './cloud';

type Tab = 'add' | 'selected' | 'issues' | 'settings';

export function Editor(props: { roomId: string; layoutId: string }) {
  const { data: room } = useLive(() => db.get('rooms', props.roomId), [props.roomId]);
  const { data: furniture } = useLive(() => db.getAll('furniture'), []);
  const { data: settings0 } = useLive(() => db.getSettings(), []);
  const [layout, setLayout] = useState<Layout | null | undefined>(undefined);
  useEffect(() => { db.get('layouts', props.layoutId).then((l) => setLayout(l ?? null)); }, [props.layoutId]);
  if (room === undefined || furniture === undefined || layout === undefined || settings0 === undefined) return <div class="editor"><p class="muted pad">読み込み中…</p></div>;
  if (!room || !layout) return <div class="editor"><div class="pad"><Notice kind="error">この配置案は見つかりません。</Notice><a class="button" href="#/">ホームへ</a></div></div>;
  return <EditorInner key={layout.id} room={room} furniture={furniture} initial={layout} settings0={settings0} />;
}

function EditorInner(props: { room: Room; furniture: Furniture[]; initial: Layout; settings0: JudgeSettings }) {
  const { room, furniture } = props;
  const [layout, setLayoutRaw] = useState(props.initial);
  const [settings, setSettings] = useState(props.settings0);
  const [undo, setUndo] = useState<Placement[][]>([]);
  const [redo, setRedo] = useState<Placement[][]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [issueId, setIssueId] = useState<string | null>(null);
  const [mode, setMode] = useState<'2d' | '3d'>('2d');
  const [tab, setTab] = useState<Tab>(props.initial.items.length ? 'issues' : 'add');
  const [drag, setDrag] = useState<{ id: string; x: number; z: number } | null>(null);
  const [saveState, setSaveState] = useState<'saved' | 'saving' | 'error'>('saved');
  const [showCloud, setShowCloud] = useState(false);
  const [showPhoto, setShowPhoto] = useState(true);
  const cloud = useCloud(room.id, room.hasPointCloud && showCloud);
  const surfaces = useSurfaces(room, mode === '3d' && showPhoto);

  // ---- 保存: 変更のたびに少し待って保存。画面を離れる・隠れるときは待たずに保存。
  const latest = useRef(layout); latest.current = layout;
  const dirty = useRef(false);
  const timer = useRef(0);
  const flush = async () => {
    if (!dirty.current) return;
    dirty.current = false; clearTimeout(timer.current);
    try { await db.put('layouts', latest.current); setSaveState('saved'); } catch (e) { dirty.current = true; setSaveState('error'); db.diag(`layout save failed: ${(e as Error).message}`); }
  };
  const setLayout = (next: Layout) => {
    setLayoutRaw(next); latest.current = next; dirty.current = true; setSaveState('saving');
    clearTimeout(timer.current); timer.current = window.setTimeout(flush, 400);
  };
  useEffect(() => {
    const onHide = () => { void flush(); };
    document.addEventListener('visibilitychange', onHide); window.addEventListener('pagehide', onHide);
    return () => { document.removeEventListener('visibilitychange', onHide); window.removeEventListener('pagehide', onHide); void flush(); };
  }, []);

  const commit = (items: Placement[]) => {
    setUndo((u) => [...u.slice(-49), layout.items]); setRedo([]);
    setLayout({ ...layout, items, updatedAt: db.now() });
  };
  const doUndo = () => { if (!undo.length) return; setRedo((r) => [...r, layout.items]); setLayout({ ...layout, items: undo[undo.length - 1], updatedAt: db.now() }); setUndo(undo.slice(0, -1)); };
  const doRedo = () => { if (!redo.length) return; setUndo((u) => [...u, layout.items]); setLayout({ ...layout, items: redo[redo.length - 1], updatedAt: db.now() }); setRedo(redo.slice(0, -1)); };
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'SELECT') return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) doRedo(); else doUndo(); }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); doRedo(); }
      else if ((e.key === 'Delete' || e.key === 'Backspace') && sel) { e.preventDefault(); commit(layout.items.filter((p) => p.id !== sel)); setSel(null); }
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  });

  const fmap = useMemo(() => new Map(furniture.map((f) => [f.id, f])), [furniture]);
  // 判定は確定した配置に対してだけ行う（ドラッグ中の毎フレームでは行わない）
  const result = useMemo(() => judgeLayout(room, furniture, layout, settings), [room, furniture, layout.items, settings]);
  const issue = result.issues.find((i) => i.id === issueId) ?? null;
  const targets = useMemo(() => {
    const m = new Map<string, 'problem' | 'needs_check'>();
    for (const i of result.issues) for (const t of i.targets) if (i.status === 'problem' || !m.has(t)) m.set(t, i.status === 'problem' ? 'problem' : m.get(t) ?? 'needs_check');
    return m;
  }, [result]);
  const viewItems = layout.items.flatMap((p) => { const f = fmap.get(p.furnitureId); return f ? [{ p: drag?.id === p.id ? { ...p, x: drag.x, z: drag.z } : p, f }] : []; });
  const selP = layout.items.find((p) => p.id === sel) ?? null;
  const selF = selP ? fmap.get(selP.furnitureId) ?? null : null;
  const nProblem = result.issues.filter((i) => i.status === 'problem').length, nCheck = result.issues.filter((i) => i.status === 'needs_check').length;

  const onMove = (id: string, x: number, z: number, done: boolean) => {
    if (!done) { setDrag({ id, x, z }); return; }
    setDrag(null);
    commit(layout.items.map((p) => (p.id === id ? { ...p, x, z } : p)));
  };
  const patchSel = (patch: Partial<Placement>) => { if (selP) commit(layout.items.map((p) => (p.id === selP.id ? { ...p, ...patch } : p))); };
  const addFurniture = (f: Furniture) => {
    const c = polyCentroid(room.outline);
    const pos = pointInPolygon(c, room.outline) ? c : room.outline[0];
    const p: Placement = { id: db.uid(), furnitureId: f.id, x: pos.x, z: pos.z, rotY: 0, elevation: 0 };
    commit([...layout.items, p]); setSel(p.id); setTab('selected');
    toast(`「${f.name}」を部屋の中央に置きました。ドラッグで動かせます。`);
  };
  const select = (id: string | null) => { setSel(id); if (id) setTab('selected'); };
  const pickIssue = (i: Issue) => { setIssueId(i.id === issueId ? null : i.id); const pid = i.targets.find((t) => layout.items.some((p) => p.id === t)); if (pid) setSel(pid); };
  const applyFix = (i: Issue) => {
    if (!i.fix) return;
    commit(layout.items.map((p) => (p.id === i.fix!.placementId ? { ...p, x: p.x + i.fix!.dx, z: p.z + i.fix!.dz } : p)));
    setIssueId(null); toast('提案の分だけ動かしました。下の一覧で結果を確かめてください。');
  };

  return (
    <div class="editor">
      <header class="bar">
        <a class="back" href={`#/rooms/${room.id}`} onClick={() => void flush()}>‹ {room.name}</a>
        <input class="title-input" value={layout.name} aria-label="配置案の名前" maxLength={30} onInput={(e) => setLayout({ ...layout, name: (e.target as HTMLInputElement).value })} />
        <span class={`save ${saveState}`} role="status">{saveState === 'saved' ? '✓ 保存済み' : saveState === 'saving' ? '保存中…' : '！保存できませんでした'}</span>
      </header>
      <div class="toolbar">
        <div class="seg"><button class={mode === '2d' ? 'on' : ''} aria-pressed={mode === '2d'} onClick={() => setMode('2d')}>平面図</button><button class={mode === '3d' ? 'on' : ''} aria-pressed={mode === '3d'} onClick={() => setMode('3d')}>3D</button></div>
        <button onClick={doUndo} disabled={!undo.length} aria-label="元に戻す">↶ 元に戻す</button>
        <button onClick={doRedo} disabled={!redo.length} aria-label="やり直す">↷ やり直す</button>
        {mode === '3d' && room.surfaces && <label class="check inline"><input type="checkbox" checked={showPhoto} onChange={(e) => setShowPhoto((e.target as HTMLInputElement).checked)} /> 壁と床に写真を貼る</label>}
        {mode === '3d' && room.hasPointCloud && <label class="check inline"><input type="checkbox" checked={showCloud} onChange={(e) => setShowCloud((e.target as HTMLInputElement).checked)} /> 写真から復元した点</label>}
        <span class="spacer" />
        <StatusChip status="problem" count={nProblem} /><StatusChip status="needs_check" count={nCheck} />
      </div>
      <div class="split">
        <div class="split-view">
          {mode === '2d'
            ? <Plan2D room={room} items={viewItems} selectedId={sel} issue={issue} issueTargets={targets} onSelect={select} onMove={onMove} showLengths />
            : <Scene3D room={room} items={viewItems} selectedId={sel} issue={issue} issueTargets={targets} cloud={cloud?.pos} cloudColors={cloud?.col} showCloud={showCloud} surfaces={showPhoto ? surfaces : null} onSelect={select} onMove={onMove} />}
        </div>
        <div class="split-panel">
          <div class="seg" role="tablist">
            <button role="tab" aria-selected={tab === 'add'} class={tab === 'add' ? 'on' : ''} onClick={() => setTab('add')}>家具を置く</button>
            <button role="tab" aria-selected={tab === 'selected'} class={tab === 'selected' ? 'on' : ''} onClick={() => setTab('selected')}>選択中</button>
            <button role="tab" aria-selected={tab === 'issues'} class={tab === 'issues' ? 'on' : ''} onClick={() => setTab('issues')}>確認結果 {nProblem + nCheck > 0 ? `(${nProblem + nCheck})` : ''}</button>
            <button role="tab" aria-selected={tab === 'settings'} class={tab === 'settings' ? 'on' : ''} onClick={() => setTab('settings')}>基準</button>
          </div>

          {tab === 'add' && (
            <>
              {furniture.length === 0 ? <Notice>まだ家具が登録されていません。</Notice> : <p class="muted small">タップすると部屋の中央に置かれます。</p>}
              <ul class="list">
                {furniture.map((f) => (
                  <li><button class="link row" onClick={() => addFurniture(f)}>
                    <i class="swatch" style={{ background: f.color }} /><span><b>{f.name}</b><br /><span class="muted small">{fmtCmExact(f.w)} × {fmtCmExact(f.d)} × 高さ {fmtCmExact(f.h)}{f.dimConfirmed ? '' : ' ・寸法未確認'}</span></span><span class="plus" aria-hidden="true">＋</span>
                  </button></li>
                ))}
              </ul>
              <a class="button" href="#/furniture/new" onClick={() => void flush()}>新しい家具を登録</a>
            </>
          )}

          {tab === 'selected' && (!selP || !selF ? <Notice>家具をタップして選ぶと、位置や向きを数値で調整できます。</Notice> : (
            <section class="panel-card">
              <h3>{selF.name}</h3>
              <p class="muted small">幅 {fmtCmExact(selF.w)} × 奥行き {fmtCmExact(selF.d)} × 高さ {fmtCmExact(selF.h)}{selF.dimConfirmed ? '' : '（寸法未確認）'}・直方体として判定</p>
              <div class="btn-row">
                <button onClick={() => patchSel({ rotY: selP.rotY + Math.PI / 2 })}>⟲ 左へ 90°</button>
                <button onClick={() => patchSel({ rotY: selP.rotY - Math.PI / 2 })}>⟳ 右へ 90°</button>
              </div>
              <div class="row2">
                <CmField label="左右の位置（図の左が小さい）" value={selP.x} min={-100000} onChange={(x) => x != null && patchSel({ x })} />
                <CmField label="上下の位置（図の上が小さい）" value={selP.z} min={-100000} onChange={(z) => z != null && patchSel({ z })} />
              </div>
              <div class="nudge" role="group" aria-label="1cm ずつ動かす">
                <button onClick={() => patchSel({ z: selP.z - 0.01 })} aria-label="上へ 1cm">↑</button>
                <button onClick={() => patchSel({ x: selP.x - 0.01 })} aria-label="左へ 1cm">←</button>
                <span>1cm</span>
                <button onClick={() => patchSel({ x: selP.x + 0.01 })} aria-label="右へ 1cm">→</button>
                <button onClick={() => patchSel({ z: selP.z + 0.01 })} aria-label="下へ 1cm">↓</button>
              </div>
              <div class="row2">
                <label class="field"><span>向き（度）</span><input type="number" step="1" value={Math.round(((selP.rotY * 180) / Math.PI) % 360)} onChange={(e) => { const n = Number((e.target as HTMLInputElement).value); if (Number.isFinite(n)) patchSel({ rotY: (n * Math.PI) / 180 }); }} /></label>
                <CmField label="床からの高さ（壁掛けなど）" value={selP.elevation} onChange={(y) => patchSel({ elevation: y ?? 0 })} />
              </div>
              <div class="btn-row">
                <button onClick={() => { const c: Placement = { ...selP, id: db.uid(), x: selP.x + 0.2, z: selP.z + 0.2 }; commit([...layout.items, c]); setSel(c.id); }}>複製</button>
                <button class="danger" onClick={() => { commit(layout.items.filter((p) => p.id !== selP.id)); setSel(null); }}>取り除く</button>
                <a class="button" href={`#/furniture/${selF.id}`} onClick={() => void flush()}>寸法を編集</a>
              </div>
              {result.issues.filter((i) => i.targets.includes(selP.id)).length > 0 && <p class="small">この家具に関する確認結果: {result.issues.filter((i) => i.targets.includes(selP.id)).length} 件 <button class="link" onClick={() => setTab('issues')}>一覧を見る</button></p>}
            </section>
          ))}

          {tab === 'issues' && (
            <>
              {layout.items.length === 0 ? <Notice>家具を置くと、ここに確認結果が出ます。</Notice> : result.issues.length === 0 ? (
                <Notice><b>✓ 確認した範囲では、問題を検出しませんでした。</b><p class="small">調べた項目: 壁・柱とのぶつかり、家具どうしの重なり、天井、扉の開く範囲、使うための空間、入口からの通路、コンセントなどの設備。これは保証ではなく、登録された形と寸法の範囲での結果です。</p></Notice>
              ) : null}
              <ul class="issues">
                {result.issues.map((i) => (
                  <li class={`issue ${i.status} ${i.id === issueId ? 'on' : ''}`}>
                    <button class="issue-head" onClick={() => pickIssue(i)} aria-expanded={i.id === issueId}>
                      <StatusChip status={i.status} /><span class="issue-title">{i.title}</span>
                    </button>
                    {i.id === issueId && (
                      <div class="issue-body">
                        <p>{i.reason}</p>
                        {i.dims.length > 0 && <dl class="dims">{i.dims.map((d) => <><dt>{d.label}</dt><dd>約 {fmtCm(d.valueM)}</dd></>)}</dl>}
                        {i.unverified.length > 0 && <div class="unv"><b>確認が必要なこと</b><ul>{i.unverified.map((u) => <li>{u}</li>)}</ul></div>}
                        {i.suggestion && <p class="suggest">→ {i.suggestion}</p>}
                        {i.fix && <button class="primary" onClick={() => applyFix(i)}>提案の分だけ動かす</button>}
                      </div>
                    )}
                  </li>
                ))}
              </ul>
              <p class="muted small">{result.toleranceBasis}</p>
            </>
          )}

          {tab === 'settings' && (
            <section class="panel-card">
              <h3>判定の基準（自分で決める目安）</h3>
              <CmField label="通路として確保したい幅" value={settings.passageWidthM} min={20} onChange={(x) => { if (x) { const s = { ...settings, passageWidthM: x }; setSettings(s); void db.setMeta('settings', s); } }} hint="初期値 60cm は、人ひとりが正面を向いて通る目安として置いた値です。法令や安全の基準ではありません。暮らし方に合わせて変えてください。" />
              <CmField label="コンセントなどの前に空けたい奥行き" value={settings.outletAccessM} min={1} onChange={(x) => { if (x) { const s = { ...settings, outletAccessM: x }; setSettings(s); void db.setMeta('settings', s); } }} hint="プラグの抜き差しに使う空間の目安です。" />
              <p class="muted small">{result.toleranceBasis}</p>
              <button onClick={async () => { await flush(); const c: Layout = { ...layout, id: db.uid(), name: `${layout.name} のコピー`, updatedAt: db.now() }; await db.put('layouts', c); navigate(`/rooms/${room.id}/layout/${c.id}`); toast('配置案を複製しました'); }}>この配置案を複製</button>
              <button class="danger" onClick={async () => { if (!confirm(`配置案「${layout.name}」を削除します。`)) return; dirty.current = false; clearTimeout(timer.current); await db.del('layouts', layout.id); navigate(`/rooms/${room.id}`); }}>この配置案を削除</button>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
