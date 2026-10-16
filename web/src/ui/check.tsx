// 部屋の確認: 形（壁・柱・天井）、縮尺と精度、設備の登録。
import { useState } from 'preact/hooks';
import type { Fixture, FixtureType, Obstacle, Room, Vec2 } from '../core/types';
import {
  FIXTURE_DEFAULTS, accuracyTable, addMeasurement, calibrateByCeiling, edgeLength, moveEdge, nearestEdge, newFixture, removeFixture, removeMeasurement,
  removeObstacle, setCeiling, setEdgeStatus, upsertFixture, upsertObstacle,
} from '../core/room';
import { roomTolerance } from '../core/judge';
import { dist, polyCentroid } from '../core/geom';
import { fmtCm, fmtCmExact } from '../core/units';
import * as db from '../store/db';
import { CmField, Notice, Page, toast, useLive } from './common';
import { Plan2D } from '../view/Plan2D';
import { useCloud } from './cloud';
import { PhotoMeasure, pickDistance, type Pick3 } from './PhotoMeasure';

const EDGE_LABEL = { observed: '自動で復元（未確認）', unobserved: '形が分からない（未確認）', confirmed: '確認済み', manual: '手入力' } as const;

export function RoomCheck(props: { roomId: string }) {
  const { data: room } = useLive(() => db.get('rooms', props.roomId), [props.roomId]);
  const [tab, setTab] = useState<'shape' | 'scale' | 'fixtures'>('shape');
  const [edge, setEdge] = useState<number | null>(null);
  const [obId, setObId] = useState<string | null>(null);
  const [fxId, setFxId] = useState<string | null>(null);
  const [addType, setAddType] = useState<FixtureType | null>(null);
  const [pick, setPick] = useState<Vec2[]>([]);
  const [photoPick, setPhotoPick] = useState<Pick3[]>([]);
  const [pickMode, setPickMode] = useState<'photo' | 'plan'>('photo');
  const [ceilIn, setCeilIn] = useState<number | null>(null);
  const [mLabel, setMLabel] = useState('');
  const [mLen, setMLen] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showCloud, setShowCloud] = useState(true);
  const cloud = useCloud(props.roomId, !!room?.hasPointCloud);
  if (room === undefined) return <Page title="部屋の確認" back="/"><p class="muted">読み込み中…</p></Page>;
  if (!room || !room.outline.length) return <Page title="部屋の確認" back="/"><Notice kind="error">この部屋はまだ形がありません。</Notice></Page>;

  const save = async (next: Room) => { setErr(null); await db.put('rooms', next); };
  const tryDo = (f: () => Room) => { try { void save(f()); } catch (e) { setErr((e as Error).message); } };
  const acc = accuracyTable(room);
  const tol = roomTolerance(room);
  const hasCal = room.measurements.some((m) => m.role === 'calibration');
  const hasViews = (room.views?.length ?? 0) > 0;
  const ob = room.obstacles.find((o) => o.id === obId) ?? null;
  const fx = room.fixtures.find((f) => f.id === fxId) ?? null;

  const snap = (p: Vec2): Vec2 => {
    let best: Vec2 | null = null, bd = 0.15;
    for (const q of room.outline) { const d = dist(p, q); if (d < bd) { bd = d; best = q; } }
    for (const o of room.obstacles) for (const q of [{ x: o.rect[0], z: o.rect[1] }, { x: o.rect[2], z: o.rect[1] }, { x: o.rect[2], z: o.rect[3] }, { x: o.rect[0], z: o.rect[3] }]) { const d = dist(p, q); if (d < bd) { bd = d; best = q; } }
    if (best) return best;
    const ne = nearestEdge(room, p);
    return ne.distance < 0.1 ? ne.point : p;
  };
  const onTapAny = (p: Vec2, e?: number) => {
    if (tab === 'scale') { setPick((cur) => (cur.length >= 2 ? [snap(p)] : [...cur, snap(p)])); return; }
    if (tab === 'fixtures' && addType && e != null) {
      const ne = nearestEdge(room, p);
      const f = newFixture(db.uid(), addType, ne.edge, ne.offset, room.fixtures.some((x) => x.door?.isEntrance));
      if (edgeLength(room, ne.edge) < f.width) { setErr(`この壁は${FIXTURE_DEFAULTS[addType].label}（幅 ${fmtCm(f.width)}）より短いため置けません。`); return; }
      tryDo(() => upsertFixture(room, f)); setFxId(f.id); setAddType(null); return;
    }
    if (tab === 'shape' && e != null) { setEdge(e); setObId(null); }
  };

  const saveMeasurement = () => {
    const usePhoto = pickMode === 'photo' && (room.views?.length ?? 0) > 0;
    if ((usePhoto ? photoPick : pick).length !== 2) { setErr(usePhoto ? '写真の上で 2 点を選んでください。' : '平面図で 2 点を選んでください。'); return; }
    if (!mLen) { setErr('実際の長さを入れてください。'); return; }
    const role = hasCal ? 'validation' as const : 'calibration' as const;
    const label = mLabel.trim() || `測定 ${room.measurements.length + 1}`;
    if (usePhoto) {
      const [p, q] = photoPick;
      tryDo(() => addMeasurement(room, { id: db.uid(), label, a: { x: p.x, z: p.z }, b: { x: q.x, z: q.z }, ya: p.y, yb: q.y, photo: { n: p.n, ua: p.u, va: p.v, ub: q.u, vb: q.v }, measuredM: mLen, role }));
    } else tryDo(() => addMeasurement(room, { id: db.uid(), label, a: pick[0], b: pick[1], measuredM: mLen, role }));
    setPhotoPick([]);
    toast(role === 'calibration' ? '縮尺を合わせました。別の場所でも測って確かめてください。' : '検証用の測定を記録しました。');
    setPick([]); setMLabel(''); setMLen(null);
  };

  return (
    <Page title={`${room.name} — 部屋の確認`} back={`/rooms/${room.id}`} wide>
      <div class="split">
        <div class="split-view">
          <Plan2D room={room} focusKey={String(room.scale.factor)} cloud={showCloud ? cloud?.pos : null} showEdgeNumbers showLengths selectedEdge={tab === 'shape' ? edge : null}
            selectedObstacleId={obId} selectedFixtureId={fxId} pickPoints={tab === 'scale' ? pick : []}
            onTapEdge={(e, p) => onTapAny(p, e)} onTapFloor={(p) => { if (tab === 'scale') onTapAny(p); else { setEdge(null); setObId(null); setFxId(null); } }}
            onTapObstacle={(id) => { if (tab === 'scale') return; setTab('shape'); setObId(id); setEdge(null); }}
            onTapFixture={(id) => { if (tab === 'scale') return; setTab('fixtures'); setFxId(id); }}
            onMoveObstacle={tab === 'shape' ? (id, dx, dz, done) => {
              const o = room.obstacles.find((x) => x.id === id); if (!o) return;
              if (done) return;
              void save(upsertObstacle(room, { ...o, rect: [o.rect[0] + dx, o.rect[1] + dz, o.rect[2] + dx, o.rect[3] + dz], confirm: o.confirm === 'auto' ? 'auto' : 'manual' }));
            } : undefined} />
          <div class="view-legend">
            <span><i class="lg solid" />確認済みの壁</span><span><i class="lg thin" />自動復元（未確認）</span><span><i class="lg dash" />形が分からない</span>
            {room.hasPointCloud && <label><input type="checkbox" checked={showCloud} onChange={(e) => setShowCloud((e.target as HTMLInputElement).checked)} /> 復元した点を重ねる</label>}
          </div>
        </div>
        <div class="split-panel">
          <div class="seg" role="tablist">
            <button role="tab" aria-selected={tab === 'shape'} class={tab === 'shape' ? 'on' : ''} onClick={() => setTab('shape')}>① 形</button>
            <button role="tab" aria-selected={tab === 'scale'} class={tab === 'scale' ? 'on' : ''} onClick={() => setTab('scale')}>② 縮尺と精度（任意）</button>
            <button role="tab" aria-selected={tab === 'fixtures'} class={tab === 'fixtures' ? 'on' : ''} onClick={() => setTab('fixtures')}>③ 設備</button>
          </div>
          {err && <Notice kind="error">{err}</Notice>}

          {tab === 'shape' && (
            <>
              <p class="muted small">平面図の壁をタップして、実物と合っているか確かめます。灰色の点は写真から復元した点（床から 30〜150cm）で、壁の線と見比べる目安です。</p>
              {edge != null && room.edges[edge] ? (
                <section class="panel-card">
                  <h3>壁 {edge + 1} <span class="muted">（{fmtCm(edgeLength(room, edge))}）</span></h3>
                  <p>状態: <b>{EDGE_LABEL[room.edges[edge].status]}</b></p>
                  <div class="btn-row">
                    <button class="primary" disabled={room.edges[edge].status === 'confirmed' || room.edges[edge].status === 'manual'} onClick={() => save(setEdgeStatus(room, edge, 'confirmed'))}>実物と合っている</button>
                    <button disabled={room.edges[edge].status === 'unobserved'} onClick={() => save(setEdgeStatus(room, edge, 'unobserved'))}>形が分からない</button>
                  </div>
                  <p class="small">位置がずれているとき（動かすと確認済みになります）:</p>
                  <div class="btn-row">
                    {[-5, -1, 1, 5].map((c) => <button onClick={() => tryDo(() => moveEdge(room, edge, c / 100))}>{c > 0 ? `室内側へ ${c}cm` : `外側へ ${-c}cm`}</button>)}
                  </div>
                </section>
              ) : <Notice>壁をタップすると、状態の確認と位置の修正ができます。</Notice>}
              {room.edges.some((e) => e.status === 'observed') && (
                <button onClick={() => { if (confirm('自動で復元した壁をすべて「確認済み」にします。実物と見比べましたか？')) void save(room.edges.reduce((r, e, i) => (e.status === 'observed' ? setEdgeStatus(r, i, 'confirmed') : r), room)); }}>
                  自動復元の壁をまとめて確認済みにする
                </button>
              )}
              <section class="panel-card">
                <h3>天井高</h3>
                <CmField label={room.ceiling.status === 'observed' ? '自動で復元した値（未確認）' : room.ceiling.height == null ? '分かりません（入力すると判定に使います）' : '入力した値'} value={room.ceiling.height} onChange={(h) => save(setCeiling(room, h))} />
              </section>
              <section class="panel-card">
                <h3>柱・梁・出っ張り</h3>
                <p class="muted small">壁の凹凸は輪郭に含まれます。足りないもの（復元されなかった柱、梁など）はここで足し、平面図でドラッグして位置を決めます。</p>
                <ul class="list">
                  {room.obstacles.map((o) => (
                    <li><button class={obId === o.id ? 'link on' : 'link'} onClick={() => { setObId(o.id); setEdge(null); }}>
                      {o.label ?? { pillar: '柱', beam: '梁', protrusion: '出っ張り', unknown_vertical: '縦の構造物（種類未確認）' }[o.kind]} — {fmtCm(o.rect[2] - o.rect[0])} × {fmtCm(o.rect[3] - o.rect[1])}{o.confirm === 'auto' ? '（自動・未確認）' : ''}
                    </button></li>
                  ))}
                </ul>
                <div class="btn-row">
                  {(['pillar', 'beam', 'protrusion'] as const).map((k) => (
                    <button onClick={() => {
                      const c = polyCentroid(room.outline);
                      const H = room.ceiling.height ?? 2.4;
                      const o: Obstacle = k === 'beam'
                        ? { id: db.uid(), kind: k, rect: [c.x - 0.5, c.z - 0.15, c.x + 0.5, c.z + 0.15], yMin: H - 0.3, yMax: H, confirm: 'manual' }
                        : { id: db.uid(), kind: k, rect: [c.x - 0.15, c.z - 0.15, c.x + 0.15, c.z + 0.15], yMin: 0, yMax: H, confirm: 'manual' };
                      void save(upsertObstacle(room, o)); setObId(o.id); setEdge(null);
                    }}>＋ {{ pillar: '柱', beam: '梁', protrusion: '出っ張り' }[k]}</button>
                  ))}
                </div>
                {ob && <ObstacleForm key={ob.id} ob={ob} onChange={(o) => save(upsertObstacle(room, o))} onDelete={() => { void save(removeObstacle(room, ob.id)); setObId(null); }} />}
              </section>
            </>
          )}

          {tab === 'scale' && (
            <>
              <Notice kind={room.scale.status === 'model_estimate' ? 'warn' : 'info'}>{tol.basis}</Notice>
              <section class="panel-card">
                <h3>{hasCal ? '別の場所でも確かめる（検証）' : '長さの分かる場所で大きさを合わせる（任意）'}</h3>
                {!hasCal && <p class="small">しなくても配置は試せます。メジャーが無ければ、寸法の分かっている物（手持ちの家具の幅、扉の幅など）でも構いません。</p>}
                {hasViews && (
                  <div class="seg">
                    <button class={pickMode === 'photo' ? 'on' : ''} aria-pressed={pickMode === 'photo'} onClick={() => setPickMode('photo')}>写真の上で選ぶ</button>
                    <button class={pickMode === 'plan' ? 'on' : ''} aria-pressed={pickMode === 'plan'} onClick={() => setPickMode('plan')}>平面図で選ぶ</button>
                  </div>
                )}
                {hasViews && pickMode === 'photo' ? (
                  <>
                    <PhotoMeasure room={room} picks={photoPick} onPicks={setPhotoPick} />
                    <p class="muted small">扉の幅・高さのように、端がはっきりした長い物ほど正確に合わせられます（短い物は、タップのずれが大きく効きます）。</p>
                  </>
                ) : (
                  <>
                    <ol class="small">
                      <li>平面図で、長さを測る 2 点（<b class="ptA">A</b> と <b class="ptB">B</b>）をタップします。壁の角の近くは角に吸い付きます。</li>
                      <li>その 2 点の間の実際の長さを入れます（メジャーで測るか、寸法の分かっている物の端から端を選びます）。</li>
                    </ol>
                    <p>選んだ点: {pick.length === 0 ? 'なし' : pick.length === 1 ? 'A のみ（B をタップ）' : <>A・B（いまの図での長さ 約 <b>{fmtCm(dist(pick[0], pick[1]))}</b>）</>} {pick.length > 0 && <button class="link" onClick={() => setPick([])}>選び直す</button>}</p>
                  </>
                )}
                <label class="field"><span>場所の名前（任意）</span><input value={mLabel} onInput={(e) => setMLabel((e.target as HTMLInputElement).value)} placeholder="例: 窓側の壁の幅" maxLength={30} /></label>
                <CmField label="実際の長さ" value={mLen} onChange={setMLen} min={1} />
                <button class="primary block" onClick={saveMeasurement} disabled={(hasViews && pickMode === 'photo' ? photoPick.length : pick.length) !== 2 || !mLen}>{hasCal ? '検証として記録する' : 'この長さで縮尺を合わせる'}</button>
                {hasViews && pickMode === 'photo' && photoPick.length === 2 && mLen ? <p class="small">実際の長さ ÷ 復元での長さ = <b>{(mLen / pickDistance(photoPick[0], photoPick[1])).toFixed(3)}</b>{hasCal ? '（1.000 に近いほど合っています）' : '（この比で部屋全体の大きさを直します）'}</p> : null}
                {!hasCal && <p class="muted small">縮尺合わせで直るのは全体の大きさだけです。部屋のゆがみや一部のずれは直りません。そのため、合わせたあとに別の場所（5 か所以上を推奨）を測って確かめます。</p>}
              </section>
              {!hasCal && room.scale.status === 'model_estimate' && room.ceiling.status === 'observed' && room.ceiling.height != null && (
                <section class="panel-card">
                  <h3>天井高が分かる場合（任意）</h3>
                  <p class="small">物件情報に天井高が書いてあれば、それだけで大きさを合わせられます。写真からの推定は <b>{fmtCm(room.ceiling.height)}</b> です。</p>
                  <CmField label="実際の天井高" value={ceilIn} onChange={setCeilIn} min={150} />
                  <button class="block" disabled={!ceilIn} onClick={() => { if (ceilIn) { tryDo(() => calibrateByCeiling(room, ceilIn)); setCeilIn(null); toast('天井高で大きさを合わせました。'); } }}>この天井高で合わせる</button>
                </section>
              )}
              <section class="panel-card">
                <h3>測定の記録</h3>
                {acc.rows.length === 0 ? <p class="muted">まだありません。</p> : (
                  <table class="tbl">
                    <thead><tr><th>場所</th><th>役割</th><th>実測</th><th>図での長さ</th><th>差</th><th /></tr></thead>
                    <tbody>
                      {acc.rows.map((r) => (
                        <tr>
                          <td>{r.label}<br /><span class="muted small">{r.measuredM >= 1 ? '長い区間' : '短い区間'}</span></td>
                          <td>{r.role === 'calibration' ? '縮尺合わせ' : '検証'}</td>
                          <td>{fmtCmExact(r.measuredM)}</td>
                          <td>{r.role === 'calibration' ? '—' : fmtCm(r.estimatedM)}</td>
                          <td>{r.role === 'calibration' ? '（評価に含めない）' : `${r.errorM >= 0 ? '+' : '−'}${fmtCmExact(Math.abs(r.errorM))}`}</td>
                          <td><button class="link" onClick={() => save(removeMeasurement(room, r.id))} aria-label={`${r.label} を削除`}>削除</button></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {acc.nValidation > 0 && <p>検証 {acc.nValidation} か所: 差の平均 <b>{fmtCmExact(acc.meanAbs!)}</b>、最大 <b>{fmtCmExact(acc.maxAbs!)}</b>{acc.nValidation < 5 ? '（5 か所以上で確かめることをおすすめします）' : ''}</p>}
                <p class="muted small">目安: 壁の幅のような長い区間は差 3cm 以内、柱の出っ張りのような短い区間は差 1cm 以内を目標にしています。長い区間だけ合っていても、数 cm の出っ張りが正しいとは限らないため、短い区間も測ってください。この最大の差が、配置の判定で「要確認」とする幅になります。</p>
              </section>
            </>
          )}

          {tab === 'fixtures' && (
            <>
              <Notice>設備の位置は自分で登録します（写真からの自動認識には対応していません）。種類を選び、平面図の壁をタップしてください。</Notice>
              <div class="btn-row wrap">
                {(Object.keys(FIXTURE_DEFAULTS) as FixtureType[]).map((t) => (
                  <button class={addType === t ? 'on' : ''} aria-pressed={addType === t} onClick={() => setAddType(addType === t ? null : t)}>＋ {FIXTURE_DEFAULTS[t].label}</button>
                ))}
              </div>
              {addType && <p><b>{FIXTURE_DEFAULTS[addType].label}</b>を置く壁をタップしてください。</p>}
              <ul class="list">
                {room.fixtures.map((f) => (
                  <li><button class={fxId === f.id ? 'link on' : 'link'} onClick={() => setFxId(f.id)}>{FIXTURE_DEFAULTS[f.type].label}{f.door?.isEntrance ? '（入口）' : ''} — 壁 {f.edgeIndex + 1}、端から {fmtCm(f.offset)}</button></li>
                ))}
              </ul>
              {room.fixtures.length > 0 && !room.fixtures.some((f) => f.door?.isEntrance) && <Notice kind="warn">入口の扉が決まっていません。通路の判定には入口が必要です。</Notice>}
              {fx && <FixtureForm key={fx.id} room={room} fx={fx} onChange={(f) => tryDo(() => upsertFixture(room, f))} onDelete={() => { void save(removeFixture(room, fx.id)); setFxId(null); }} />}
            </>
          )}
          <a class="button block" href={`#/rooms/${room.id}`}>確認を終えて部屋へ戻る</a>
        </div>
      </div>
    </Page>
  );
}

function ObstacleForm(props: { ob: Obstacle; onChange: (o: Obstacle) => void; onDelete: () => void }) {
  const o = props.ob;
  const cx = (o.rect[0] + o.rect[2]) / 2, cz = (o.rect[1] + o.rect[3]) / 2;
  const size = (w: number | null, d: number | null) => {
    const ww = w ?? o.rect[2] - o.rect[0], dd = d ?? o.rect[3] - o.rect[1];
    props.onChange({ ...o, rect: [cx - ww / 2, cz - dd / 2, cx + ww / 2, cz + dd / 2], confirm: 'manual' });
  };
  return (
    <div class="subform">
      <label class="field"><span>種類</span>
        <select value={o.kind} onChange={(e) => props.onChange({ ...o, kind: (e.target as HTMLSelectElement).value as Obstacle['kind'], confirm: 'manual' })}>
          <option value="pillar">柱</option><option value="beam">梁</option><option value="protrusion">出っ張り</option><option value="unknown_vertical">種類未確認</option>
        </select>
      </label>
      <div class="row2">
        <CmField label="横幅（図の左右）" value={o.rect[2] - o.rect[0]} onChange={(w) => w && size(w, null)} min={1} />
        <CmField label="縦幅（図の上下）" value={o.rect[3] - o.rect[1]} onChange={(d) => d && size(null, d)} min={1} />
      </div>
      <div class="row2">
        <CmField label="床から下端" value={o.yMin} onChange={(y) => y != null && props.onChange({ ...o, yMin: y, confirm: 'manual' })} />
        <CmField label="床から上端" value={o.yMax} onChange={(y) => y != null && y > o.yMin && props.onChange({ ...o, yMax: y, confirm: 'manual' })} />
      </div>
      {o.confirm === 'auto' && <Notice kind="warn">自動で見つけた縦の構造物です。背の高い家具やカーテンの可能性もあります。実物を確かめ、寸法を直すか、違っていれば削除してください。</Notice>}
      <div class="btn-row">
        {o.confirm === 'auto' && <button class="primary" onClick={() => props.onChange({ ...o, confirm: 'confirmed' })}>この大きさで合っている</button>}
        <button class="danger" onClick={props.onDelete}>削除</button>
      </div>
    </div>
  );
}

function FixtureForm(props: { room: Room; fx: Fixture; onChange: (f: Fixture) => void; onDelete: () => void }) {
  const f = props.fx;
  const set = (patch: Partial<Fixture>) => props.onChange({ ...f, ...patch });
  return (
    <section class="panel-card">
      <h3>{FIXTURE_DEFAULTS[f.type].label} <span class="muted">（壁 {f.edgeIndex + 1}・長さ {fmtCm(edgeLength(props.room, f.edgeIndex))}）</span></h3>
      <CmField label={`壁 ${f.edgeIndex + 1} の始点から中心まで`} value={f.offset} onChange={(x) => x != null && set({ offset: x })} hint="平面図で、壁の番号が付いた辺を反時計回りにたどった始点からの距離です。" />
      <div class="row3">
        <CmField label="幅" value={f.width} onChange={(x) => x && set({ width: x })} min={1} />
        <CmField label="高さ" value={f.height} onChange={(x) => x && set({ height: x })} min={1} />
        <CmField label="床から下端" value={f.bottom} onChange={(x) => x != null && set({ bottom: x })} />
      </div>
      {f.type === 'door' && f.door && (
        <>
          <div class="row2">
            <label class="field"><span>開き方</span>
              <select value={f.door.opensInto} onChange={(e) => set({ door: { ...f.door!, opensInto: (e.target as HTMLSelectElement).value as 'room' | 'outside' | 'sliding' } })}>
                <option value="room">この部屋の中へ開く</option><option value="outside">部屋の外へ開く</option><option value="sliding">引き戸</option>
              </select>
            </label>
            <label class="field"><span>蝶番の位置</span>
              <select value={f.door.hinge} onChange={(e) => set({ door: { ...f.door!, hinge: (e.target as HTMLSelectElement).value as 'start' | 'end' } })} disabled={f.door.opensInto !== 'room'}>
                <option value="start">壁の始点側</option><option value="end">壁の終点側</option>
              </select>
            </label>
          </div>
          <label class="check"><input type="checkbox" checked={f.door.isEntrance} onChange={(e) => set({ door: { ...f.door!, isEntrance: (e.target as HTMLInputElement).checked } })} /> この扉を部屋の入口にする（通路の判定の出発点）</label>
          <p class="muted small">平面図の扇形が、扉の開く範囲です。向きが違うときは蝶番の位置を切り替えてください。</p>
        </>
      )}
      {f.type === 'aircon' && <p class="muted small">エアコンのまわりに必要な空間はメーカーの条件によります。判定では「要確認」として知らせるだけです。</p>}
      <button class="danger" onClick={props.onDelete}>削除</button>
    </section>
  );
}
