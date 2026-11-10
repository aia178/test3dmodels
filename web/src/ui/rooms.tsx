import { useEffect, useRef, useState } from 'preact/hooks';
import type { Room } from '../core/types';
import { applyJobResult, floorArea, newRoomBase, setManualRect, surfacesUsable } from '../core/room';
import { qualityReason, roomTolerance } from '../core/judge';
import * as db from '../store/db';
import * as api from '../store/api';
import { CmField, Empty, Notice, Page, TabBar, fmtDate, navigate, toast, useLive } from './common';
import { Plan2D } from '../view/Plan2D';
import { Scene3D } from '../view/Scene3D';
import { useCloud, useSurfaces } from './cloud';

const NO_TARGETS = new Map<string, 'problem' | 'needs_check'>();
const noop = () => {};

const KIND_LABEL = { current: '今の部屋', viewing: '内見した部屋' } as const;

async function roomState(r: Room): Promise<{ label: string; href: string; tone: 'todo' | 'busy' | 'ok' }> {
  if (r.outline.length === 0) {
    const cap = await db.get('captures', r.id);
    if (cap?.jobId) return { label: '写真から復元しています', href: `/rooms/${r.id}/job`, tone: 'busy' };
    return { label: `写真の登録が途中です（${cap?.photos.length ?? 0} 枚）`, href: `/rooms/${r.id}/capture`, tone: 'todo' };
  }
  const unobs = r.edges.filter((e) => e.status === 'unobserved').length;
  if (unobs) return { label: `形の分からない壁が ${unobs} か所`, href: `/rooms/${r.id}`, tone: 'todo' };
  return { label: r.scale.status === 'model_estimate' ? '配置できます（大きさは推定）' : '配置できます', href: `/rooms/${r.id}`, tone: 'ok' };
}

export function Home() {
  const { data } = useLive(async () => {
    const rooms = (await db.getAll('rooms')).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    return Promise.all(rooms.map(async (r) => ({ r, st: await roomState(r), layouts: (await db.layoutsOf(r.id)).length })));
  });
  const nFurn = useLive(async () => (await db.getAll('furniture')).length).data;
  return (
    <>
      <Page title="RoomFit">
        <p class="lead">部屋を写真から立体にして、手持ちの家具や買いたい家具が「置けるか・使えるか」を確かめます。</p>
        <div class="start-grid">
          <a class="start" href="#/rooms/new?kind=current"><b>今の部屋を登録</b><span>模様替え・家具の購入前に</span></a>
          <a class="start" href="#/rooms/new?kind=viewing"><b>内見した部屋を登録</b><span>手持ちの家具が入るか確かめる</span></a>
          <a class="start" href="#/furniture/new"><b>家具を登録</b><span>{nFurn ? `登録済み ${nFurn} 点` : '寸法を入れて何度でも使える'}</span></a>
        </div>
        <h2>登録した部屋</h2>
        <RoomFileImport />
        {data === undefined ? <p class="muted">読み込み中…</p> : data.length === 0 ? (
          <Empty title="まだ部屋がありません"><p>上の「今の部屋を登録」か「内見した部屋を登録」から始めてください。登録や保存にアカウントは要りません（この端末に保存されます）。</p></Empty>
        ) : (
          <ul class="cards">
            {data.map(({ r, st, layouts }) => (
              <li>
                <a class="card" href={`#${st.href}`}>
                  <div class="card-main">
                    <b>{r.name}</b>
                    <span class="tag">{KIND_LABEL[r.kind]}</span>
                    {r.purpose && <span class="tag">{r.purpose}</span>}
                    <span class="tag">{r.source === 'photo' ? '写真から作成' : r.outline.length ? '寸法入力' : '作成中'}</span>
                  </div>
                  <div class={`card-state ${st.tone}`}>{st.tone === 'ok' ? '✓' : st.tone === 'busy' ? '…' : '！'} {st.label}</div>
                  <div class="muted small">{r.outline.length ? `約 ${floorArea(r).toFixed(1)}㎡ ・ 配置案 ${layouts} 件 ・ ` : ''}更新 {fmtDate(r.updatedAt)}</div>
                </a>
              </li>
            ))}
          </ul>
        )}
      </Page>
      <TabBar current="/" />
    </>
  );
}

function RoomFileImport() {
  const ref = useRef<HTMLInputElement>(null);
  const [err, setErr] = useState<string | null>(null);
  const load = async (file: File | undefined) => {
    if (!file) return;
    setErr(null);
    try {
      const room = await db.importRoomFile(await file.text());
      toast(`「${room.name}」を読み込みました`);
      navigate(`/rooms/${room.id}`);
    } catch (e) { setErr((e as Error).message); }
  };
  return (
    <div class="import-row">
      <button class="link" onClick={() => ref.current?.click()}>部屋ファイルを読み込む</button>
      <span class="muted small">（別の端末で書き出した部屋や、用意しておいた部屋）</span>
      <input ref={ref} type="file" accept=".json,application/json" hidden onChange={(e) => { void load((e.target as HTMLInputElement).files?.[0]); (e.target as HTMLInputElement).value = ''; }} />
      {err && <Notice kind="error">{err}</Notice>}
    </div>
  );
}

export function RoomNew(props: { query: URLSearchParams }) {
  const [name, setName] = useState('');
  const [purpose, setPurpose] = useState('');
  const [kind, setKind] = useState<Room['kind']>(props.query.get('kind') === 'viewing' ? 'viewing' : 'current');
  const [method, setMethod] = useState<'camera' | 'photos' | 'manual'>('camera');
  const [w, setW] = useState<number | null>(null), [d, setD] = useState<number | null>(null), [h, setH] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const submit = async () => {
    if (!name.trim()) { setErr('部屋の名前を入れてください（例: 今の寝室、A物件の洋室）。'); return; }
    let room = newRoomBase(db.uid(), name.trim(), purpose, kind);
    if (method === 'manual') {
      if (!w || !d) { setErr('幅と奥行きを入れてください。'); return; }
      room = setManualRect(room, w, d, h);
      await db.put('rooms', room);
      navigate(`/rooms/${room.id}`);
    } else {
      await db.put('rooms', room);
      await db.put('captures', { roomId: room.id, photos: [] });
      navigate(`/rooms/${room.id}/capture${method === 'camera' ? '?camera=1' : ''}`);
    }
  };
  return (
    <Page title="部屋を登録" back="/">
      <label class="field"><span>部屋の名前</span><input value={name} onInput={(e) => setName((e.target as HTMLInputElement).value)} placeholder="例: A物件の洋室" maxLength={40} /></label>
      <div class="row2">
        <label class="field"><span>種類</span>
          <select value={kind} onChange={(e) => setKind((e.target as HTMLSelectElement).value as Room['kind'])}><option value="current">今の部屋</option><option value="viewing">内見した部屋</option></select>
        </label>
        <label class="field"><span>用途（任意）</span>
          <select value={purpose} onChange={(e) => setPurpose((e.target as HTMLSelectElement).value)}>
            <option value="">選ばない</option><option>ワンルーム</option><option>寝室</option><option>リビング</option><option>書斎</option><option>その他</option>
          </select>
        </label>
      </div>
      <p class="muted small">住所は不要です。</p>
      <fieldset class="choices">
        <legend>作り方</legend>
        <label class={method === 'camera' ? 'choice on' : 'choice'}><input type="radio" name="m" checked={method === 'camera'} onChange={() => setMethod('camera')} /><div><b>撮影から作成</b><span>スマホのカメラで部屋を撮り、写真から立体にします。</span></div></label>
        <label class={method === 'photos' ? 'choice on' : 'choice'}><input type="radio" name="m" checked={method === 'photos'} onChange={() => setMethod('photos')} /><div><b>撮影済みの写真から作成</b><span>内見で撮った写真などを選びます。</span></div></label>
        <label class={method === 'manual' ? 'choice on' : 'choice'}><input type="radio" name="m" checked={method === 'manual'} onChange={() => setMethod('manual')} /><div><b>寸法を入力して作成</b><span>撮影できないときの方法です。長方形の部屋になり、柱や出っ張りは後から自分で足します。</span></div></label>
      </fieldset>
      {method === 'manual' && (
        <div class="row3">
          <CmField label="幅" value={w} onChange={setW} /><CmField label="奥行き" value={d} onChange={setD} /><CmField label="天井高（任意）" value={h} onChange={setH} />
        </div>
      )}
      {err && <Notice kind="error">{err}</Notice>}
      <button class="primary block" onClick={submit}>{method === 'manual' ? 'この寸法で作成' : '写真の登録へ進む'}</button>
    </Page>
  );
}

// 案内つき撮影の手順。実測との比較（docs/検証記録.md §9・§11・§12）で効いた条件をそのまま手順にした:
// 壁から 1.5〜2m、物の全体が入る、隣の写真と半分以上重なる、同じ場所を 2 か所の立ち位置から。
const GUIDE_STEPS: { where: string; how: string }[] = [
  { where: '壁から 1.5〜2m 離れて立ち、角の左側の壁を撮ります。', how: '扉や収納があれば、その全体が入るように。床と壁の境目も入れます。スマホは縦向きで、ズームは使いません。' },
  { where: '同じ場所に立ったまま、少し右を向いて撮ります。', how: 'いま撮った写真と、半分くらい重なるようにします。' },
  { where: 'さらに右を向いて、角と、角の右側の壁を撮ります。', how: 'これも、前の写真と半分くらい重なるように。' },
  { where: '右へ 1〜2 歩動きます。そこから、最初と同じ左側の壁を撮ります。', how: '同じ物を別の位置から撮ると、奥行きが正しく出ます。壁からの距離（1.5〜2m）は保ちます。' },
  { where: '同じ場所に立ったまま、少し右を向いて撮ります。', how: '前の写真と半分くらい重なるように。' },
  { where: 'さらに右を向いて、角と右側の壁を撮ります。', how: 'これで、この角は 2 か所の立ち位置から撮れました。' },
];

// ---- 写真の簡易チェック（明るさ・ぼけ）。重なりや撮影範囲は端末側では測れないので、案内文で補う。
async function inspectPhoto(blob: Blob): Promise<string[]> {
  try {
    const bmp = await createImageBitmap(blob);
    const w = 160, h = Math.max(1, Math.round((bmp.height / bmp.width) * w));
    const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
    const ctx = cv.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(bmp, 0, 0, w, h);
    const px = ctx.getImageData(0, 0, w, h).data;
    const g = new Float32Array(w * h);
    let sum = 0;
    for (let i = 0; i < w * h; i++) { g[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2]; sum += g[i]; }
    const mean = sum / (w * h);
    let lap = 0, n = 0;
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x; const v2 = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w]; lap += v2 * v2; n++;
    }
    const warn: string[] = [];
    if (mean < 45) warn.push('暗い');
    if (mean > 235) warn.push('白とびしている');
    if (lap / n < 12) warn.push('ぶれているか、模様の少ない面だけが写っている');
    if (Math.min(bmp.width, bmp.height) < 480) warn.push('画像が小さい');
    bmp.close();
    return warn;
  } catch { return ['画像として開けない']; }
}

function Thumb(props: { blob: Blob }) {
  const [url, setUrl] = useState('');
  useEffect(() => { const u = URL.createObjectURL(props.blob); setUrl(u); return () => URL.revokeObjectURL(u); }, [props.blob]);
  return url ? <img src={url} alt="" loading="lazy" /> : null;
}

export function Capture(props: { roomId: string; query: URLSearchParams }) {
  const { data: room } = useLive(() => db.get('rooms', props.roomId), [props.roomId]);
  const { data: cap, reload } = useLive(async () => (await db.get('captures', props.roomId)) ?? { roomId: props.roomId, photos: [] }, [props.roomId]);
  const { data: h, error: hErr } = useLive(() => api.health(), []);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<{ message: string; files?: string[] } | null>(null);
  const [agree, setAgree] = useState(false);
  const camRef = useRef<HTMLInputElement>(null), fileRef = useRef<HTMLInputElement>(null);
  const [guide, setGuide] = useState<number | null>(null); // 案内つき撮影の、いまの手順（0 始まり）。null は案内なし
  const guided = useRef(false);
  if (room === undefined || cap === undefined) return <Page title="写真の登録" back="/"><p class="muted">読み込み中…</p></Page>;
  if (!room) return <Page title="写真の登録" back="/"><Notice kind="error">この部屋は見つかりません。</Notice></Page>;

  const addFiles = async (list: FileList | null) => {
    if (!list?.length) return;
    setBusy(true);
    const photos = [...cap.photos];
    for (const f of Array.from(list)) {
      if (!f.type.startsWith('image/')) continue;
      photos.push({ id: db.uid(), name: f.name || 'photo.jpg', blob: f, warn: await inspectPhoto(f) });
    }
    await db.put('captures', { ...cap, photos }); // 途中で閉じても残る
    setBusy(false); reload();
    if (guided.current) { guided.current = false; setGuide((g) => (g == null ? g : g + 1)); }
  };
  const remove = async (id: string) => { await db.put('captures', { ...cap, photos: cap.photos.filter((p) => p.id !== id) }); reload(); };
  const min = h?.min_photos ?? 3, max = h?.max_photos ?? 40;
  const n = cap.photos.length;
  const send = async () => {
    setErr(null); setBusy(true);
    try {
      const job = await api.createJob(cap.photos.map((p) => ({ name: p.name, blob: p.blob })));
      await db.put('captures', { ...cap, jobId: job.id, jobStartedAt: db.now() });
      navigate(`/rooms/${room.id}/job`);
    } catch (e) {
      const a = e as api.ApiError;
      setErr({ message: a.message, files: a.files });
    } finally { setBusy(false); }
  };

  return (
    <Page title={`${room.name} — 写真の登録`} back="/">
      <details class="guide" open={n === 0}>
        <summary>撮り方のコツ</summary>
        <ol>
          <li><b>置きたい場所の壁を、2 か所の立ち位置から</b>撮ります（1 枚撮って、1〜2 歩横へ動いてもう 1 枚）。<b>隣り合う写真が半分以上重なる</b>ように、少しずつ向きを変えます。1 つの角で 6〜8 枚が目安です。部屋全体なら、角ごとに同じことを繰り返します。</li>
          <li><b>壁から 1.5〜2m 離れ、扉や収納の全体が 1 枚に入るように</b>撮ります。近づいて一部分ずつ撮った写真ばかりだと、形と大きさが 1 割前後ずれました（実測との比較で確認）。</li>
          <li>毎回、<b>床と壁の境目</b>が写るようにします。天井高を知りたいときは天井も入れます。</li>
          <li>柱・壁の出っ張り・窓・扉・コンセントは、<b>少し離れた位置から</b>追加で撮ります。近すぎる写真だけだと位置を決められません。</li>
          <li>明るくして、スマホを止めてから撮ります。スマホの向き（縦・横）は揃えてください。</li>
          <li>鏡・大きなガラス・真っ白な壁だけの写真は、形が崩れる原因になります。人や個人情報が写らないようにしてください。</li>
        </ol>
      </details>

      <section class="panel-card guide-card">
        {guide == null ? (
          <>
            <h3>案内に沿って撮る（おすすめ）</h3>
            <p class="small">1 つの角を、2 か所の立ち位置から 6 枚で撮ります。1 枚ずつ、どこを向くかを案内します。</p>
            <button class="primary" onClick={() => setGuide(0)} disabled={busy || n >= max}>案内を始める</button>
          </>
        ) : guide < GUIDE_STEPS.length ? (
          <>
            <h3>{guide + 1} 枚目 <span class="muted">／ {GUIDE_STEPS.length} 枚</span></h3>
            <p><b>{GUIDE_STEPS[guide].where}</b></p>
            <p class="small">{GUIDE_STEPS[guide].how}</p>
            <div class="btn-row">
              <button class="primary" onClick={() => { guided.current = true; camRef.current?.click(); }} disabled={busy || n >= max}>この向きで撮る</button>
              {guide > 0 && <button onClick={() => setGuide(guide - 1)}>1 つ戻る</button>}
              <button onClick={() => setGuide(null)}>案内をやめる</button>
            </div>
          </>
        ) : (
          <>
            <h3>✓ この角は撮れました</h3>
            <p class="small">ほかの角も撮る場合は、その角の前へ移動して続けてください。終わりなら、下の「立体にする」へ進みます。</p>
            <div class="btn-row">
              <button onClick={() => setGuide(0)} disabled={n >= max}>次の角を撮る</button>
              <button onClick={() => setGuide(null)}>案内を終える</button>
            </div>
          </>
        )}
      </section>
      <div class="btn-row">
        <button class={guide == null ? 'primary' : ''} onClick={() => camRef.current?.click()} disabled={busy || n >= max}>カメラで撮る</button>
        <button onClick={() => fileRef.current?.click()} disabled={busy || n >= max}>写真を選ぶ</button>
        <input ref={camRef} type="file" accept="image/*" capture="environment" hidden onChange={(e) => { addFiles((e.target as HTMLInputElement).files); (e.target as HTMLInputElement).value = ''; }} />
        <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={(e) => { addFiles((e.target as HTMLInputElement).files); (e.target as HTMLInputElement).value = ''; }} />
      </div>
      <p class="muted small">「カメラで撮る」はスマホでカメラが開きます（PC ではファイル選択になります）。1 枚撮るごとに一覧へ追加されます。</p>

      <h2>登録した写真 <span class="count">{n} 枚</span></h2>
      {n === 0 ? <Empty title="まだ写真がありません" /> : (
        <ul class="thumbs">
          {cap.photos.map((p, i) => (
            <li class={p.warn.length ? 'warn' : ''}>
              <Thumb blob={p.blob} />
              <span class="thumb-no">{i + 1}</span>
              <button class="thumb-x" onClick={() => remove(p.id)} aria-label={`${i + 1} 枚目を外す`}>外す</button>
              {p.warn.length > 0 && <span class="thumb-warn">！{p.warn.join('・')}</span>}
            </li>
          ))}
        </ul>
      )}
      {n > 0 && n < 4 && <Notice kind="warn">{n < min ? `あと ${min - n} 枚以上必要です。` : ''}3 枚以下では形と大きさが崩れやすくなります（実測との比較で確認）。4 枚以上をおすすめします。</Notice>}
      {n > 16 && n <= max && <Notice kind="warn">{n} 枚あります。17 枚以上は、撮った順に {Math.max(2, Math.ceil((n - 3) / 10))} つの範囲に分けて復元し、あとでつなぎます（目安: 1 範囲 約 8 分、合計 約 {Math.max(2, Math.ceil((n - 3) / 10)) * 8} 分）。<b>写真が撮った順に並んでいること</b>、隣り合う写真が大きく重なっていることが前提です。処理中は PC が重くなります。</Notice>}
      {n > max && <Notice kind="error">一度に処理できるのは {max} 枚までです。似た写真を外してください。</Notice>}
      {cap.photos.some((p) => p.warn.length) && <Notice kind="warn">「！」の付いた写真は、外すか撮り直すと結果が良くなる可能性があります（そのまま使うこともできます）。</Notice>}

      <h2>立体にする</h2>
      {hErr && <Notice kind="error">{hErr}</Notice>}
      {h && !h.recon_available && <Notice kind="error">このサーバーでは写真からの復元を実行できません。{h.recon_unavailable_reason}</Notice>}
      {h && (
        <label class="agree">
          <input type="checkbox" checked={agree} onChange={(e) => setAgree((e.target as HTMLInputElement).checked)} />
          <span><b>写真の送り先を確認しました。</b> {h.photo_destination} 位置情報などの付帯情報は処理の前に取り除きます。写真と結果は「設定」からいつでも削除できます。</span>
        </label>
      )}
      {err && <Notice kind="error">{err.message}{err.files && <ul>{err.files.map((f) => <li>{f}</li>)}</ul>}</Notice>}
      <button class="primary block" disabled={busy || !agree || n < min || n > max || !h?.recon_available} onClick={send}>
        {busy ? '送信中…' : `${n} 枚の写真から部屋を立体にする`}
      </button>
      <p class="muted small">このPCでの実測では、4 枚で約 3 分、13 枚で約 8 分かかりました（GPU なし）。17 枚以上は範囲に分けるので、10 枚増えるごとに約 8 分増えます。待っている間も他の画面を使えます。</p>
    </Page>
  );
}

const STAGE_LABEL: Record<string, string> = {
  preprocessing: '写真の向きと大きさを整えています',
  loading_model: '復元の準備をしています（初回は数分かかります）',
  inferring: '写真から奥行きと撮影位置を推定しています（いちばん時間がかかる段階です）',
  stitching: '範囲ごとの結果をつないでいます',
  structuring: '床・壁・柱の形を取り出しています',
  done: '完了しました',
};

export function JobView(props: { roomId: string }) {
  const { data: room } = useLive(() => db.get('rooms', props.roomId), [props.roomId]);
  const { data: cap } = useLive(() => db.get('captures', props.roomId), [props.roomId]);
  const [job, setJob] = useState<api.JobInfo | null>(null);
  const [netErr, setNetErr] = useState<string | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [nowT, setNowT] = useState(Date.now());
  const [nonce, setNonce] = useState(0);
  const importing = useRef(false);

  useEffect(() => { const t = setInterval(() => setNowT(Date.now()), 1000); return () => clearInterval(t); }, []);
  useEffect(() => {
    if (!cap?.jobId || !room) return;
    let alive = true, timer = 0;
    const poll = async () => {
      try {
        const j = await api.getJob(cap.jobId!);
        if (!alive) return;
        setJob(j); setNetErr(null);
        if (j.state === 'done' && !importing.current) {
          importing.current = true;
          const [jr, cloud] = await Promise.all([api.getJobRoom(j.id), api.getJobCloud(j.id)]);
          const updated = applyJobResult(room, j.id, jr, db.uid);
          await db.put('blobs', { id: `cloud:${room.id}`, blob: cloud });
          for (const v of jr.views ?? []) { // 写真の上で測るためのデータ
            await db.put('blobs', { id: `view:${room.id}:${v.n}:jpg`, blob: await api.getJobViewImage(j.id, v.n) });
            await db.put('blobs', { id: `view:${room.id}:${v.n}:xyz`, blob: await api.getJobViewXyz(j.id, v.n) });
          }
          for (const f of db.surfaceFiles(updated)) { // 壁・床に写真を貼った画像（見た目のため）
            await db.put('blobs', { id: `surface:${room.id}:${f}`, blob: await api.getJobSurface(j.id, f) });
          }
          await db.put('rooms', updated);
          await db.del('captures', room.id); // 端末側の送信用コピーは不要になる
          toast('部屋を立体にしました。形と縮尺を確認してください。');
          navigate(`/rooms/${room.id}/check`);
          return;
        }
        if (j.state === 'queued' || j.state === 'running') timer = window.setTimeout(poll, 2500);
      } catch (e) {
        if (!alive) return;
        const a = e as api.ApiError;
        if (a.kind === 'offline') { setNetErr(a.message); timer = window.setTimeout(poll, 5000); } // 通信が戻れば続きから
        else setFatal(a.message);
      }
    };
    poll();
    return () => { alive = false; clearTimeout(timer); };
  }, [cap?.jobId, room?.id, nonce]);

  if (room === undefined || cap === undefined) return <Page title="復元" back="/"><p class="muted">読み込み中…</p></Page>;
  if (!room) return <Page title="復元" back="/"><Notice kind="error">この部屋は見つかりません。</Notice></Page>;
  if (!cap?.jobId) return <Page title="復元" back="/"><Notice>進行中の処理はありません。</Notice><a class="button" href={`#/rooms/${room.id}${room.outline.length ? '' : '/capture'}`}>部屋を開く</a></Page>;

  const backToPhotos = async () => { await db.put('captures', { ...cap, jobId: undefined, jobStartedAt: undefined }); navigate(`/rooms/${room.id}/capture`); };
  const elapsed = job?.started_at ? Math.max(0, Math.floor(((job.finished_at ? job.finished_at * 1000 : nowT) - job.started_at * 1000) / 1000)) : 0;
  const order = ['preprocessing', 'loading_model', 'inferring', ...(job?.chunks ? ['stitching'] : []), 'structuring'];
  const cur = job?.stage ? order.indexOf(job.stage) : -1;

  return (
    <Page title={`${room.name} — 立体にしています`} back="/">
      {netErr && <Notice kind="warn">{netErr} 処理はサーバー側で続いています。つながり次第、自動で表示を更新します。</Notice>}
      {fatal && <Notice kind="error">{fatal}<div class="btn-row"><button onClick={backToPhotos}>写真の登録へ戻る</button></div></Notice>}
      {!job && !fatal && !netErr && <p class="muted">状態を確認しています…</p>}
      {job && (job.state === 'queued' || job.state === 'running') && (
        <>
          {job.state === 'queued' && <Notice>順番を待っています{job.queue_ahead ? `（前に ${job.queue_ahead} 件）` : ''}。</Notice>}
          <ol class="stages">
            {order.map((s, i) => (
              <li class={i < cur ? 'done' : i === cur ? 'now' : ''}>
                <span class="stage-mark" aria-hidden="true">{i < cur ? '✓' : i === cur ? '●' : '○'}</span>{STAGE_LABEL[s]}{s === 'inferring' && i === cur && job.chunks && job.chunk ? `（${job.chunks} 範囲中 ${job.chunk} 番目）` : ''}{i === cur && <span class="spinner" aria-label="処理中" />}
              </li>
            ))}
          </ol>
          <p class="muted">写真 {job.n_photos} 枚 ・ 開始から {Math.floor(elapsed / 60)} 分 {elapsed % 60} 秒。残り時間は正確に分からないため表示していません。</p>
          <p class="muted small">この画面を閉じても処理は続きます。ホームの部屋一覧から戻れます。</p>
          <button onClick={async () => { await api.cancelJob(job.id).catch(() => {}); }}>処理を中止する</button>
        </>
      )}
      {job && (job.state === 'failed' || job.state === 'cancelled' || job.state === 'interrupted') && (
        <>
          <Notice kind={job.state === 'failed' ? 'error' : 'warn'}>
            <b>{job.state === 'failed' ? '立体にできませんでした' : job.state === 'cancelled' ? '処理を中止しました' : 'サーバーの再起動で処理が中断されました'}</b>
            {job.error_message && <p>{job.error_message}</p>}
          </Notice>
          <div class="btn-row">
            <button class="primary" onClick={async () => { setJob(null); await api.retryJob(job.id).catch((e) => setFatal((e as Error).message)); setNonce((x) => x + 1); }}>同じ写真でもう一度</button>
            <button onClick={backToPhotos}>写真を足す・入れ替える</button>
          </div>
          <p class="muted small">うまくいかない場合は、ホームから「寸法を入力して作成」で部屋を作ることもできます（その場合、柱や出っ張りは自分で足します）。</p>
          {job.error_detail && <details class="dev"><summary>開発者向けの詳細</summary><code>{job.error_code}: {job.error_detail}</code></details>}
        </>
      )}
    </Page>
  );
}

export function RoomHub(props: { roomId: string }) {
  const { data: room } = useLive(() => db.get('rooms', props.roomId), [props.roomId]);
  const { data: layouts } = useLive(() => db.layoutsOf(props.roomId), [props.roomId]);
  const { data: cap } = useLive(async () => (await db.get('captures', props.roomId)) ?? null, [props.roomId]);
  const [view, setView] = useState<'2d' | '3d'>('2d');
  const [look, setLook] = useState<'photo' | 'points'>('photo');
  const surfaces = useSurfaces(room ?? undefined, view === '3d');
  const photoLook = !!surfaces && look === 'photo';
  const cloud = useCloud(props.roomId, view === '3d' && !!room?.hasPointCloud && !photoLook);
  if (room === undefined) return <Page title="部屋" back="/"><p class="muted">読み込み中…</p></Page>;
  if (!room) return <Page title="部屋" back="/"><Notice kind="error">この部屋は見つかりません。</Notice></Page>;
  if (!room.outline.length) { navigate(`/rooms/${room.id}/capture`); return null; }
  const tol = roomTolerance(room);
  const unobs = room.edges.filter((e) => e.status === 'unobserved').length;
  const auto = room.edges.filter((e) => e.status === 'observed').length;
  const newLayout = async () => {
    const l = { id: db.uid(), roomId: room.id, name: `配置案 ${(layouts?.length ?? 0) + 1}`, items: [], updatedAt: db.now() };
    await db.put('layouts', l); navigate(`/rooms/${room.id}/layout/${l.id}`);
  };
  const removeRoom = async () => {
    if (!confirm(`「${room.name}」と、その配置案・復元データを削除します。元に戻せません。`)) return;
    if (room.job && room.job.id !== 'file') await api.deleteJob(room.job.id).catch(() => {});
    await db.deleteRoom(room.id); navigate('/');
  };
  return (
    <Page title={room.name} back="/">
      <div class="seg hub-seg">
        <button class={view === '2d' ? 'on' : ''} aria-pressed={view === '2d'} onClick={() => setView('2d')}>平面図</button>
        <button class={view === '3d' ? 'on' : ''} aria-pressed={view === '3d'} onClick={() => setView('3d')}>3D で見る</button>
      </div>
      <div class={view === '3d' ? 'hub-plan tall' : 'hub-plan'}>
        {view === '2d'
          ? <Plan2D room={room} showLengths />
          : <Scene3D room={room} items={[]} selectedId={null} issue={null} issueTargets={NO_TARGETS} cloud={photoLook ? null : cloud?.pos} cloudColors={photoLook ? null : cloud?.col} showCloud={room.hasPointCloud && !photoLook} surfaces={photoLook ? surfaces : null} onSelect={noop} onMove={noop} />}
      </div>
      {view === '3d' && !surfaces && room.surfaces && surfacesUsable(room) && <p class="muted small">壁と床の写真を読み込んでいます…（数秒かかります）</p>}
      {view === '3d' && surfaces && (
        <div class="seg hub-seg small-seg">
          <button class={look === 'photo' ? 'on' : ''} aria-pressed={look === 'photo'} onClick={() => setLook('photo')}>写真で表示</button>
          <button class={look === 'points' ? 'on' : ''} aria-pressed={look === 'points'} onClick={() => setLook('points')}>復元した点で表示</button>
        </div>
      )}
      {view === '3d' && <p class="muted small">ドラッグで回転、ホイール・ピンチで拡大縮小。{photoLook
        ? '壁と床に、撮った写真をそのまま貼っています（見た目のため。寸法は復元した形のまま）。灰色の所は、どの写真にも写っていない場所です。'
        : room.hasPointCloud ? '色のついた点は、写真から復元した 3D の点です。点が無い所は、どの写真にも写っていない場所です。' : 'この部屋は寸法入力で作ったため、写真から復元した点はありません。'}</p>}
      <p class="muted small">約 {floorArea(room).toFixed(1)}㎡ ・ 天井高 {room.ceiling.height ? `${Math.round(room.ceiling.height * 100)}cm${room.ceiling.status === 'observed' ? '（自動・未確認）' : ''}` : '未確認'} ・ {room.source === 'photo' ? `写真 ${room.job?.nPhotos ?? '?'} 枚から作成` : '寸法入力で作成'}</p>
      {cap && (cap.jobId
        ? <Notice>撮り直した写真から、形を作り直しています。<a href={`#/rooms/${room.id}/job`}>進み具合を見る</a></Notice>
        : <Notice>撮り直しの途中です（写真 {cap.photos.length} 枚）。<a href={`#/rooms/${room.id}/capture`}>写真の登録を続ける</a>　<button class="link" onClick={() => db.del('captures', room.id)}>撮り直しをやめる</button></Notice>)}
      {room.captureQuality?.level === 'poor' && <Notice kind="warn"><b>この復元は信頼しにくい状態です。</b> {qualityReason(room.captureQuality.reasons)} 測りたい物の全体が入る距離（1.5〜2m）から、2 か所の立ち位置で撮り直すことをおすすめします。</Notice>}
      {room.captureQuality?.level === 'good' && <Notice>写真どうしの食い違いは小さく、復元は安定しています（{room.captureQuality.nPhotos} 枚・食い違い 約 {room.captureQuality.disagreementMm}mm）。これは形の安定さの目安で、大きさの正しさの保証ではありません。</Notice>}
      {(room.captureQuality?.chunks ?? 0) > 1 && <Notice kind="warn">写真が多いため、<b>{room.captureQuality!.chunks} つの範囲に分けて復元し、つないだ</b>結果です。ひとつの範囲の中の寸法に比べて、つなぎ目をまたぐ寸法（部屋の端から端など）は当てにならず、実測のある公開データでは 1 割前後ずれることがありました。壁から壁の寸法が大事なときは、「部屋の確認」で長い辺を 1 か所測って確かめてください。</Notice>}
      {room.scale.status === 'model_estimate' && <Notice>部屋の大きさは<b>写真からの推定</b>です。このまま家具を置いて試せます。数 cm が効く配置を確かめたいときだけ、「部屋の確認」で寸法の分かる場所を 1 か所入れると確実になります（任意）。</Notice>}
      {unobs > 0 && <Notice kind="warn">形の分からない壁が {unobs} か所あります（平面図の破線）。追加で撮影するか、形を確認してください。</Notice>}
      {auto > 0 && <Notice>自動で復元した壁が {auto} 枚あります。実物と合っているか確認すると、判定の「要確認」が減ります。</Notice>}
      <Notice>{tol.basis}</Notice>
      <div class="btn-row">
        <a class="button primary" href={`#/rooms/${room.id}/check`}>部屋の確認（形・縮尺・設備）</a>
      </div>
      <h2>配置案</h2>
      {layouts && layouts.length === 0 && <Empty title="まだ配置案がありません"><p>家具を置いて、問題がないか確かめます。</p></Empty>}
      <ul class="cards">
        {(layouts ?? []).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)).map((l) => (
          <li><a class="card" href={`#/rooms/${room.id}/layout/${l.id}`}><div class="card-main"><b>{l.name}</b></div><div class="muted small">家具 {l.items.length} 点 ・ 更新 {fmtDate(l.updatedAt)}</div></a></li>
        ))}
      </ul>
      <button class="primary" onClick={newLayout}>新しい配置案を作る</button>
      <h2>この部屋のデータ</h2>
      <div class="btn-row">
        {room.source === 'photo' && <button onClick={async () => { await db.put('captures', { roomId: room.id, photos: [] }); navigate(`/rooms/${room.id}/capture`); }}>撮り直す（形を作り直す）</button>}
        <button onClick={async () => {
          const f = await db.exportRoomFile(room.id);
          const a = document.createElement('a');
          a.href = URL.createObjectURL(new Blob([f.text], { type: 'application/json' }));
          a.download = `${f.name}.roomfit.json`; a.click(); URL.revokeObjectURL(a.href);
        }}>部屋ファイルに書き出す</button>
        <button class="danger" onClick={removeRoom}>この部屋を削除</button>
      </div>
      {room.source === 'photo' && <p class="muted small">撮り直すと壁の形が変わるため、壁に付けた設備と測定の記録は付け直しになります。配置案と家具は残ります。</p>}
    </Page>
  );
}
