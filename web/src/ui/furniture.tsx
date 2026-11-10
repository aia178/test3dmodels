import { useEffect, useRef, useState } from 'preact/hooks';
import type { Furniture } from '../core/types';
import { dimSourceLabel } from '../core/judge';
import { fmtCmExact } from '../core/units';
import * as db from '../store/db';
import * as api from '../store/api';
import { CmField, Empty, Notice, Page, TabBar, fmtDate, navigate, toast, useLive } from './common';

const CATEGORIES = ['ベッド', '机', '椅子', '棚・本棚', 'チェスト・収納', 'ソファ', 'テーブル', 'テレビ台', '家電', 'その他'];
const COLORS = ['#c9b79c', '#a8b8a2', '#9db0c4', '#c7a6a0', '#b9b2c9', '#d6c98f', '#9a9f9c', '#e3ded2'];
const USE_KIND: Record<Furniture['useSpace']['kind'], string> = {
  unknown: 'まだ決めていない（判定では要確認になります）', none: '特に要らない', drawer: '引き出しを引く', swing_door: '扉を開く', chair: '椅子を引く・座る', access: '前に立って使う',
};

export function blankFurniture(): Furniture {
  return {
    id: db.uid(), name: '', category: 'その他', w: 0, d: 0, h: 0, dimSource: 'measured', dimConfirmed: false, shape: 'box', legClearance: null,
    ownership: 'owned', useSpace: { kind: 'unknown', depth: 0 }, photoIds: [], color: COLORS[0], createdAt: db.now(),
  };
}

export function FurnitureList() {
  const { data } = useLive(async () => (await db.getAll('furniture')).sort((a, b) => b.createdAt.localeCompare(a.createdAt)));
  return (
    <>
      <Page title="家具" actions={<a class="button primary" href="#/furniture/new">＋ 登録</a>}>
        <p class="muted">登録した家具は、どの部屋の配置案でも使えます。店を問わず、手持ちの家具と買いたい家具を一緒に置けます。</p>
        <div class="btn-row"><a class="button" href="#/products">商品を検索して登録</a></div>
        {data === undefined ? <p class="muted">読み込み中…</p> : data.length === 0 ? (
          <Empty title="まだ家具がありません"><p>幅・奥行き・高さをメジャーで測って登録します。買いたい商品は、商品ページの寸法と URL を控えて登録できます。</p></Empty>
        ) : (
          <ul class="cards">
            {data.map((f) => (
              <li><a class="card" href={`#/furniture/${f.id}`}>
                <div class="card-main"><i class="swatch" style={{ background: f.color }} /><b>{f.name}</b><span class="tag">{f.category}</span><span class="tag">{f.ownership === 'owned' ? '持っている' : '購入予定'}</span></div>
                <div class="small">幅 {fmtCmExact(f.w)} × 奥行き {fmtCmExact(f.d)} × 高さ {fmtCmExact(f.h)}</div>
                <div class={f.dimConfirmed ? 'muted small' : 'card-state todo'}>{f.dimConfirmed ? `✓ 寸法確認済み（${dimSourceLabel(f.dimSource)}）` : `？ 寸法が未確認（${dimSourceLabel(f.dimSource)}）`}</div>
              </a></li>
            ))}
          </ul>
        )}
      </Page>
      <TabBar current="/furniture" />
    </>
  );
}

function PhotoThumb(props: { id: string; onRemove: () => void }) {
  const [url, setUrl] = useState('');
  useEffect(() => {
    let u = '';
    db.get('blobs', props.id).then((b) => { if (b && b.blob instanceof Blob) { u = URL.createObjectURL(b.blob); setUrl(u); } });
    return () => { if (u) URL.revokeObjectURL(u); };
  }, [props.id]);
  return <li>{url && <img src={url} alt="家具の写真" />}<button class="thumb-x" onClick={props.onRemove}>外す</button></li>;
}

export function FurnitureEdit(props: { id: string; query: URLSearchParams }) {
  const isNew = props.id === 'new';
  const [f, setF] = useState<Furniture | null | undefined>(undefined);
  const [err, setErr] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const prefilled = useRef(false);
  useEffect(() => {
    if (isNew) {
      const b = blankFurniture();
      const q = props.query;
      if (q.get('title') && !prefilled.current) {
        prefilled.current = true;
        b.name = (q.get('title') ?? '').slice(0, 40); b.ownership = 'planned'; b.dimSource = 'product_page';
        b.product = { url: q.get('url') ?? '', shop: q.get('shop') ?? undefined, title: q.get('title') ?? undefined, price: q.get('price') ? Number(q.get('price')) : undefined, fetchedAt: q.get('at') ?? db.now(), affiliate: q.get('aff') === '1' };
      }
      setF(b);
    } else db.get('furniture', props.id).then((x) => setF(x ?? null));
  }, [props.id]);
  if (f === undefined) return <Page title="家具" back="/furniture"><p class="muted">読み込み中…</p></Page>;
  if (f === null) return <Page title="家具" back="/furniture"><Notice kind="error">この家具は見つかりません。</Notice></Page>;
  const set = (patch: Partial<Furniture>) => setF((prev) => (prev ? { ...prev, ...patch } : prev));
  const save = async () => {
    if (!f.name.trim()) { setErr('家具の名前を入れてください。'); return; }
    if (!(f.w > 0 && f.d > 0 && f.h > 0)) { setErr('幅・奥行き・高さを入れてください。分からないときは、おおよその値を入れて「寸法の根拠」を「おおよその見積もり」にしてください。'); return; }
    if (f.useSpace.kind !== 'none' && f.useSpace.kind !== 'unknown' && !(f.useSpace.depth > 0)) { setErr('使うときに必要な奥行きを入れてください。'); return; }
    if (f.product?.url && !/^https?:\/\//.test(f.product.url)) { setErr('商品ページの URL は http:// または https:// で始めてください。'); return; }
    await db.put('furniture', { ...f, name: f.name.trim() });
    toast('家具を保存しました'); navigate('/furniture');
  };
  const addPhotos = async (list: FileList | null) => {
    if (!list) return;
    const ids = [...f.photoIds];
    for (const file of Array.from(list)) { if (!file.type.startsWith('image/')) continue; const id = `fphoto:${db.uid()}`; await db.put('blobs', { id, blob: file }); ids.push(id); }
    set({ photoIds: ids });
  };
  return (
    <Page title={isNew ? '家具を登録' : '家具を編集'} back="/furniture">
      <label class="field"><span>名前</span><input value={f.name} onInput={(e) => set({ name: (e.target as HTMLInputElement).value })} placeholder="例: 無垢材のデスク" maxLength={40} /></label>
      <div class="row2">
        <label class="field"><span>種類</span><select value={f.category} onChange={(e) => set({ category: (e.target as HTMLSelectElement).value })}>{CATEGORIES.map((c) => <option>{c}</option>)}</select></label>
        <label class="field"><span>持っている／購入予定</span><select value={f.ownership} onChange={(e) => set({ ownership: (e.target as HTMLSelectElement).value as Furniture['ownership'] })}><option value="owned">持っている</option><option value="planned">購入予定</option></select></label>
      </div>

      <h2>寸法</h2>
      <p class="muted small">「幅」は正面から見た左右、「奥行き」は前後です。配置画面では三角の印が正面を示します。</p>
      <div class="row3">
        <CmField label="幅" value={f.w || null} onChange={(x) => set({ w: x ?? 0 })} min={0.1} />
        <CmField label="奥行き" value={f.d || null} onChange={(x) => set({ d: x ?? 0 })} min={0.1} />
        <CmField label="高さ" value={f.h || null} onChange={(x) => set({ h: x ?? 0 })} min={0.1} />
      </div>
      <div class="row2">
        <label class="field"><span>寸法の根拠</span>
          <select value={f.dimSource} onChange={(e) => set({ dimSource: (e.target as HTMLSelectElement).value as Furniture['dimSource'], dimConfirmed: false })}>
            <option value="measured">自分で測った</option><option value="product_page">商品ページの記載</option><option value="estimated">おおよその見積もり</option><option value="unknown">不明</option>
          </select>
        </label>
        <CmField label="脚の下の空き（任意）" value={f.legClearance} onChange={(x) => set({ legClearance: x })} hint="分かるときだけ。コンセントが脚の下に収まるかの判定に使います。" />
      </div>
      {f.dimSource === 'product_page' && <Notice kind="warn">商品ページには、梱包の寸法・本体の寸法・脚や取っ手を含む寸法が混ざって書かれていることがあります。<b>設置したときの本体の外寸</b>を選んで入れてください。</Notice>}
      <label class="check"><input type="checkbox" checked={f.dimConfirmed} disabled={f.dimSource === 'estimated' || f.dimSource === 'unknown'} onChange={(e) => set({ dimConfirmed: (e.target as HTMLInputElement).checked })} /> この寸法は確認済み（幅・奥行き・高さの向きも合っている）</label>
      {!f.dimConfirmed && <p class="muted small">未確認のままでも配置できますが、判定は「問題あり」と断定せず「要確認」になります。</p>}
      <label class="field"><span>寸法についてのメモ（任意）</span><input value={f.dimNote ?? ''} onInput={(e) => set({ dimNote: (e.target as HTMLInputElement).value })} placeholder="例: 取っ手を含む。扉は左開き" maxLength={80} /></label>

      <h2>使うときに必要な空間</h2>
      <div class="row2">
        <label class="field"><span>正面でどう使うか</span>
          <select value={f.useSpace.kind} onChange={(e) => set({ useSpace: { ...f.useSpace, kind: (e.target as HTMLSelectElement).value as Furniture['useSpace']['kind'] } })}>
            {(Object.keys(USE_KIND) as Furniture['useSpace']['kind'][]).map((k) => <option value={k}>{USE_KIND[k]}</option>)}
          </select>
        </label>
        {f.useSpace.kind !== 'none' && f.useSpace.kind !== 'unknown' && <CmField label="正面に必要な奥行き" value={f.useSpace.depth || null} onChange={(x) => set({ useSpace: { ...f.useSpace, depth: x ?? 0 } })} hint="引き出しを全部引いた長さ、扉の幅、椅子を引く距離など。" />}
      </div>

      <h2>見た目</h2>
      <Notice>形は<b>直方体</b>で表示・判定します。脚のすき間や、上が空いている形は表現できません。写真からの立体の自動生成には対応していません（写真は記録として保存します）。</Notice>
      <div class="swatches" role="radiogroup" aria-label="色">
        {COLORS.map((c) => <button role="radio" aria-checked={f.color === c} class={f.color === c ? 'swatch-btn on' : 'swatch-btn'} style={{ background: c }} onClick={() => set({ color: c })} aria-label={`色 ${c}`} />)}
      </div>
      <ul class="thumbs">{f.photoIds.map((id) => <PhotoThumb id={id} onRemove={async () => { await db.del('blobs', id); set({ photoIds: f.photoIds.filter((x) => x !== id) }); }} />)}</ul>
      <button onClick={() => fileRef.current?.click()}>写真を追加（任意）</button>
      <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={(e) => { addPhotos((e.target as HTMLInputElement).files); (e.target as HTMLInputElement).value = ''; }} />

      <h2>商品情報（任意）</h2>
      <label class="field"><span>商品ページの URL</span><input type="url" value={f.product?.url ?? ''} onInput={(e) => { const url = (e.target as HTMLInputElement).value; set({ product: url || f.product?.shop ? { ...(f.product ?? { affiliate: false }), url } : undefined }); }} placeholder="https://…" /></label>
      {f.product && (
        <>
          <div class="row2">
            <label class="field"><span>店の名前</span><input value={f.product.shop ?? ''} onInput={(e) => set({ product: { ...f.product!, shop: (e.target as HTMLInputElement).value } })} maxLength={40} /></label>
            <label class="field"><span>価格（円）</span><input type="number" inputMode="numeric" min={0} value={f.product.price ?? ''} onInput={(e) => { const t = (e.target as HTMLInputElement).value; set({ product: { ...f.product!, price: t === '' ? undefined : Number(t), fetchedAt: db.now() } }); }} /></label>
          </div>
          <p class="muted small">{f.product.fetchedAt ? `価格は ${fmtDate(f.product.fetchedAt)} 時点のものです。最新の価格・在庫は商品ページで確認してください。` : ''}{f.product.affiliate ? ' このリンクはアフィリエイトリンクです（購入されると運営に報酬が入ることがあります。判定や並び順には影響しません）。' : ''}</p>
        </>
      )}

      {err && <Notice kind="error">{err}</Notice>}
      <div class="btn-row sticky">
        <button class="primary" onClick={save}>保存</button>
        {!isNew && <button class="danger" onClick={async () => { if (confirm(`「${f.name}」を削除します。配置案からも取り除かれます。`)) { await db.deleteFurniture(f.id); navigate('/furniture'); } }}>削除</button>}
      </div>
    </Page>
  );
}

/** 商品説明の文章から寸法らしい記載を拾う。候補として見せるだけで、自動では採用しない。 */
function dimensionHints(text: string): string[] {
  const out = new Set<string>();
  const re = /[^。\n]{0,24}(?:幅|奥行|高さ|サイズ|W\s*\d|外寸|梱包)[^。\n]{0,60}?\d+(?:\.\d+)?\s*(?:cm|mm|ｃｍ|ｍｍ|センチ)[^。\n]{0,40}/gi;
  for (const m of text.matchAll(re)) { out.add(m[0].trim().slice(0, 120)); if (out.size >= 4) break; }
  return [...out];
}

export function ProductSearch() {
  const { data: h } = useLive(() => api.health().catch(() => null), []);
  const [q, setQ] = useState('');
  const [res, setRes] = useState<api.ProductHit[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const run = async () => {
    if (!q.trim()) return;
    setBusy(true); setErr(null);
    try { setRes((await api.searchProducts(q.trim())).items); } catch (e) { setErr((e as Error).message); setRes(null); } finally { setBusy(false); }
  };
  const configured = h?.products.rakuten;
  return (
    <Page title="商品を検索して登録" back="/furniture">
      <p class="muted">対応している検索先は<b>楽天市場</b>（公式の商品検索 API）だけです。ほかの店の商品は、寸法と URL を自分で入力して登録できます。</p>
      {h && !configured && <Notice kind="warn">このサーバーには楽天市場のアプリ ID が設定されていないため、検索は使えません。設定方法は README の「商品検索」を見てください。</Notice>}
      <a class="button" href="#/furniture/new">寸法と URL を自分で入力して登録</a>
      <div class="search">
        <input value={q} onInput={(e) => setQ((e.target as HTMLInputElement).value)} onKeyDown={(e) => { if (e.key === 'Enter') run(); }} placeholder="例: デスク 幅120" disabled={!configured} />
        <button class="primary" onClick={run} disabled={!configured || busy}>{busy ? '検索中…' : '検索'}</button>
      </div>
      {err && <Notice kind="error">{err}</Notice>}
      {res && res.length === 0 && <Empty title="見つかりませんでした" />}
      {res && res.length > 0 && <p class="muted small">並び順は楽天市場の標準の順です。報酬の高さで並べ替えていません。価格は取得した時点のものです。</p>}
      <ul class="cards">
        {(res ?? []).map((p) => {
          const hints = dimensionHints(p.caption);
          const qs = new URLSearchParams({ title: p.title, url: p.url, shop: p.shop, price: String(p.price), at: p.fetchedAt, aff: p.affiliate ? '1' : '0' });
          return (
            <li class="card product">
              {p.image && <img src={p.image} alt="" loading="lazy" />}
              <div>
                <b>{p.title}</b>
                <div class="small">{p.price.toLocaleString()} 円（{fmtDate(p.fetchedAt)} 時点）・{p.shop}</div>
                {hints.length ? <div class="small"><b>寸法らしい記載（要確認）:</b><ul>{hints.map((x) => <li>{x}</li>)}</ul></div> : <div class="muted small">説明文から寸法の記載を見つけられませんでした。商品ページで確認してください。</div>}
                <div class="btn-row">
                  <a class="button primary" href={`#/furniture/new?${qs}`}>寸法を入れて登録</a>
                  <a class="button" href={p.url} target="_blank" rel="noopener noreferrer">商品ページ{p.affiliate ? '（アフィリエイトリンク）' : ''}</a>
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </Page>
  );
}
