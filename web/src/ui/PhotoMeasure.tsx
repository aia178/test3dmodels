// 写真の上で 2 点を選び、その間の長さ（3D での距離）を出す。
// 写真の各画素には、復元した 3D の位置（部屋座標・縮尺補正前・mm の int16）が対応している。
import { useEffect, useRef, useState } from 'preact/hooks';
import type { Room } from '../core/types';
import * as db from '../store/db';
import { fmtCm } from '../core/units';
import { luminance, measureInOtherViews, median, snapToEdge, type SnapImage, type Snapped } from '../core/snap';

export interface Pick3 { x: number; y: number; z: number; u: number; v: number; n: number }

const INVALID = -32768;

/** 画素 (u,v) のまわり 5×5 で、値のある画素の中央値をとる（1 画素だけのはずれ値を避ける）。 */
export function lookupXyz(xyz: Int16Array, w: number, h: number, u: number, v: number, scale: number): { x: number; y: number; z: number } | null {
  const xs: number[] = [], ys: number[] = [], zs: number[] = [];
  for (let dv = -2; dv <= 2; dv++) for (let du = -2; du <= 2; du++) {
    const uu = Math.round(u) + du, vv = Math.round(v) + dv;
    if (uu < 0 || vv < 0 || uu >= w || vv >= h) continue;
    const i = (vv * w + uu) * 3;
    if (xyz[i] === INVALID) continue;
    xs.push(xyz[i]); ys.push(xyz[i + 1]); zs.push(xyz[i + 2]);
  }
  if (xs.length < 5) return null;
  const med = (a: number[]) => { a.sort((p, q) => p - q); return a[a.length >> 1]; };
  return { x: (med(xs) / 1000) * scale, y: (med(ys) / 1000) * scale, z: (med(zs) / 1000) * scale };
}

export const pickDistance = (a: Pick3, b: Pick3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

export function PhotoMeasure(props: { room: Room; picks: Pick3[]; onPicks: (p: Pick3[]) => void }) {
  const views = props.room.views ?? [];
  const [n, setN] = useState(views[0]?.n ?? 0);
  const [thumbs, setThumbs] = useState<Record<number, string>>({});
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [xyz, setXyz] = useState<Int16Array | null>(null);
  const [miss, setMiss] = useState(false);
  const [lum, setLum] = useState<Float32Array | null>(null);
  const [snapOn, setSnapOn] = useState(true);
  const [snapped, setSnapped] = useState<boolean[]>([]);
  const [multi, setMulti] = useState<{ n: number; lo: number; hi: number } | null>(null);
  const cache = useRef(new Map<number, SnapImage>());
  const run = useRef(0);

  /** ほかの写真の、明るさと 3D 位置を読み込む（最初に必要になったときだけ）。 */
  const loadView = async (v: { n: number; w: number; h: number }): Promise<SnapImage | null> => {
    const hit = cache.current.get(v.n);
    if (hit) return hit;
    const j = await db.get('blobs', `view:${props.room.id}:${v.n}:jpg`), x = await db.get('blobs', `view:${props.room.id}:${v.n}:xyz`);
    if (!j || !(j.blob instanceof Blob) || !x || !(x.blob instanceof ArrayBuffer)) return null;
    try {
      const bmp = await createImageBitmap(j.blob);
      const c = document.createElement('canvas'); c.width = v.w; c.height = v.h;
      const cx = c.getContext('2d', { willReadFrequently: true });
      if (!cx) return null;
      cx.drawImage(bmp, 0, 0, v.w, v.h); bmp.close();
      const im: SnapImage = { w: v.w, h: v.h, lum: luminance(cx.getImageData(0, 0, v.w, v.h).data, v.w, v.h), xyz: new Int16Array(x.blob) };
      cache.current.set(v.n, im);
      return im;
    } catch { return null; }
  };

  /** 選んだ 2 点を、ほかの写真でも測り、中央値に合わせて長さを直す（2 点の中点は動かさない）。 */
  const refineAcross = async (a: Snapped, b: Snapped, picks: Pick3[]) => {
    const token = ++run.current;
    const others: SnapImage[] = [];
    for (const v of views) { if (v.n === n) continue; const im = await loadView(v); if (im) others.push(im); }
    if (token !== run.current) return; // その間に選び直された
    const base = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z), sc = props.room.scale.factor;
    const all = [base, ...measureInOtherViews(others, a, b)];
    setMulti({ n: all.length, lo: Math.min(...all) * sc, hi: Math.max(...all) * sc });
    if (all.length < 2) return;
    const f = median(all) / base;
    const mx = (picks[0].x + picks[1].x) / 2, my = (picks[0].y + picks[1].y) / 2, mz = (picks[0].z + picks[1].z) / 2;
    props.onPicks(picks.map((p) => ({ ...p, x: mx + (p.x - mx) * f, y: my + (p.y - my) * f, z: mz + (p.z - mz) * f })));
  };
  const cvRef = useRef<HTMLCanvasElement>(null);
  const view = views.find((x) => x.n === n);

  useEffect(() => {
    let alive = true;
    const urls: string[] = [];
    (async () => {
      const out: Record<number, string> = {};
      for (const v of views) {
        const b = await db.get('blobs', `view:${props.room.id}:${v.n}:jpg`);
        if (b && b.blob instanceof Blob) { const u = URL.createObjectURL(b.blob); urls.push(u); out[v.n] = u; }
      }
      if (alive) setThumbs(out);
    })();
    return () => { alive = false; urls.forEach((u) => URL.revokeObjectURL(u)); };
  }, [props.room.id, views.length]);

  useEffect(() => {
    let alive = true;
    setImg(null); setXyz(null); setMiss(false); setLum(null); setSnapped([]); setMulti(null); run.current++;
    (async () => {
      const x = await db.get('blobs', `view:${props.room.id}:${n}:xyz`);
      if (!alive) return;
      if (x && x.blob instanceof ArrayBuffer) setXyz(new Int16Array(x.blob));
    })();
    const url = thumbs[n];
    if (url) {
      const im = new Image();
      im.onload = () => {
        if (!alive) return;
        setImg(im);
        // 端を探すための明るさ（写真そのままの大きさで読む）
        const c = document.createElement('canvas'); c.width = im.naturalWidth; c.height = im.naturalHeight;
        const cx = c.getContext('2d', { willReadFrequently: true });
        if (cx) { cx.drawImage(im, 0, 0); setLum(luminance(cx.getImageData(0, 0, c.width, c.height).data, c.width, c.height)); }
      };
      im.src = url;
    }
    return () => { alive = false; };
  }, [n, thumbs[n]]);

  // 描画: 写真、選んだ 2 点、拡大表示（指やカーソルで隠れる端を確かめるため）
  useEffect(() => {
    const cv = cvRef.current;
    if (!cv || !img || !view) return;
    const dpr = window.devicePixelRatio || 1;
    // 写真全体が画面に収まる大きさにする（縦長の写真でスクロールしないと端が選べない、を避ける）
    const cw = Math.min(cv.parentElement!.clientWidth, Math.round((window.innerHeight * 0.62 * view.w) / view.h));
    const ch = Math.round((cw * view.h) / view.w);
    cv.style.width = `${cw}px`;
    cv.style.height = `${ch}px`;
    cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
    const ctx = cv.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, cw, ch);
    const s = cw / view.w;
    const mine = props.picks.filter((p) => p.n === n);
    if (mine.length === 2) {
      ctx.beginPath(); ctx.moveTo(mine[0].u * s, mine[0].v * s); ctx.lineTo(mine[1].u * s, mine[1].v * s);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 4; ctx.stroke(); ctx.strokeStyle = '#2a2f2c'; ctx.lineWidth = 2; ctx.stroke();
    }
    mine.forEach((p, i) => {
      const x = p.u * s, y = p.v * s, col = i === 0 ? '#7a4fa3' : '#1f6fb2';
      ctx.beginPath(); ctx.moveTo(x - 12, y); ctx.lineTo(x + 12, y); ctx.moveTo(x, y - 12); ctx.lineTo(x, y + 12);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 4; ctx.stroke(); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = col; ctx.font = 'bold 13px sans-serif'; ctx.fillText(i === 0 ? 'A' : 'B', x + 8, y - 8);
    });
    const last = mine[mine.length - 1];
    if (last) {
      // 最後に選んだ点のまわりを拡大して、選んだ点の反対側の隅に出す（写真と点を隠さない）
      const B = Math.min(96, Math.round(cw * 0.36)), SRC = 16;
      const bx = last.u * s > cw / 2 ? 6 : cw - B - 6, by = last.v * s > ch / 2 ? 6 : ch - B - 6;
      ctx.save();
      ctx.beginPath(); ctx.rect(bx, by, B, B); ctx.clip();
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, last.u - SRC / 2, last.v - SRC / 2, SRC, SRC, bx, by, B, B);
      ctx.restore();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.strokeRect(bx, by, B, B);
      ctx.strokeStyle = '#2a2f2c'; ctx.lineWidth = 1; ctx.strokeRect(bx, by, B, B);
      ctx.beginPath(); ctx.moveTo(bx + B / 2, by); ctx.lineTo(bx + B / 2, by + B); ctx.moveTo(bx, by + B / 2); ctx.lineTo(bx + B, by + B / 2);
      ctx.strokeStyle = 'rgba(255,255,255,.9)'; ctx.lineWidth = 1; ctx.stroke();
    }
  }, [img, view?.n, props.picks]);

  if (!views.length) return null;
  const tap = (e: MouseEvent) => {
    if (!view || !xyz) return;
    const r = cvRef.current!.getBoundingClientRect();
    const u = ((e.clientX - r.left) / r.width) * view.w, v = ((e.clientY - r.top) / r.height) * view.h;
    const p = lookupXyz(xyz, view.w, view.h, u, v, props.room.scale.factor);
    if (!p) { setMiss(true); return; }
    setMiss(false);
    const pick: Pick3 = { ...p, u, v, n };
    let next = props.picks.length >= 2 ? [pick] : [...props.picks, pick];
    let did = next.map(() => false);
    let snapRes: (Snapped | null)[] = [];
    setMulti(null); run.current++;
    if (next.length === 2 && snapOn && lum && lum.length === view.w * view.h) {
      // 2 点がそろったら、それぞれを近くの端へ寄せる（どちらも、タップした位置から計算する）
      const im = { w: view.w, h: view.h, lum, xyz }, sc = props.room.scale.factor;
      const s = [snapToEdge(im, next[0], next[1]), snapToEdge(im, next[1], next[0])];
      did = s.map((x) => !!x);
      snapRes = s;
      next = next.map((q, i) => (s[i] ? { x: s[i]!.x * sc, y: s[i]!.y * sc, z: s[i]!.z * sc, u: s[i]!.u, v: s[i]!.v, n } : q));
    }
    setSnapped(did);
    props.onPicks(next);
    if (snapRes[0] && snapRes[1] && views.length > 1) void refineAcross(snapRes[0], snapRes[1], next);
  };
  const nudge = (du: number, dv: number) => {
    const last = props.picks[props.picks.length - 1];
    if (!last || !view || !xyz) return;
    const u = last.u + du, v = last.v + dv;
    const p = lookupXyz(xyz, view.w, view.h, u, v, props.room.scale.factor);
    if (p) { run.current++; setMulti(null); props.onPicks([...props.picks.slice(0, -1), { ...p, u, v, n }]); setSnapped((s) => s.map((x, i) => (i === props.picks.length - 1 ? false : x))); }
  };
  return (
    <div class="photo-measure">
      <ul class="photo-strip" role="listbox" aria-label="測る写真を選ぶ">
        {views.map((v) => (
          <li><button role="option" aria-selected={v.n === n} class={v.n === n ? 'on' : ''} onClick={() => { setN(v.n); props.onPicks([]); }}>
            {thumbs[v.n] ? <img src={thumbs[v.n]} alt={`写真 ${v.n + 1}`} /> : <span>{v.n + 1}</span>}
          </button></li>
        ))}
      </ul>
      <canvas ref={cvRef} class="photo-canvas" onClick={tap} aria-label="写真。タップして 2 点を選ぶ" />
      {!img && <p class="muted small">写真を読み込んでいます…</p>}
      {miss && <p class="small warn-text">！ その場所は 3D の位置が取れていません。少しずらしてタップしてください。</p>}
      {props.picks.length > 0 && (
        <div class="nudge" role="group" aria-label="最後に選んだ点を 1 画素ずつ動かす">
          <span>{props.picks.length === 1 ? 'A' : 'B'} を微調整</span>
          <button onClick={() => nudge(0, -1)} aria-label="上へ">↑</button><button onClick={() => nudge(-1, 0)} aria-label="左へ">←</button>
          <button onClick={() => nudge(1, 0)} aria-label="右へ">→</button><button onClick={() => nudge(0, 1)} aria-label="下へ">↓</button>
        </div>
      )}
      <label class="check inline"><input type="checkbox" checked={snapOn} onChange={(e) => setSnapOn((e.target as HTMLInputElement).checked)} /> タップした近くの端に自動で合わせる</label>
      {props.picks.length === 2 && snapOn && (
        <p class="muted small">{snapped.every(Boolean) ? 'A・B とも、近くの端に合わせました。' : snapped.some(Boolean) ? `${snapped[0] ? 'A' : 'B'} だけ端に合わせました（もう一方は、はっきりした端が近くに無かったのでタップした位置のままです）。` : '近くにはっきりした端が無かったので、タップした位置のままです。'}</p>
      )}
      <p class="small">
        {props.picks.length === 0 ? '写真の上で、長さの分かる物の端を 2 か所タップします（例: 扉の左端と右端）。'
          : props.picks.length === 1 ? 'A を選びました。もう一方の端（B）をタップしてください。'
            : <>復元での長さ: 約 <b>{fmtCm(pickDistance(props.picks[0], props.picks[1]))}</b>{multi && multi.n > 1 ? `（同じ場所が写った ${multi.n} 枚の写真で測った中央値。写真ごとの値は ${fmtCm(multi.lo)}〜${fmtCm(multi.hi)}）` : multi ? '（この写真だけで測った値。ほかの写真には同じ場所がはっきり写っていません）' : ''}</>}
        {props.picks.length > 0 && <> <button class="link" onClick={() => { run.current++; setMulti(null); props.onPicks([]); }}>選び直す</button></>}
      </p>
    </div>
  );
}
