// 写真の上でタップした点を、近くの「端」（明るさが急に変わる所）へ吸い付け、その 3D の位置を求める。
// 端そのものの画素は手前と奥の位置が混ざるので、端の内側（もう一方の点の側）の画素から直線で外挿する。
// 指でのタップを想定してずれ ±10 画素を乱数で作った実験（experiments/sim_tap_snap.py）で、探す範囲 ±12 画素のとき
// 測定の標準偏差が 4.6〜5.0cm から 1.1〜2.2cm に減った。丁寧に測った値との差（中央値）は 1〜2cm 残る。

export const XYZ_INVALID = -32768;

export interface SnapImage { w: number; h: number; lum: Float32Array; xyz: Int16Array }
export interface Snapped { u: number; v: number; x: number; y: number; z: number }

function bilinear(L: Float32Array, w: number, h: number, u: number, v: number): number {
  u = Math.min(Math.max(u, 0), w - 1.001); v = Math.min(Math.max(v, 0), h - 1.001);
  const i = Math.floor(u), j = Math.floor(v), a = u - i, b = v - j;
  return L[j * w + i] * (1 - a) * (1 - b) + L[j * w + i + 1] * a * (1 - b) + L[(j + 1) * w + i] * (1 - a) * b + L[(j + 1) * w + i + 1] * a * b;
}

/** 最小二乗の直線 y = a·k + b の b（k=0 での値）を返す。 */
function interceptAtZero(ks: number[], ys: number[]): number {
  const n = ks.length;
  let sk = 0, sy = 0, skk = 0, sky = 0;
  for (let i = 0; i < n; i++) { sk += ks[i]; sy += ys[i]; skk += ks[i] * ks[i]; sky += ks[i] * ys[i]; }
  const den = n * skk - sk * sk;
  const a = Math.abs(den) < 1e-12 ? 0 : (n * sky - sk * sy) / den;
  return (sy - a * sk) / n;
}

/**
 * tap から other へ向かう直線に沿って ±reach 画素の範囲で、最もはっきりした端を探す。
 * 座標は画素の左上を原点とする連続座標（画素 i の中心は i+0.5）。戻り値の x,y,z は m（縮尺補正前）。
 * はっきりした端が無い・内側の 3D 位置が足りない場合は null（呼び出し側は、タップ位置そのままの値を使う）。
 */
export function snapToEdge(img: SnapImage, tap: { u: number; v: number }, other: { u: number; v: number }, reach = 12): Snapped | null {
  let dx = other.u - tap.u, dy = other.v - tap.v;
  const n = Math.hypot(dx, dy);
  if (n < 8) return null;
  dx /= n; dy /= n;
  const ts: number[] = [], prof: number[] = [];
  for (let t = -reach; t <= reach + 1e-9; t += 0.5) { ts.push(t); prof.push(bilinear(img.lum, img.w, img.h, tap.u + t * dx - 0.5, tap.v + t * dy - 0.5)); }
  let k = -1, best = 0;
  const g: number[] = [];
  for (let i = 0; i + 4 < prof.length; i++) { const d = Math.abs(prof[i + 4] - prof[i]); g.push(d); if (d > best) { best = d; k = i; } }
  if (k < 0 || best < 12) return null;
  let off = 0;
  if (k > 0 && k < g.length - 1) { const den = g[k - 1] - 2 * g[k] + g[k + 1]; if (Math.abs(den) > 1e-9) off = (0.5 * (g[k - 1] - g[k + 1])) / den; }
  const tEdge = ts[k + 2] + off * 0.5;
  const eu = tap.u + tEdge * dx, ev = tap.v + tEdge * dy;
  const ks: number[] = [], xs: number[] = [], ys: number[] = [], zs: number[] = [];
  for (let kk = 1.5; kk < 5.6; kk += 1) { // 細い枠の中に収まるよう、内側 1.5〜5.5 画素だけを使う
    const ui = Math.floor(eu + kk * dx), vi = Math.floor(ev + kk * dy);
    if (ui < 0 || vi < 0 || ui >= img.w || vi >= img.h) continue;
    const i = (vi * img.w + ui) * 3;
    if (img.xyz[i] === XYZ_INVALID) continue;
    ks.push(kk); xs.push(img.xyz[i] / 1000); ys.push(img.xyz[i + 1] / 1000); zs.push(img.xyz[i + 2] / 1000);
  }
  if (ks.length < 3) return null;
  return { u: eu, v: ev, x: interceptAtZero(ks, xs), y: interceptAtZero(ks, ys), z: interceptAtZero(ks, zs) };
}

export function luminance(rgba: Uint8ClampedArray, w: number, h: number): Float32Array {
  const L = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) L[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
  return L;
}

/** 写真の中で、3D の位置 p（m・縮尺補正前）に最も近い 3D 位置を持つ画素（中心の座標）を返す。maxD より遠ければ null。 */
export function findNearest(img: SnapImage, p: { x: number; y: number; z: number }, maxD = 0.03): { u: number; v: number } | null {
  const px = p.x * 1000, py = p.y * 1000, pz = p.z * 1000, a = img.xyz;
  let best = Infinity, bi = -1;
  for (let i = 0, n = img.w * img.h; i < n; i++) {
    const x = a[i * 3];
    if (x === XYZ_INVALID) continue;
    const dx = x - px, dy = a[i * 3 + 1] - py, dz = a[i * 3 + 2] - pz, d = dx * dx + dy * dy + dz * dz;
    if (d < best) { best = d; bi = i; }
  }
  if (bi < 0 || Math.sqrt(best) > maxD * 1000) return null;
  return { u: (bi % img.w) + 0.5, v: Math.floor(bi / img.w) + 0.5 };
}

/**
 * ある写真で選んだ 2 点（端へ吸い付け済み）を、ほかの写真でも探して測り直す。戻り値は、ほかの写真それぞれでの長さ [m]。
 * 同じ場所でも写真によって復元の値が少し違う（実データで 89.5cm と 87.3cm）。複数の写真の中央値をとると、
 * どの写真でタップしても同じ値になり、3 か所の縮尺のそろい方が 4.4% → 1.1% に良くなった（experiments/sim_multiview.py）。
 */
export function measureInOtherViews(others: SnapImage[], a: Snapped, b: Snapped): number[] {
  const base = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
  const out: number[] = [];
  for (const img of others) {
    const qa = findNearest(img, a), qb = findNearest(img, b);
    if (!qa || !qb) continue;
    const ra = snapToEdge(img, qa, qb, 6), rb = snapToEdge(img, qb, qa, 6);
    if (!ra || !rb) continue;
    const w = Math.hypot(ra.x - rb.x, ra.y - rb.y, ra.z - rb.z);
    if (Math.abs(w - base) < 0.15 * base) out.push(w); // 別の線を拾った明らかなはずれは除く
  }
  return out;
}

export function median(values: number[]): number {
  const s = [...values].sort((p, q) => p - q), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
