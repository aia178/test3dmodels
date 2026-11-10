// 端末内の保存（IndexedDB）。アカウント登録なしで使える。部屋・家具・配置案は変更のたびに保存する。
import { openDB, type IDBPDatabase } from 'idb';
import type { Furniture, JudgeSettings, Layout, Room } from '../core/types';
import { DEFAULT_SETTINGS } from '../core/judge';
import { addMeasurement, applyJobResult, newRoomBase, type JobRoomLike } from '../core/room';

export interface PendingCapture {
  roomId: string;
  /** 送信前の写真。送信・復元が終わったら消す。 */
  photos: { id: string; name: string; blob: Blob; warn: string[] }[];
  jobId?: string;
  jobStartedAt?: string;
}

interface Schema {
  rooms: Room;
  furniture: Furniture;
  layouts: Layout;
  captures: PendingCapture;
  blobs: { id: string; blob: Blob | ArrayBuffer };
  meta: { key: string; value: unknown };
}

let dbp: Promise<IDBPDatabase> | null = null;
function db() {
  dbp ??= openDB('roomfit', 1, {
    upgrade(d) {
      d.createObjectStore('rooms', { keyPath: 'id' });
      d.createObjectStore('furniture', { keyPath: 'id' });
      d.createObjectStore('layouts', { keyPath: 'id' }).createIndex('roomId', 'roomId');
      d.createObjectStore('captures', { keyPath: 'roomId' });
      d.createObjectStore('blobs', { keyPath: 'id' });
      d.createObjectStore('meta', { keyPath: 'key' });
    },
  });
  return dbp;
}

export const uid = () => crypto.randomUUID().replace(/-/g, '').slice(0, 16);
export const now = () => new Date().toISOString();

type Store = keyof Schema;
export async function getAll<S extends Store>(s: S): Promise<Schema[S][]> { return (await db()).getAll(s); }
export async function get<S extends Store>(s: S, id: string): Promise<Schema[S] | undefined> { return (await db()).get(s, id); }
export async function put<S extends Store>(s: S, value: Schema[S]): Promise<void> { await (await db()).put(s, value); notify(); }
export async function del(s: Store, id: string): Promise<void> { await (await db()).delete(s, id); notify(); }

export async function layoutsOf(roomId: string): Promise<Layout[]> {
  return (await db()).getAllFromIndex('layouts', 'roomId', roomId);
}

export async function getMeta<T>(key: string, fallback: T): Promise<T> {
  const r = await get('meta', key);
  return r ? (r.value as T) : fallback;
}
export async function setMeta(key: string, value: unknown) { await put('meta', { key, value }); }

/** この端末を表すトークン。サーバーはこれが一致する要求にだけ写真・結果を返す。 */
export async function ownerToken(): Promise<string> {
  let t = await getMeta<string | null>('owner', null);
  if (!t) { t = crypto.randomUUID() + crypto.randomUUID(); await setMeta('owner', t); }
  return t;
}

export async function getSettings(): Promise<JudgeSettings> {
  return { ...DEFAULT_SETTINGS, ...(await getMeta<Partial<JudgeSettings>>('settings', {})) };
}

export async function deleteRoom(roomId: string) {
  const d = await db();
  for (const l of await layoutsOf(roomId)) await d.delete('layouts', l.id);
  await d.delete('captures', roomId);
  await d.delete('blobs', `cloud:${roomId}`);
  const room = (await d.get('rooms', roomId)) as Room | undefined;
  for (const v of room?.views ?? []) { await d.delete('blobs', `view:${roomId}:${v.n}:jpg`); await d.delete('blobs', `view:${roomId}:${v.n}:xyz`); }
  for (const f of surfaceFiles(room)) await d.delete('blobs', `surface:${roomId}:${f}`);
  await d.delete('rooms', roomId);
  notify();
}

export async function deleteFurniture(id: string) {
  const d = await db();
  const f = (await d.get('furniture', id)) as Furniture | undefined;
  for (const pid of f?.photoIds ?? []) await d.delete('blobs', pid);
  for (const l of (await d.getAll('layouts')) as Layout[]) {
    const items = l.items.filter((p) => p.furnitureId !== id);
    if (items.length !== l.items.length) await d.put('layouts', { ...l, items, updatedAt: now() });
  }
  await d.delete('furniture', id);
  notify();
}

export async function deleteEverything() {
  const d = await db();
  for (const s of ['rooms', 'furniture', 'layouts', 'captures', 'blobs'] as const) await d.clear(s);
  notify();
}

export async function exportBackup(): Promise<string> {
  return JSON.stringify({
    app: 'roomfit', version: 1, exportedAt: now(),
    note: '写真と点群は含みません（部屋の形・家具・配置案のみ）。',
    rooms: (await getAll('rooms')).map((r) => ({ ...r, hasPointCloud: false })),
    furniture: (await getAll('furniture')).map((f) => ({ ...f, photoIds: [] })),
    layouts: await getAll('layouts'),
  }, null, 1);
}

export async function importBackup(text: string): Promise<{ rooms: number; furniture: number; layouts: number }> {
  const data = JSON.parse(text);
  if (data?.app !== 'roomfit' || !Array.isArray(data.rooms)) throw new Error('RoomFit のバックアップファイルではありません。');
  const d = await db();
  for (const r of data.rooms) await d.put('rooms', r);
  for (const f of data.furniture ?? []) await d.put('furniture', f);
  for (const l of data.layouts ?? []) await d.put('layouts', l);
  notify();
  return { rooms: data.rooms.length, furniture: (data.furniture ?? []).length, layouts: (data.layouts ?? []).length };
}

/** 部屋が持っている、壁・床の写真の画像ファイル名。 */
export function surfaceFiles(room: Room | undefined): string[] {
  const s = room?.surfaces;
  return s ? [...s.walls.map((w) => w.file), ...(s.floor ? [s.floor.file] : [])] : [];
}

// ---- 部屋ファイル: 1 つの部屋を、復元した点・配置案・使っている家具ごと 1 ファイルにする。
// 別のブラウザや端末へ移すとき、発表用に用意しておくときに使う。元の写真は含まない。
function toBase64(buf: ArrayBuffer): string {
  const b = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode(...b.subarray(i, i + 0x8000));
  return btoa(s);
}
function fromBase64(text: string): ArrayBuffer {
  const s = atob(text);
  const b = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i);
  return b.buffer;
}

export async function exportRoomFile(roomId: string): Promise<{ name: string; text: string }> {
  const room = await get('rooms', roomId);
  if (!room) throw new Error('この部屋は見つかりません。');
  const layouts = await layoutsOf(roomId);
  const used = new Set(layouts.flatMap((l) => l.items.map((p) => p.furnitureId)));
  const furniture = (await getAll('furniture')).filter((f) => used.has(f.id)).map((f) => ({ ...f, photoIds: [] }));
  const cloud = await get('blobs', `cloud:${roomId}`);
  const views = [];
  for (const v of room.views ?? []) {
    const j = await get('blobs', `view:${roomId}:${v.n}:jpg`), x = await get('blobs', `view:${roomId}:${v.n}:xyz`);
    if (j && j.blob instanceof Blob && x && x.blob instanceof ArrayBuffer) views.push({ ...v, jpgB64: toBase64(await j.blob.arrayBuffer()), xyzB64: toBase64(x.blob) });
  }
  const surfacesB64: Record<string, string> = {};
  for (const f of surfaceFiles(room)) {
    const b = await get('blobs', `surface:${roomId}:${f}`);
    if (b && b.blob instanceof Blob) surfacesB64[f] = toBase64(await b.blob.arrayBuffer());
  }
  const text = JSON.stringify({
    app: 'roomfit', kind: 'room', version: 1, exportedAt: now(), name: room.name,
    note: '復元した点（色つき）、壁・床に貼った写真、位置を選ぶための縮小した写真を含みます。',
    room, layouts, furniture, views, cloudB64: cloud && cloud.blob instanceof ArrayBuffer ? toBase64(cloud.blob) : null, surfacesB64,
  });
  return { name: room.name, text };
}

/** 部屋ファイルを読み込み、新しい部屋として追加する（既存の部屋は上書きしない）。 */
export async function importRoomFile(text: string): Promise<Room> {
  let data: {
    app?: string; kind?: string; name?: string; room?: Room; layouts?: Layout[]; furniture?: Furniture[]; cloudB64?: string | null;
    jobResult?: JobRoomLike;
    /** 復元結果のファイルに、検証用の測定を最初から入れておく場合（デモ・検証用） */
    measurements?: Room['measurements'];
    /** 復元結果のファイルに、設備（ドア・コンセントなど）を最初から入れておく場合（デモ用）。位置は復元のままの縮尺で書く */
    fixtures?: Room['fixtures'];
    views?: { n: number; w: number; h: number; jpgB64: string; xyzB64: string }[];
    /** 壁・床に写真を貼った画像（ファイル名 → JPEG の base64） */
    surfacesB64?: Record<string, string>;
  };
  try { data = JSON.parse(text); } catch { throw new Error('ファイルを読めませんでした（RoomFit の部屋ファイルではないようです）。'); }
  if (data?.app !== 'roomfit' || (data.kind !== 'room' && data.kind !== 'recon')) throw new Error('RoomFit の部屋ファイルではありません。');
  const id = uid();
  let room: Room;
  if (data.kind === 'recon') {
    // 復元の結果そのもの。サーバーの復元が終わったときと同じ処理で部屋にする
    if (!data.jobResult) throw new Error('復元結果が入っていません。');
    room = applyJobResult(newRoomBase(id, data.name || '読み込んだ部屋', '', 'current'), 'file', data.jobResult, uid);
    // 設備は縮尺合わせの前に入れる（縮尺合わせで位置が一緒に伸び縮みする。幅・高さは実寸なので変えない）
    if (Array.isArray(data.fixtures)) room = { ...room, fixtures: data.fixtures };
    // 縮尺合わせの測定は最後に入れる（アプリで測ったときと同じ処理で、それまでの測定も含めて部屋全体の大きさが合わせ直される）
    const ms = Array.isArray(data.measurements) ? [...data.measurements].sort((p, q) => (p.role === 'calibration' ? 1 : 0) - (q.role === 'calibration' ? 1 : 0)) : [];
    for (const m of ms) room = addMeasurement(room, m);
  } else {
    if (!data.room?.outline?.length) throw new Error('部屋の形が入っていません。');
    room = { ...data.room, id, updatedAt: now() };
  }
  const d = await db();
  const cloud = data.cloudB64 ? fromBase64(data.cloudB64) : null;
  if (cloud) await d.put('blobs', { id: `cloud:${id}`, blob: cloud });
  const views = data.views ?? [];
  for (const v of views) {
    await d.put('blobs', { id: `view:${id}:${v.n}:jpg`, blob: new Blob([fromBase64(v.jpgB64)], { type: 'image/jpeg' }) });
    await d.put('blobs', { id: `view:${id}:${v.n}:xyz`, blob: fromBase64(v.xyzB64) });
  }
  let haveAllSurfaces = true;
  for (const f of surfaceFiles(room)) {
    const b64 = data.surfacesB64?.[f];
    if (!b64) { haveAllSurfaces = false; continue; }
    await d.put('blobs', { id: `surface:${id}:${f}`, blob: new Blob([fromBase64(b64)], { type: 'image/jpeg' }) });
  }
  room = { ...room, hasPointCloud: !!cloud, views: views.map((v) => ({ n: v.n, w: v.w, h: v.h })), ...(haveAllSurfaces ? {} : { surfaces: undefined }) };
  await d.put('rooms', room);
  for (const f of data.furniture ?? []) if (!(await d.get('furniture', f.id))) await d.put('furniture', f);
  // 配置案の位置は、読み込んだ後の部屋（縮尺合わせ後）の座標で書いてあるものとして、そのまま使う
  for (const l of data.layouts ?? []) await d.put('layouts', { ...l, id: uid(), roomId: id });
  notify();
  return room;
}

// 変更の通知（一覧画面の再読込用）
const listeners = new Set<() => void>();
function notify() { listeners.forEach((f) => f()); }
export function subscribe(f: () => void) { listeners.add(f); return () => { listeners.delete(f); }; }

// 開発者向けの診断ログ。画像・トークン・住所などは入れない。
const logBuf: string[] = [];
export function diag(msg: string) {
  logBuf.push(`${new Date().toISOString()} ${msg}`);
  if (logBuf.length > 300) logBuf.shift();
}
export const diagLog = () => logBuf.slice();
