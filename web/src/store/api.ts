import { diag, ownerToken } from './db';

export class ApiError extends Error {
  status: number;
  files?: string[];
  /** offline: サーバーに届かない / http: サーバーが理由つきで断った */
  kind: 'offline' | 'http';
  constructor(kind: 'offline' | 'http', status: number, message: string, files?: string[]) {
    super(message);
    this.kind = kind; this.status = status; this.files = files;
  }
}

async function call(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set('X-Owner', await ownerToken());
  let res: Response;
  try {
    res = await fetch(path, { ...init, headers });
  } catch {
    diag(`api offline ${path}`);
    throw new ApiError('offline', 0, 'サーバーに接続できません。通信の状態と、RoomFit サーバーが起動しているかを確認してください。');
  }
  if (!res.ok) {
    let message = `サーバーでエラーが起きました（${res.status}）。`;
    let files: string[] | undefined;
    try {
      const body = await res.json();
      const d = body?.detail;
      if (typeof d === 'string') message = d;
      else if (d?.message) { message = d.message; files = d.files; }
    } catch { /* 本文なし */ }
    if (res.status === 502 || res.status === 504) message = 'RoomFit サーバーが起動していないようです。起動してからもう一度お試しください。';
    diag(`api ${res.status} ${path}`);
    throw new ApiError(res.status >= 502 && res.status <= 504 ? 'offline' : 'http', res.status, message, files);
  }
  return res;
}

export interface Health {
  recon_available: boolean;
  recon_unavailable_reason: string | null;
  pipeline: string;
  min_photos: number;
  max_photos: number;
  products: { rakuten: boolean };
  photo_destination: string;
}
export const health = async (): Promise<Health> => (await call('/api/health')).json();

export interface JobInfo {
  id: string;
  state: 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted';
  stage: string | null;
  /** 写真が多く、範囲に分けて復元しているとき: 何範囲中の何番目か */
  chunk?: number | null;
  chunks?: number;
  n_photos: number;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  queue_ahead?: number;
  error_code?: string;
  error_message?: string;
  error_detail?: string;
}

export async function createJob(photos: { name: string; blob: Blob }[]): Promise<{ id: string; reused: boolean }> {
  const fd = new FormData();
  for (const p of photos) fd.append('photos', p.blob, p.name);
  return (await call('/api/jobs', { method: 'POST', body: fd })).json();
}
export const getJob = async (id: string): Promise<JobInfo> => (await call(`/api/jobs/${id}`)).json();
export const cancelJob = async (id: string) => { await call(`/api/jobs/${id}/cancel`, { method: 'POST' }); };
export const retryJob = async (id: string) => { await call(`/api/jobs/${id}/retry`, { method: 'POST' }); };
export const deleteJob = async (id: string) => { await call(`/api/jobs/${id}`, { method: 'DELETE' }); };
export const deleteAllJobs = async (): Promise<number> => (await (await call('/api/jobs', { method: 'DELETE' })).json()).deleted;

export interface JobRoom {
  outline: [number, number][];
  edge_status: ('observed' | 'unobserved')[];
  ceiling_height: number | null;
  ceiling_observed: boolean;
  obstacles: { kind: string; rect: [number, number, number, number]; yMin: number; yMax: number }[];
  cloud_points: number;
  views?: { n: number; w: number; h: number }[];
  diagnostics: Record<string, unknown>;
  meta: { model: string; n_views: number; sec_inference?: number };
}
export const getJobRoom = async (id: string): Promise<JobRoom> => (await call(`/api/jobs/${id}/room.json`)).json();
export const getJobViewImage = async (id: string, n: number): Promise<Blob> => (await call(`/api/jobs/${id}/views/${String(n).padStart(3, '0')}.jpg`)).blob();
export const getJobViewXyz = async (id: string, n: number): Promise<ArrayBuffer> => (await call(`/api/jobs/${id}/views/${String(n).padStart(3, '0')}.xyz`)).arrayBuffer();
export const getJobSurface = async (id: string, file: string): Promise<Blob> => (await call(`/api/jobs/${id}/surfaces/${file}`)).blob();
export const getJobCloud = async (id: string): Promise<ArrayBuffer> => (await call(`/api/jobs/${id}/cloud.bin`)).arrayBuffer();

export interface ProductHit { title: string; price: number; shop: string; url: string; affiliate: boolean; image: string | null; caption: string; fetchedAt: string }
export const searchProducts = async (q: string): Promise<{ items: ProductHit[]; count: number; source: string }> =>
  (await call(`/api/products/search?q=${encodeURIComponent(q)}`)).json();
