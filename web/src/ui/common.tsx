import type { ComponentChildren } from 'preact';
import { useEffect, useState } from 'preact/hooks';
import { subscribe } from '../store/db';
import type { IssueStatus } from '../core/types';
import { mToCm, cmToM } from '../core/units';

export function navigate(path: string) { location.hash = path; }

export function useRoute(): string {
  const [h, setH] = useState(location.hash.slice(1) || '/');
  useEffect(() => {
    const f = () => setH(location.hash.slice(1) || '/');
    window.addEventListener('hashchange', f);
    return () => window.removeEventListener('hashchange', f);
  }, []);
  return h;
}

/** 端末内の保存を読み、変更があれば読み直す。 */
export function useLive<T>(load: () => Promise<T>, deps: unknown[] = []): { data: T | undefined; error: string | null; reload: () => void } {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    load().then((d) => { if (alive) { setData(d); setError(null); } }).catch((e) => { if (alive) setError(String(e?.message ?? e)); });
    return () => { alive = false; };
  }, [...deps, tick]);
  useEffect(() => subscribe(() => setTick((t) => t + 1)), []);
  return { data, error, reload: () => setTick((t) => t + 1) };
}

export function Page(props: { title: string; back?: string; actions?: ComponentChildren; children: ComponentChildren; wide?: boolean }) {
  return (
    <div class="page">
      <header class="bar">
        {props.back != null && <a class="back" href={`#${props.back}`} aria-label="戻る">‹ 戻る</a>}
        <h1>{props.title}</h1>
        <div class="bar-actions">{props.actions}</div>
      </header>
      <main class={props.wide ? 'content wide' : 'content'}>{props.children}</main>
    </div>
  );
}

export function TabBar(props: { current: string }) {
  const tabs = [['/', '部屋', '▢'], ['/furniture', '家具', '▤'], ['/compare', '比較', '⇄'], ['/settings', '設定', '⚙']] as const;
  return (
    <nav class="tabs" aria-label="メインメニュー">
      {tabs.map(([path, label, glyph]) => (
        <a href={`#${path}`} class={props.current === path ? 'on' : ''} aria-current={props.current === path ? 'page' : undefined}>
          <span aria-hidden="true">{glyph}</span>{label}
        </a>
      ))}
    </nav>
  );
}

// 状態は色だけでなく、記号と言葉でも区別する
export const STATUS_LABEL: Record<IssueStatus, string> = { problem: '問題あり', needs_check: '要確認', clear: '問題を検出せず' };
export const STATUS_GLYPH: Record<IssueStatus, string> = { problem: '✕', needs_check: '？', clear: '✓' };
export function StatusChip(props: { status: IssueStatus; count?: number }) {
  return (
    <span class={`chip ${props.status}`}>
      <span aria-hidden="true">{STATUS_GLYPH[props.status]}</span> {STATUS_LABEL[props.status]}{props.count != null ? ` ${props.count}` : ''}
    </span>
  );
}

export function Notice(props: { kind?: 'info' | 'warn' | 'error'; children: ComponentChildren }) {
  const k = props.kind ?? 'info';
  return <div class={`notice ${k}`} role={k === 'error' ? 'alert' : 'note'}><span aria-hidden="true">{k === 'error' ? '✕' : k === 'warn' ? '！' : 'ⓘ'}</span><div>{props.children}</div></div>;
}

/** cm で入力し、内部は m で持つ欄。空欄は null。 */
export function CmField(props: { label: string; value: number | null; onChange: (m: number | null) => void; min?: number; hint?: string; disabled?: boolean }) {
  const [text, setText] = useState(props.value == null ? '' : String(Math.round(mToCm(props.value) * 10) / 10));
  useEffect(() => {
    const cur = text === '' ? null : cmToM(Number(text));
    if (props.value == null ? cur != null : cur == null || Math.abs(cur - props.value) > 0.0006) setText(props.value == null ? '' : String(Math.round(mToCm(props.value) * 10) / 10));
  }, [props.value]);
  return (
    <label class="field">
      <span>{props.label}</span>
      <span class="unit-input">
        <input type="number" inputMode="decimal" step="0.1" min={props.min ?? 0} value={text} disabled={props.disabled}
          onInput={(e) => {
            const t = (e.target as HTMLInputElement).value; setText(t);
            const n = Number(t);
            props.onChange(t === '' || !Number.isFinite(n) || n < (props.min ?? 0) ? null : cmToM(n));
          }} />
        <span class="unit">cm</span>
      </span>
      {props.hint && <small>{props.hint}</small>}
    </label>
  );
}

export function Empty(props: { title: string; children?: ComponentChildren }) {
  return <div class="empty"><p class="empty-title">{props.title}</p>{props.children}</div>;
}

let toastSet: ((t: string | null) => void) | null = null;
export function toast(msg: string) { toastSet?.(msg); }
export function ToastHost() {
  const [t, setT] = useState<string | null>(null);
  toastSet = setT;
  useEffect(() => { if (!t) return; const h = setTimeout(() => setT(null), 2600); return () => clearTimeout(h); }, [t]);
  return t ? <div class="toast" role="status">{t}</div> : null;
}

export function fmtDate(iso: string): string {
  const d = new Date(iso);
  return `${d.getFullYear()}/${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}
