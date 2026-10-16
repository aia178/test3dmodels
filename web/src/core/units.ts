// 内部は m。画面との境界でだけ cm / mm に変換する。

export const cmToM = (cm: number) => cm / 100;
export const mmToM = (mm: number) => mm / 1000;
export const mToCm = (m: number) => m * 100;

/** 画面表示用。推定値に小数点以下を付けて精密に見せない（cm の整数）。 */
export function fmtCm(m: number): string {
  return `${Math.round(m * 100)}cm`;
}

/** 人が入力・確認した寸法の表示用（0.1cm まで） */
export function fmtCmExact(m: number): string {
  const c = Math.round(m * 1000) / 10;
  return `${Number.isInteger(c) ? c : c.toFixed(1)}cm`;
}

export function parseCm(text: string): number | null {
  const t = text.trim().replace(/[０-９．]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0xfee0)).replace(/cm$/i, '');
  if (t === '') return null;
  const n = Number(t);
  return Number.isFinite(n) && n > 0 ? cmToM(n) : null;
}

export const deg = (rad: number) => (rad * 180) / Math.PI;
export const rad = (d: number) => (d * Math.PI) / 180;
