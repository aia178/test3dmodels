import { defineConfig } from 'vitest/config';
import preact from '@preact/preset-vite';

// 開発時は API を RoomFit サーバー (127.0.0.1:8787) へ中継する。host: true で同じ Wi-Fi のスマホからも開ける。
export default defineConfig({
  plugins: [preact()],
  server: { host: true, port: 5173, proxy: { '/api': { target: 'http://127.0.0.1:8787', changeOrigin: false } } },
  test: { include: ['src/**/*.test.ts'] },
});
