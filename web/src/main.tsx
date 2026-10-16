import { render } from 'preact';
import './styles.css';
import { ToastHost, useRoute, Page, Notice } from './ui/common';
import { Capture, Home, JobView, RoomHub, RoomNew } from './ui/rooms';
import { RoomCheck } from './ui/check';
import { FurnitureEdit, FurnitureList, ProductSearch } from './ui/furniture';
import { Editor } from './ui/editor';
import { Compare, Settings } from './ui/more';
import { diag } from './store/db';

window.addEventListener('error', (e) => diag(`error: ${e.message}`));
window.addEventListener('unhandledrejection', (e) => diag(`rejection: ${String((e.reason as Error)?.message ?? e.reason)}`));

function App() {
  const route = useRoute();
  const [path, qs] = route.split('?');
  const query = new URLSearchParams(qs ?? '');
  const seg = path.split('/').filter(Boolean);
  let view;
  if (seg.length === 0) view = <Home />;
  else if (seg[0] === 'rooms' && seg[1] === 'new') view = <RoomNew query={query} />;
  else if (seg[0] === 'rooms' && seg[2] === 'capture') view = <Capture key={seg[1]} roomId={seg[1]} query={query} />;
  else if (seg[0] === 'rooms' && seg[2] === 'job') view = <JobView key={seg[1]} roomId={seg[1]} />;
  else if (seg[0] === 'rooms' && seg[2] === 'check') view = <RoomCheck key={seg[1]} roomId={seg[1]} />;
  else if (seg[0] === 'rooms' && seg[2] === 'layout' && seg[3]) view = <Editor key={seg[3]} roomId={seg[1]} layoutId={seg[3]} />;
  else if (seg[0] === 'rooms' && seg[1]) view = <RoomHub key={seg[1]} roomId={seg[1]} />;
  else if (seg[0] === 'furniture' && seg[1]) view = <FurnitureEdit key={seg[1] + (qs ?? '')} id={seg[1]} query={query} />;
  else if (seg[0] === 'furniture') view = <FurnitureList />;
  else if (seg[0] === 'products') view = <ProductSearch />;
  else if (seg[0] === 'compare') view = <Compare />;
  else if (seg[0] === 'settings') view = <Settings />;
  else view = <Page title="ページが見つかりません" back="/"><Notice kind="warn">この画面はありません。</Notice></Page>;
  return <>{view}<ToastHost /></>;
}

render(<App />, document.getElementById('app')!);
