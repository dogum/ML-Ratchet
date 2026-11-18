import { parsePack, type PackManifest } from '../../../../site/src/lib/nn/pack.ts';
import { show } from '../../../../site/src/instruments/stormnet/ui.ts';
import type { Examples, StormWeights } from '../../../../site/src/instruments/stormnet/net.ts';

function unpack<T>(id: string) {
  const data = JSON.parse(document.getElementById(id)!.textContent!) as { manifest: PackManifest; bytes: string };
  const bytes = Uint8Array.from(atob(data.bytes), c => c.charCodeAt(0));
  return parsePack<T>(data.manifest, bytes.buffer);
}

const root = document.querySelector<HTMLElement>('#instrument')!;
show({ root, compact: false, base: '', accent: '#39a88b' }, unpack<StormWeights>('weights'), unpack<Examples>('examples'));
const toggle = document.querySelector<HTMLButtonElement>('#theme')!;
const dark = () => document.documentElement.dataset.theme
  ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
const themeLabel = () => { toggle.textContent = dark() ? 'Light mode' : 'Dark mode'; };
themeLabel();
toggle.addEventListener('click', () => { document.documentElement.dataset.theme = dark() ? 'light' : 'dark'; themeLabel(); });
