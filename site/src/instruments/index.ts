// Registry of live instruments, discovered automatically: every site/src/instruments/<id>/ui.ts is one.
// Each is lazy-loaded, so a page only downloads the instrument it shows.
import type { Mount } from './types.ts';

const found = import.meta.glob<{ mount: Mount }>('./*/ui.ts');

export const instruments: Record<string, () => Promise<{ mount: Mount }>> = Object.fromEntries(
  Object.entries(found).map(([path, load]) => [path.split('/')[1], load]),
);

export const hasInstrument = (id: string) => id in instruments;
