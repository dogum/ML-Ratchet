// The contract every in-browser instrument follows.
//
// An instrument is two files in site/src/instruments/<id>/:
//   net.ts  pure: load the exported pack(s) and run the network — no DOM, so the parity test can run it in Node.
//   ui.ts   export `mount(ctx)`: build the instrument inside ctx.root using the shared kit (kit.ts).
// Its data lives in site/public/instruments/<id>/ and is written by models/<id>/web/export_instrument.py.
// site/tests/parity.test.ts checks net.ts against reference.json from the exporter.

export interface InstrumentContext {
  root: HTMLElement;
  /** URL of this instrument's data folder, e.g. /ML-Ratchet/instruments/quakenet/ (ends with /) */
  base: string;
  accent: string;
  /** compact = embedded in a list/rack (smaller screen, fewer controls) */
  compact: boolean;
}

export type Mount = (ctx: InstrumentContext) => Promise<void>;

/** Each instrument's reference.json (written by the exporter) has this shape. */
export interface Reference {
  cases: { example: number; variant: string; outputs: number[] }[];
  tolerance: number;
}
