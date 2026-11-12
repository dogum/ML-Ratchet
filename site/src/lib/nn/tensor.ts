// A minimal dense tensor: row-major Float32Array + shape. Enough to run the collection's networks in the browser.
export interface Tensor {
  data: Float32Array;
  shape: number[];
}

export const size = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

export function tensor(shape: number[], data?: Float32Array | number[]): Tensor {
  const n = size(shape);
  const d = data instanceof Float32Array ? data : data ? Float32Array.from(data) : new Float32Array(n);
  if (d.length !== n) throw new Error(`tensor: ${d.length} values for shape [${shape}]`);
  return { data: d, shape: [...shape] };
}

export const zeros = (shape: number[]): Tensor => tensor(shape);

export function reshape(t: Tensor, shape: number[]): Tensor {
  const s = [...shape];
  const neg = s.indexOf(-1);
  if (neg >= 0) s[neg] = t.data.length / size(s.filter((_, i) => i !== neg));
  if (size(s) !== t.data.length) throw new Error(`reshape: [${t.shape}] -> [${shape}]`);
  return { data: t.data, shape: s };
}

/** Select index i along axis 0 (a view-free copy). */
export function take0(t: Tensor, i: number): Tensor {
  const inner = size(t.shape.slice(1));
  return { data: t.data.slice(i * inner, (i + 1) * inner), shape: t.shape.slice(1) };
}

export function clone(t: Tensor): Tensor {
  return { data: new Float32Array(t.data), shape: [...t.shape] };
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
