/** Prefix a site path with the deployment base (e.g. /ML-Ratchet/). */
export function url(path = ''): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const p = path.replace(/^\//, '');
  return `${base}/${p}`;
}
