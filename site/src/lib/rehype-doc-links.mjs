// Rewrites links between the repo's markdown docs into site routes, so docs/*.md read well on GitHub
// AND on the site:  method.md -> /method/,  ../models/quakenet -> /models/quakenet/,  other repo files -> GitHub.
export const DOC_ROUTES = {
  'method.md': 'method',
  'model-contract.md': 'contract',
  'adding-a-model.md': 'contribute',
  'agent-brief.md': 'agent-brief',
  'lessons.md': 'lessons',
  'backlog.md': 'backlog',
  'data-sources.md': 'data',
  'roadmap.md': 'roadmap',
  'vehicle-platform.md': 'vehicle',
};

const REPO = 'https://github.com/dogum/ML-Ratchet/blob/main/';

function walk(node, fn) {
  fn(node);
  if (node.children) for (const c of node.children) walk(c, fn);
}

export function rewriteDocLinks({ base = '/' } = {}) {
  const root = base.endsWith('/') ? base : base + '/';
  return (tree) => {
    walk(tree, (node) => {
      if (node.type !== 'element' || node.tagName !== 'a') return;
      const href = node.properties?.href;
      if (typeof href !== 'string' || /^(https?:|mailto:|#)/.test(href)) return;
      const [path, hash] = href.split('#');
      const file = path.split('/').pop() ?? '';
      let out;
      if (DOC_ROUTES[file]) out = `${root}${DOC_ROUTES[file]}/`;
      else if (/(^|\/)models\/([a-z0-9-]+)\/?$/.test(path)) out = `${root}models/${path.match(/models\/([a-z0-9-]+)/)[1]}/`;
      else out = REPO + path.replace(/^(\.\.\/)+/, '').replace(/^\.\//, 'docs/');
      node.properties.href = out + (hash ? `#${hash}` : '');
    });
  };
}
