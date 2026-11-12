// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import { rewriteDocLinks } from './src/lib/rehype-doc-links.mjs';

// GitHub Pages project site: https://dogum.github.io/ML-Ratchet/
const SITE = process.env.SITE_URL ?? 'https://dogum.github.io';
const BASE = process.env.SITE_BASE ?? '/ML-Ratchet';

export default defineConfig({
  site: SITE,
  base: BASE,
  trailingSlash: 'always',
  integrations: [sitemap()],
  markdown: {
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark-dimmed' } },
    rehypePlugins: [[rewriteDocLinks, { base: BASE }]],
  },
  vite: {
    server: { fs: { allow: ['..'] } },
  },
});
