import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// The long-form docs live in the repo's docs/ folder (they read well on GitHub too).
const docs = defineCollection({
  loader: glob({ pattern: ['*.md', '!backlog.md'], base: '../docs' }),
  schema: z.object({}).passthrough(),
});

export const collections = { docs };
