import { defineConfig } from 'astro/config'
import mdx from '@astrojs/mdx'
import { unified } from '@astrojs/markdown-remark'
import remarkMath from 'remark-math'
import remarkCjkFriendly from 'remark-cjk-friendly'
import rehypeKatex from 'rehype-katex'
import { transformerMetaHighlight, transformerNotationDiff, transformerNotationFocus, transformerNotationHighlight } from '@shikijs/transformers'
import { transformerBookMeta } from './src/plugins/shiki-meta.mjs'
import { rehypeTableScroll } from './src/plugins/rehype-table-scroll.mjs'
import { rehypeBaseLinks } from './src/plugins/rehype-base-links.mjs'
import tailwindcss from '@tailwindcss/vite'

const base = '/llms-from-scratch'
// A running dev server must keep its dependency files when check/build run concurrently.
const cacheCommand = process.argv.find(arg => ['dev', 'check', 'build'].includes(arg)) ?? 'build'

export default defineConfig({
  site: 'https://xingcanlee.github.io',
  base,
  trailingSlash: 'always',
  integrations: [mdx()],
  vite: {
    cacheDir: `node_modules/.vite/${cacheCommand}`,
    plugins: [tailwindcss()],
  },
  markdown: {
    syntaxHighlight: {
      type: 'shiki',
    },
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      defaultColor: false,
      transformers: [
        transformerMetaHighlight(),
        transformerNotationHighlight({ matchAlgorithm: 'v3' }),
        transformerNotationDiff({ matchAlgorithm: 'v3' }),
        transformerNotationFocus({ matchAlgorithm: 'v3' }),
        transformerBookMeta(),
      ],
    },
    processor: unified({
      remarkPlugins: [remarkCjkFriendly, remarkMath],
      rehypePlugins: [
        rehypeTableScroll,
        [rehypeBaseLinks, { base }],
        [rehypeKatex, { throwOnError: true, strict: 'error' }],
      ],
    }),
  },
})
