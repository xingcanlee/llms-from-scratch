import { getCollection, type CollectionEntry } from 'astro:content'
import { getPart, parts, type Part } from './parts'

export type Chapter = {
  entry: CollectionEntry<'book'>
  part: Part
  /** Path under the site base, without slashes: "math" or "math/chain-rule". */
  path: string
  isIntro: boolean
  /** Book-wide chapter number, 1-based; 0 for a part introduction. */
  number: number
  title: string
  description: string
}

const base = import.meta.env.BASE_URL.replace(/\/?$/, '/')

/** Absolute URL for a path under the site base. */
export function url(path = ''): string {
  return path ? `${base}${path.replace(/^\/|\/$/g, '')}/` : base
}

let cache: Promise<Chapter[]> | undefined

/** Every page in reading order. Chapters are numbered continuously through the book. */
export function getBook(): Promise<Chapter[]> {
  cache ??= getCollection('book').then((entries) => {
    const rank = (entry: CollectionEntry<'book'>) => {
      const part = parts.indexOf(getPart(entry.id.split('/')[0]))
      return part * 1000 + entry.data.order
    }
    entries.sort((a, b) => rank(a) - rank(b))
    let number = 0
    return entries.map((entry, index) => {
      const [partId, name] = entry.id.split('/')
      const isIntro = name === 'index'
      if (isIntro !== (entry.data.order === 0)) throw new Error(`${entry.id}: only index.mdx may use order 0`)
      const previous = entries[index - 1]
      if (previous && rank(previous) === rank(entry)) throw new Error(`${previous.id} and ${entry.id} share an order`)
      if (!isIntro) number += 1
      const part = getPart(partId)
      return {
        entry,
        part,
        path: isIntro ? part.id : `${part.id}/${name}`,
        isIntro,
        number: isIntro ? 0 : number,
        title: entry.data.title,
        description: entry.data.description,
      }
    })
  })
  return cache
}

export async function getPartChapters(partId: string): Promise<Chapter[]> {
  return (await getBook()).filter((chapter) => chapter.part.id === partId)
}

/** "第 12 章" for a chapter, the part numeral for an introduction. */
export function chapterLabel(chapter: Chapter): string {
  return chapter.isIntro ? chapter.part.numeral : `第 ${chapter.number} 章`
}

/** Minutes to read, at roughly 400 CJK characters or 200 Latin words a minute. */
export function readingMinutes(body: string): number {
  const prose = body
    .replace(/```[\s\S]*?```/g, '')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/<[^>]+>/g, '')
    .replace(/\]\([^)]*\)/g, ']')
  const cjk = prose.match(/[㐀-鿿]/g)?.length ?? 0
  const words = prose.replace(/[㐀-鿿]/g, ' ').match(/[A-Za-z0-9]+/g)?.length ?? 0
  return Math.max(1, Math.round(cjk / 400 + words / 200))
}

export const repoUrl = 'https://github.com/xingcanlee/llms-from-scratch'
