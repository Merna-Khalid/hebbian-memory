import type { ExtractedConcept } from '@/types'
import { CONCEPT_COLORS } from '@/types'

export default function ConceptTags({ concepts }: { concepts: ExtractedConcept[] }) {
  if (!concepts.length) return null
  return (
    <div className="flex flex-wrap gap-1.5 px-1 mt-1.5">
      {concepts.map((c, i) => (
        <span
          key={i}
          title={c.description}
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[11px] bg-paper-0 text-ink-1 cursor-default"
          style={{ borderColor: CONCEPT_COLORS[c.type] ?? '#888780' }}
        >
          <span
            className="text-[9px] uppercase tracking-wide"
            style={{ color: CONCEPT_COLORS[c.type] ?? '#888780' }}
          >
            {c.type}
          </span>
          <span className={c.lang === 'ja' ? 'font-jp' : ''}>{c.label}</span>
        </span>
      ))}
    </div>
  )
}
