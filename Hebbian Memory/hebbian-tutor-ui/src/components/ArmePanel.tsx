import type { ArmeState } from '../types'
import { QUADRANT_META } from '../types'

interface Props { history: ArmeState[] }

export default function ArmePanel({ history }: Props) {
  if (!history.length) {
    return (
      <div className="border border-border/60 rounded overflow-hidden">
        <div className="font-mono text-[9px] uppercase tracking-widest text-ink-3 px-2 py-1.5 bg-paper-1 border-b border-border/40">
          ARM-E history
        </div>
        <div className="px-2 py-3 text-center text-[11px] text-ink-3">No data yet</div>
      </div>
    )
  }

  const total   = history.length
  const qCounts = { Q1: 0, Q2: 0, Q3: 0, Q4: 0 } as Record<string, number>
  history.forEach(h => { if (h.quadrant) qCounts[h.quadrant]++ })

  const mts    = history.map(h => h.m_t ?? 1)
  const mtMin  = Math.min(...mts, 0.2)
  const mtMax  = Math.max(...mts, 1.5)
  const sparkW = 196
  const sparkH = 36
  const pts    = mts.map((m, i) => {
    const x = (i / Math.max(mts.length - 1, 1)) * sparkW
    const y = sparkH - ((m - mtMin) / (mtMax - mtMin + 0.001)) * sparkH
    return `${x},${y}`
  }).join(' ')

  const recent = [...history].reverse().slice(0, 6)

  return (
    <div className="border border-border/60 rounded overflow-hidden">
      <div className="font-mono text-[9px] uppercase tracking-widest text-ink-3 px-2 py-1.5 bg-paper-1 border-b border-border/40">
        ARM-E history
      </div>

      {/* Quadrant bars */}
      <div className="px-2 py-2 border-b border-border/30">
        <div className="font-mono text-[9px] text-ink-3 mb-1.5">Emotion distribution</div>
        <div className="flex flex-col gap-1">
          {Object.entries(qCounts).map(([q, count]) => {
            const meta = QUADRANT_META[q as keyof typeof QUADRANT_META]
            return (
              <div key={q} className="flex items-center gap-1.5">
                <span className="font-mono text-[10px] font-medium w-5" style={{ color: meta.color }}>{q}</span>
                <div className="flex-1 h-1.5 bg-paper-2 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${(count / total) * 100}%`, background: meta.color }}
                  />
                </div>
                <span className="font-mono text-[9px] text-ink-3 w-4 text-right">{count}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Sparkline */}
      <div className="px-2 py-2 border-b border-border/30">
        <div className="font-mono text-[9px] text-ink-3 mb-1">m_t over session</div>
        <svg width={sparkW} height={sparkH + 8} className="block">
          <line
            x1={0} y1={sparkH - ((1 - mtMin) / (mtMax - mtMin + 0.001)) * sparkH}
            x2={sparkW} y2={sparkH - ((1 - mtMin) / (mtMax - mtMin + 0.001)) * sparkH}
            stroke="#d0ccba" strokeWidth={0.8} strokeDasharray="2 2"
          />
          <polyline points={pts} fill="none" stroke="#2d7a4f" strokeWidth={1.5}
            strokeLinejoin="round" strokeLinecap="round" />
          {mts.map((m, i) => {
            const x = (i / Math.max(mts.length - 1, 1)) * sparkW
            const y = sparkH - ((m - mtMin) / (mtMax - mtMin + 0.001)) * sparkH
            return <circle key={i} cx={x} cy={y} r={2} fill={m > 1 ? '#2d7a4f' : '#c84b2f'} />
          })}
        </svg>
      </div>

      {/* Recent */}
      <div className="px-2 py-2">
        <div className="font-mono text-[9px] text-ink-3 mb-1.5">Recent</div>
        <div className="flex flex-col gap-1">
          {recent.map((h, i) => {
            const meta = QUADRANT_META[h.quadrant as keyof typeof QUADRANT_META]
            return (
              <div key={i} className="flex items-center gap-2 font-mono text-[10px]">
                <span className="font-medium w-5" style={{ color: meta?.color }}>{h.quadrant}</span>
                <span style={{ color: h.m_t > 1 ? '#2d7a4f' : '#c84b2f' }} className="w-9">
                  {h.gated ? 'gated' : h.m_t.toFixed(2)}
                </span>
                <span className="text-ink-3">
                  V{h.valence >= 0 ? '+' : ''}{h.valence.toFixed(1)}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}