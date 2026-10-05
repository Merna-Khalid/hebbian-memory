import type { PhysioSample } from '@/types'
import { QUADRANT_META } from '@/types'

interface Props { history: PhysioSample[] }

function sparklinePoints(vals: number[], w: number, h: number, lo: number, hi: number): string {
  return vals.map((v, i) => {
    const x = (i / Math.max(vals.length - 1, 1)) * w
    const y = h - ((v - lo) / (hi - lo + 0.001)) * h
    return `${x},${y}`
  }).join(' ')
}

export default function PhysioPanel({ history }: Props) {
  if (!history.length) {
    return (
      <div className="border border-border/60 rounded overflow-hidden">
        <div className="font-mono text-[9px] uppercase tracking-widest text-ink-3 px-2 py-1.5 bg-paper-1 border-b border-border/40">
          Physio (simulated BCI)
        </div>
        <div className="px-2 py-3 text-center text-[11px] text-ink-3">No data yet — hit Step</div>
      </div>
    )
  }

  const sparkW = 196
  const sparkH = 36

  const hrs   = history.map(h => h.hr_bpm ?? 68)
  const hrLo  = Math.min(...hrs, 55)
  const hrHi  = Math.max(...hrs, 105)
  const hrPts = sparklinePoints(hrs, sparkW, sparkH, hrLo, hrHi)

  const atts   = history.map(h => h.attention)
  const attPts = sparklinePoints(atts, sparkW, sparkH, 0, 1)

  const recent = [...history].reverse().slice(0, 6)

  return (
    <div className="border border-border/60 rounded overflow-hidden">
      <div className="font-mono text-[9px] uppercase tracking-widest text-ink-3 px-2 py-1.5 bg-paper-1 border-b border-border/40">
        Physio (simulated BCI)
      </div>

      {/* Heart rate sparkline */}
      <div className="px-2 py-2 border-b border-border/30">
        <div className="font-mono text-[9px] text-ink-3 mb-1">
          heart rate — <code className="text-ink-2">{hrs[hrs.length - 1].toFixed(0)} bpm</code>
        </div>
        <svg width={sparkW} height={sparkH + 8} className="block">
          <polyline points={hrPts} fill="none" stroke="#c84b2f" strokeWidth={1.5}
            strokeLinejoin="round" strokeLinecap="round" />
          {hrs.map((v, i) => {
            const x = (i / Math.max(hrs.length - 1, 1)) * sparkW
            const y = sparkH - ((v - hrLo) / (hrHi - hrLo + 0.001)) * sparkH
            return <circle key={i} cx={x} cy={y} r={2} fill="#c84b2f" />
          })}
        </svg>
      </div>

      {/* Attention sparkline */}
      <div className="px-2 py-2 border-b border-border/30">
        <div className="font-mono text-[9px] text-ink-3 mb-1">
          attention — <code className="text-ink-2">{atts[atts.length - 1].toFixed(2)}</code>
        </div>
        <svg width={sparkW} height={sparkH + 8} className="block">
          <polyline points={attPts} fill="none" stroke="#378ADD" strokeWidth={1.5}
            strokeLinejoin="round" strokeLinecap="round" />
          {atts.map((v, i) => {
            const x = (i / Math.max(atts.length - 1, 1)) * sparkW
            const y = sparkH - (v / 1.001) * sparkH
            return <circle key={i} cx={x} cy={y} r={2} fill="#378ADD" />
          })}
        </svg>
      </div>

      {/* Recent */}
      <div className="px-2 py-2">
        <div className="font-mono text-[9px] text-ink-3 mb-1.5">Recent</div>
        <div className="flex flex-col gap-1">
          {recent.map((h, i) => {
            const meta = h.quadrant ? QUADRANT_META[h.quadrant] : undefined
            return (
              <div key={i} className="flex items-center gap-2 font-mono text-[10px]">
                <span className="font-medium w-5" style={{ color: meta?.color }}>
                  {h.quadrant ?? '—'}
                </span>
                <span className="text-ink-2 w-12">{(h.hr_bpm ?? 0).toFixed(0)} bpm</span>
                <span className="text-ink-3">attn {h.attention.toFixed(2)}</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
