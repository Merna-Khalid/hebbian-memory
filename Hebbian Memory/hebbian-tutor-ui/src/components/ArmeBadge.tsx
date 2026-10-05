import type { ArmeState } from '@/types'
import { QUADRANT_META } from '@/types'

export default function ArmeBadge({ arme }: { arme: ArmeState }) {
  const meta = QUADRANT_META[arme.quadrant]
  return (
    <div className="flex items-center gap-2.5 px-1 font-mono text-[10px] text-ink-3 flex-wrap mt-1">
      <span style={{ color: meta.color }} className="font-medium">
        {arme.quadrant} {meta.label}
      </span>
      <span style={{ color: arme.gated ? undefined : arme.m_t > 1 ? '#2d7a4f' : '#c84b2f' }}>
        {arme.gated ? 'gated' : `m_t ${arme.m_t.toFixed(2)}`}
      </span>
      <span>
        V{arme.valence >= 0 ? '+' : ''}{arme.valence.toFixed(2)}
        {' · '}
        A {arme.arousal.toFixed(2)}
        {' · '}
        D {arme.dominance.toFixed(2)}
      </span>
      {arme.delta_w_mean !== undefined && (
        <span style={{ color: arme.delta_w_mean > 0 ? '#2d7a4f' : '#c84b2f' }} className="font-medium">
          Δw {arme.delta_w_mean >= 0 ? '+' : ''}{arme.delta_w_mean.toFixed(4)}
        </span>
      )}
    </div>
  )
}
