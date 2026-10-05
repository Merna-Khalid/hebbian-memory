import { useState, useRef, useEffect, useCallback } from 'react'
import { nextSecondBrainEvent, resetSecondBrainStream } from '@/api'
import type { SecondBrainEvent, PhysioSample } from '@/types'
import ArmeBadge from '@/components/ArmeBadge'
import ConceptTags from '@/components/ConceptTags'
import PhysioPanel from '@/components/PhysioPanel'
import DashboardView from './DashboardView'

const STEP_INTERVAL_MS = 2000

interface Props { sessionId: string | null }

export default function SecondBrainView({ sessionId }: Props) {
  const [events, setEvents]     = useState<SecondBrainEvent[]>([])
  const [physio, setPhysio]     = useState<PhysioSample[]>([])
  const [playing, setPlaying]   = useState(false)
  const [done, setDone]         = useState(false)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const bottomRef               = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events])

  const step = useCallback(async () => {
    if (!sessionId || loading || done) return
    setLoading(true)
    setError(null)
    try {
      const ev = await nextSecondBrainEvent(sessionId)
      if (ev.done) {
        setDone(true)
        setPlaying(false)
        return
      }
      setEvents(prev => [...prev, ev])
      setPhysio(prev => [...prev, {
        hr_bpm:    ev.hr_bpm ?? null,
        attention: ev.attention ?? 1,
        quadrant:  ev.arme?.quadrant,
        m_t:       ev.arme?.m_t,
        timestamp: Date.now(),
      }])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPlaying(false)
    } finally {
      setLoading(false)
    }
  }, [sessionId, loading, done])

  // Auto-advance while playing.
  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => void step(), STEP_INTERVAL_MS)
    return () => clearInterval(id)
  }, [playing, step])

  const reset = useCallback(async () => {
    setPlaying(false)
    setError(null)
    try {
      await resetSecondBrainStream()
      setEvents([])
      setPhysio([])
      setDone(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  return (
    <div className="flex w-full h-full overflow-hidden">

      {/* Stream feed */}
      <div className="flex-1 flex flex-col border-r border-border overflow-hidden">

        {/* Toolbar */}
        <div className="flex items-center justify-between gap-4 px-4 py-2 border-b border-border/40 bg-paper-0 flex-shrink-0">
          <div className="font-mono text-[10px] text-ink-3 uppercase tracking-widest">
            simulated ambient stream
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => void step()} disabled={!sessionId || loading || done} className="btn text-[11px]">
              step
            </button>
            <button
              onClick={() => setPlaying(p => !p)}
              disabled={!sessionId || done}
              className="btn text-[11px]"
            >
              {playing ? '❚❚ pause' : '▶ play'}
            </button>
            <button onClick={() => void reset()} className="btn text-[11px]">↻ reset</button>
          </div>
        </div>

        {/* Feed */}
        <div className="flex-1 overflow-y-auto px-6 py-7 flex flex-col gap-6">
          {events.length === 0 && (
            <div className="m-auto text-center text-ink-3 flex flex-col gap-3">
              <span className="text-5xl">🧠</span>
              <p className="text-sm">Hit step or play to start the simulated BCI stream.</p>
            </div>
          )}
          {events.map((ev, i) => (
            <div key={i} className="flex flex-col gap-1 items-start">
              <span className="font-mono text-[10px] text-ink-3 px-1">{ev.scenario_label}</span>
              <div className="max-w-[88%] px-3.5 py-2.5 rounded-lg leading-relaxed bg-paper-1 border border-border/60 text-ink-0 font-serif text-sm rounded-bl-sm">
                {ev.utterance}
              </div>
              {ev.arme && <ArmeBadge arme={ev.arme} />}
              {ev.concepts && ev.concepts.length > 0 && <ConceptTags concepts={ev.concepts} />}
            </div>
          ))}
          {done && (
            <div className="text-center text-ink-3 text-xs">
              scenario list exhausted — hit reset to replay
            </div>
          )}
          {error && (
            <div className="text-vermillion text-xs px-3 py-2 rounded border border-vermillion-dim bg-vermillion-bg">
              ⚠ {error}
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Physio + filtered memory graph */}
      <div className="w-[600px] flex-shrink-0 flex flex-col overflow-hidden">
        <div className="p-3 flex-shrink-0">
          <PhysioPanel history={physio} />
        </div>
        <div className="flex-1 min-h-0 border-t border-border">
          <DashboardView filterSourceType="sensor" showArmeHistory={false} />
        </div>
      </div>
    </div>
  )
}
