import { useState, useEffect, useCallback, useRef } from 'react'
import {
  getFadingConcepts, nextPractice, submitPracticeAnswer,
} from '../api'
import type {
  FadingConcept, PracticeExercise, PracticeResult,
} from '@/types'

const EXERCISE_LABELS: Record<string, string> = {
  translation:         'EN → JA',
  fill_in_blank:       'fill in',
  sentence_production: 'produce',
  form_transform:      'transform',
}

function FadingBar({ value }: { value: number }) {
  // fading_score 0..~2.5 — clamp display at 1 for the bar
  const pct = Math.min(100, Math.round(value * 100))
  const color = pct > 60 ? '#c84b2f' : pct > 25 ? '#d85a30' : '#888780'
  return (
    <div className="h-1 w-16 bg-paper-2 rounded-full overflow-hidden">
      <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
    </div>
  )
}

interface Props { sessionId: string | null }

export default function PracticeView({ sessionId }: Props) {
  const [exercise, setExercise]   = useState<PracticeExercise | null>(null)
  const [result, setResult]       = useState<PracticeResult | null>(null)
  const [fading, setFading]       = useState<FadingConcept[]>([])
  const [answer, setAnswer]       = useState('')
  const [showHint, setShowHint]   = useState(false)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState<string | null>(null)
  const inputRef                  = useRef<HTMLTextAreaElement>(null)

  const refreshQueue = useCallback(() => {
    if (!sessionId) return
    getFadingConcepts(8)
      .then(setFading)
      .catch(() => {/* queue is non-critical — leave stale */})
  }, [sessionId])

  useEffect(() => { refreshQueue() }, [refreshQueue])

  const load = useCallback(async () => {
    if (!sessionId || loading) return
    setLoading(true)
    setError(null)
    setAnswer('')
    setShowHint(false)
    try {
      const data = await nextPractice(sessionId)
      if (data.none) {
        setExercise(null)
        setError(null)
        setExercise({ none: true } as PracticeExercise)
      } else {
        setExercise(data)
        setResult(null)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }, [sessionId, loading])

  const submit = useCallback(async () => {
    if (!sessionId || !exercise || !answer.trim() || result) return
    setLoading(true)
    setError(null)
    try {
      const data = await submitPracticeAnswer(sessionId, answer.trim())
      setResult(data)
      refreshQueue()   // correct answers re-encode memory — queue shifts
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [sessionId, exercise, answer, result, refreshQueue])

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void submit() }
  }

  return (
    <div className="flex h-full w-full max-w-3xl mx-auto">

      {/* Exercise column */}
      <div className="flex-1 flex flex-col overflow-y-auto px-6 py-7 gap-5">

        <div className="flex items-baseline justify-between">
          <h2 className="font-mono text-[11px] tracking-widest uppercase text-ink-3">
            Retrieval practice
          </h2>
          <span className="font-mono text-[10px] text-ink-3">
            decay decides what you review
          </span>
        </div>

        {!sessionId && (
          <p className="text-sm text-vermillion">waiting for session…</p>
        )}

        {sessionId && !exercise && !loading && (
          <button
            onClick={() => void load()}
            className="btn-primary self-start px-5 py-2.5 rounded-lg font-jp text-sm"
          >
            練習 start — exercise from faded concepts
          </button>
        )}

        {loading && (
          <div className="dot-pulse flex gap-1.5 mt-4"><span /><span /><span /></div>
        )}

        {error && (
          <div className="text-vermillion text-xs px-3 py-2 rounded border border-vermillion-dim bg-vermillion-bg">
            ⚠ {error}
          </div>
        )}

        {exercise?.none && (
          <div className="text-center py-16 flex flex-col gap-3 text-ink-3">
            <span className="font-jp text-5xl text-border/80">まだ</span>
            <p className="text-sm">
              Nothing to practice yet — chat first so concepts enter memory.
              Concepts appear here once their Hebbian weights start to fade.
            </p>
          </div>
        )}

        {exercise && !exercise.none && (
          <div className="flex flex-col gap-4">

            {/* meta */}
            <div className="flex flex-wrap items-center gap-2 font-mono text-[10px]">
              <span className="px-2 py-0.5 rounded bg-paper-2 text-ink-2 border border-border/50">
                {EXERCISE_LABELS[exercise.exercise_type] ?? exercise.exercise_type}
              </span>
              {exercise.targets.map(t => (
                <span key={t}
                  className="px-2 py-0.5 rounded bg-paper-1 border border-border/50 text-ink-2 font-jp">
                  {t}
                </span>
              ))}
            </div>

            {/* prompt */}
            <div className="bg-paper-1 border border-border/60 rounded-lg px-4 py-3.5
                            font-jp text-base leading-relaxed whitespace-pre-wrap">
              {exercise.prompt}
            </div>

            {exercise.hint && (
              <div>
                {showHint
                  ? <p className="text-xs text-ink-2 italic">hint: {exercise.hint}</p>
                  : <button onClick={() => setShowHint(true)}
                      className="font-mono text-[10px] text-ink-3 underline underline-offset-2 hover:text-ink-1">
                      show hint
                    </button>}
              </div>
            )}

            {/* answer input / grading result */}
            {!result ? (
              <div className="flex gap-2 items-end">
                <textarea
                  ref={inputRef}
                  value={answer}
                  onChange={e => setAnswer(e.target.value)}
                  onKeyDown={onKey}
                  placeholder="Your answer…"
                  rows={2}
                  disabled={loading || !sessionId}
                  className="flex-1 resize-none px-3 py-2.5 rounded-lg border border-border
                             bg-paper-0 text-ink-0 font-jp text-sm leading-relaxed
                             focus:outline-none focus:border-ink-2 disabled:opacity-40"
                />
                <button
                  onClick={() => void submit()}
                  disabled={!answer.trim() || loading}
                  className="btn-primary h-[52px] px-5 rounded-lg font-jp text-sm"
                >
                  回答
                </button>
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                <div className={[
                  'rounded-lg px-4 py-3 border text-sm leading-relaxed',
                  result.correct
                    ? 'border-[#2d7a4f]/40 bg-[#2d7a4f]/10'
                    : 'border-vermillion-dim bg-vermillion-bg',
                ].join(' ')}>
                  <span className={`font-medium ${result.correct ? 'text-[#2d7a4f]' : 'text-vermillion'}`}>
                    {result.correct ? '○ 正解！' : '× Not quite'}
                  </span>
                  {' '}
                  <span className="text-ink-1">{result.feedback}</span>
                </div>

                {!result.correct && result.answer && (
                  <p className="text-xs text-ink-2">
                    expected: <span className="font-jp">{result.answer}</span>
                  </p>
                )}

                {result.reinforced && (
                  <p className="font-mono text-[10px] text-[#1d9e75]">
                    ⚡ Hebbian edges reinforced — forgetting timer reset
                  </p>
                )}

                <button
                  onClick={() => void load()}
                  disabled={loading}
                  className="btn-primary self-start px-5 py-2.5 rounded-lg font-jp text-sm"
                >
                  次へ next
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Fading queue sidebar */}
      <aside className="w-64 flex-shrink-0 border-l border-border/40 px-5 py-7 overflow-y-auto hidden lg:block">
        <h3 className="font-mono text-[10px] tracking-widest uppercase text-ink-3 mb-4">
          Fading queue
        </h3>
        {fading.length === 0 ? (
          <p className="text-xs text-ink-3">no learned concepts yet</p>
        ) : (
          <ul className="flex flex-col gap-3.5">
            {fading.map(f => (
              <li key={f.node_id} className="flex flex-col gap-1">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-jp text-sm text-ink-0 truncate">{f.label}</span>
                  <span className="font-mono text-[9px] text-ink-3">
                    w={f.avg_w.toFixed(2)}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <FadingBar value={f.fading_score} />
                  <span className="font-mono text-[9px] text-ink-3">
                    fade {f.fading_score.toFixed(2)}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        )}
        <p className="mt-6 text-[10px] text-ink-3 leading-relaxed">
          Ranked by weakness × staleness × how often learned.
          Answering correctly strengthens edges and pushes a concept to the back.
        </p>
      </aside>
    </div>
  )
}
