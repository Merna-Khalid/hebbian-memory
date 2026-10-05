import React, { useState, useRef, useEffect, useCallback } from 'react'
import { sendMessage } from '../api'
import type { Message } from '@/types'
import ArmeBadge from '@/components/ArmeBadge'
import ConceptTags from '@/components/ConceptTags'

function MessageRow({ msg }: { msg: Message }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`flex flex-col gap-1 ${isUser ? 'items-end' : 'items-start'}`}>
      <span className="font-mono text-[10px] text-ink-3 px-1">
        {isUser ? 'あなた' : 'LFM'}
      </span>

      <div
        className={[
          'max-w-[88%] px-3.5 py-2.5 rounded-lg leading-relaxed',
          isUser
            ? 'bg-ink-0 text-paper-0 rounded-br-sm'
            : 'bg-paper-1 border border-border/60 text-ink-0 font-serif text-sm rounded-bl-sm',
          msg.lang === 'ja' ? 'font-jp text-base' : '',
        ].join(' ')}
      >
        {msg.content}
      </div>

      {!isUser && msg.arme  && <ArmeBadge arme={msg.arme} />}
      {!isUser && msg.concepts && msg.concepts.length > 0 && (
        <ConceptTags concepts={msg.concepts} />
      )}
    </div>
  )
}

function ThinkingIndicator() {
  return (
    <div className="flex flex-col items-start gap-1">
      <span className="font-mono text-[10px] text-ink-3 px-1">LFM</span>
      <div className="bg-paper-1 border border-border/60 rounded-lg rounded-bl-sm px-4 py-3">
        <div className="dot-pulse flex gap-1.5">
          <span /><span /><span />
        </div>
      </div>
    </div>
  )
}

// ── Main view ──────────────────────────────────────────────────────

interface Props { sessionId: string | null }

export default function ChatView({ sessionId }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const bottomRef               = useRef<HTMLDivElement>(null)
  const inputRef                = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = useCallback(async () => {
    const text = input.trim()
    if (!text || loading || !sessionId) return
    setInput('')
    setError(null)
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setLoading(true)
    try {
      const data = await sendMessage(text, sessionId)
      setMessages(prev => [...prev, {
        role:     'assistant',
        content:  data.response,
        arme:     data.arme ?? undefined,
        concepts: data.concepts,
        lang:     data.lang,
      }])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }, [input, loading, sessionId])

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send() }
  }

  return (
    <div className="flex flex-col w-full max-w-2xl mx-auto h-full">

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-7 flex flex-col gap-6">
        {messages.length === 0 && (
          <div className="m-auto text-center text-ink-3 flex flex-col gap-3">
            <span className="font-jp text-5xl text-border/80">日本語</span>
            <p className="text-sm">Start a conversation. The system will remember what you learn.</p>
          </div>
        )}
        {messages.map((msg, i) => <MessageRow key={i} msg={msg} />)}
        {loading && <ThinkingIndicator />}
        {error && (
          <div className="text-vermillion text-xs px-3 py-2 rounded border border-vermillion-dim bg-vermillion-bg">
            ⚠ {error}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2 px-6 pb-5 pt-3 border-t border-border/40 bg-paper-0 items-end">
        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKey}
          placeholder="英語か日本語でどうぞ…"
          rows={2}
          disabled={!sessionId}
          className="flex-1 resize-none px-3 py-2.5 rounded-lg border border-border
                     bg-paper-0 text-ink-0 font-mono text-sm leading-relaxed
                     focus:outline-none focus:border-ink-2 disabled:opacity-40
                     min-h-[52px] max-h-40"
        />
        <button
          onClick={() => void send()}
          disabled={!input.trim() || loading || !sessionId}
          className="btn-primary h-[52px] px-5 rounded-lg font-jp text-sm tracking-wide"
        >
          送信
        </button>
      </div>

    </div>
  )
}