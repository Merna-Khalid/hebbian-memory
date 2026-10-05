import React, { useState } from 'react'
import { useSession } from './hooks/useSession'
import { useSecondBrainSession } from './hooks/useSecondBrainSession'
import ChatView from './views/ChatView'
import PracticeView from './views/PracticeView'
import DashboardView from './views/DashboardView'
import SecondBrainView from './views/SecondBrainView'

type Tab = 'chat' | 'practice' | 'dashboard' | 'second-brain'

const TAB_LABELS: Record<Tab, React.ReactNode> = {
  'chat':         <><span className="font-jp mr-1.5">学習</span>Chat</>,
  'practice':     <><span className="font-jp mr-1.5">練習</span>Practice</>,
  'dashboard':    <><span className="font-jp mr-1.5">記憶</span>Memory</>,
  'second-brain': <><span className="mr-1.5">🧠</span>Second Brain</>,
}

export default function App() {
  const [tab, setTab] = useState<Tab>('chat')
  const { sessionId, error } = useSession()
  const { sessionId: sbSessionId } = useSecondBrainSession()

  return (
    <div className="flex flex-col h-full bg-paper-0">

      {/* Header */}
      <header className="flex items-center gap-6 px-6 h-13 border-b border-border bg-paper-0 flex-shrink-0">

        {/* Wordmark */}
        <div className="flex items-baseline gap-2.5 flex-shrink-0">
          <span className="font-jp text-xl font-medium tracking-wide">記憶</span>
          <span className="font-mono text-[10px] text-ink-3 tracking-widest uppercase">
            Hebbian Japanese Tutor
          </span>
        </div>

        {/* Tab switcher */}
        <nav className="flex gap-0.5 bg-paper-1 border border-border rounded p-0.5">
          {(['chat', 'practice', 'dashboard', 'second-brain'] as const).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={[
                'font-mono text-[11px] px-3.5 py-1 rounded-sm tracking-wide transition-all',
                tab === t
                  ? 'bg-paper-0 text-ink-0 shadow-sm'
                  : 'text-ink-2 hover:text-ink-1',
              ].join(' ')}
            >
              {TAB_LABELS[t]}
            </button>
          ))}
        </nav>

        {/* Session badge */}
        <div className="ml-auto font-mono text-[10px] text-ink-3">
          {error
            ? <span className="text-vermillion">⚠ {error}</span>
            : sessionId
              ? <>session <code className="bg-paper-2 px-1.5 py-0.5 rounded text-ink-2">{sessionId.slice(0, 8)}</code></>
              : <span className="text-vermillion-dim">connecting…</span>
          }
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 overflow-hidden flex">
        {tab === 'chat'          && <ChatView sessionId={sessionId} />}
        {tab === 'practice'      && <PracticeView sessionId={sessionId} />}
        {tab === 'dashboard'     && <DashboardView />}
        {tab === 'second-brain'  && <SecondBrainView sessionId={sbSessionId} />}
      </main>
    </div>
  )
}