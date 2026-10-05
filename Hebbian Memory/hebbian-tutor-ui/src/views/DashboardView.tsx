import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { getGraph, getSystemStats, getArmeHistory } from '@/api'
import type { GraphData, GraphNode, SystemStats, ArmeState } from '@/types'
import ForceGraph from '../components/ForceGraph'
import HierarchyGraph from '../components/HierarchyGraph'
import TimelineGraph from '../components/TimelineGraph'
import ArmePanel from '../components/ArmePanel'
import { CONCEPT_COLORS } from '../types'

type GraphView = 'force' | 'hierarchy' | 'timeline'

const VIEWS: { id: GraphView; label: string; jp: string }[] = [
  { id: 'force',     label: 'Force',     jp: '引力' },
  { id: 'hierarchy', label: 'Hierarchy', jp: '階層' },
  { id: 'timeline',  label: 'Timeline',  jp: '時系列' },
]

interface Props {
  /** Client-side filter to only this source_type's nodes/edges (e.g. "sensor"). */
  filterSourceType?: string
  /** Whether to fetch/show the ARM-E history panel. Default true. */
  showArmeHistory?: boolean
}

export default function DashboardView({ filterSourceType, showArmeHistory = true }: Props) {
  const [graphView, setGraphView]     = useState<GraphView>('force')
  const [graphData, setGraphData]     = useState<GraphData | null>(null)
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null)
  const [armeHistory, setArmeHistory] = useState<ArmeState[]>([])
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [minWeight, setMinWeight]     = useState(0.5)
  const [showIsolated, setShowIsolated] = useState(true)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [graph, stats, arme] = await Promise.all([
        getGraph({ minWeight, includeIsolated: showIsolated }),
        getSystemStats(),
        showArmeHistory ? getArmeHistory(30) : Promise.resolve([]),
      ])
      setGraphData(graph)
      setSystemStats(stats)
      setArmeHistory(arme)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [minWeight, showIsolated, showArmeHistory])

  const filteredGraphData = useMemo<GraphData | null>(() => {
    if (!graphData || !filterSourceType) return graphData
    const keep = new Set(
      graphData.nodes.filter(n => n.source_type === filterSourceType).map(n => n.node_id)
    )
    return {
      nodes: graphData.nodes.filter(n => keep.has(n.node_id)),
      edges: graphData.edges.filter(e => keep.has(e.source) && keep.has(e.target)),
    }
  }, [graphData, filterSourceType])

  // Deferred one tick: keeps setState out of the synchronous effect body
  // (react-hooks/set-state-in-effect). Depends on `refresh`, so moving the
  // min-weight slider also refetches automatically.
  useEffect(() => {
    const t = setTimeout(() => void refresh(), 0)
    return () => clearTimeout(t)
  }, [refresh])

  return (
    <div className="flex w-full h-full overflow-hidden">

      {/* Graph panel */}
      <div className="flex-1 flex flex-col border-r border-border overflow-hidden">

        {/* Toolbar */}
        <div className="flex items-center justify-between gap-4 px-4 py-2 border-b border-border/40 bg-paper-0 flex-shrink-0">

          {/* View switcher */}
          <div className="flex gap-0.5 bg-paper-1 border border-border rounded p-0.5">
            {VIEWS.map(v => (
              <button
                key={v.id}
                onClick={() => setGraphView(v.id)}
                className={[
                  'font-mono text-[10px] px-3 py-1 rounded-sm flex items-center gap-1.5 tracking-wide transition-all',
                  graphView === v.id
                    ? 'bg-paper-0 text-ink-0 shadow-sm'
                    : 'text-ink-2 hover:text-ink-1',
                ].join(' ')}
              >
                <span className="font-jp text-[11px] text-ink-3">{v.jp}</span>
                {v.label}
              </button>
            ))}
          </div>

          {/* Controls */}
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 font-mono text-[10px] text-ink-3 cursor-default">
              min w
              <input
                type="range" min="0" max="2" step="0.1"
                value={minWeight}
                onChange={e => setMinWeight(parseFloat(e.target.value))}
                className="w-20 accent-ink-1"
              />
              <code className="text-ink-2 w-5">{minWeight.toFixed(1)}</code>
            </label>
            <label className="flex items-center gap-1.5 font-mono text-[10px] text-ink-3 cursor-pointer">
              <input
                type="checkbox"
                checked={showIsolated}
                onChange={e => setShowIsolated(e.target.checked)}
                className="accent-ink-1"
              />
              isolated
            </label>
            <button
              onClick={() => void refresh()}
              disabled={loading}
              className="btn text-[11px]"
            >
              {loading ? '…' : '↻ refresh'}
            </button>
          </div>
        </div>

        {/* Canvas */}
        <div className="flex-1 relative overflow-hidden bg-paper-0">
          {error && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-ink-3">
              <span className="text-vermillion">⚠ {error}</span>
              <p className="text-xs">Make sure the backend is running on port 8000</p>
            </div>
          )}
          {!filteredGraphData && !error && !loading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-ink-3">
              <span className="font-jp text-6xl text-border/60">記憶</span>
              <p className="text-sm">Click refresh to load the memory graph</p>
            </div>
          )}
          {filteredGraphData && graphView === 'force' && (
            <ForceGraph data={filteredGraphData} onSelectNode={setSelectedNode} selectedNode={selectedNode} />
          )}
          {filteredGraphData && graphView === 'hierarchy' && (
            <HierarchyGraph data={filteredGraphData} onSelectNode={setSelectedNode} selectedNode={selectedNode} />
          )}
          {filteredGraphData && graphView === 'timeline' && (
            <TimelineGraph data={filteredGraphData} onSelectNode={setSelectedNode} selectedNode={selectedNode} />
          )}
        </div>

        {/* Node detail strip */}
        {selectedNode && (
          <div className="relative border-t border-border bg-paper-1 px-4 py-3 flex-shrink-0">
            <button
              onClick={() => setSelectedNode(null)}
              className="absolute top-2 right-3 text-ink-3 hover:text-ink-1 text-lg leading-none px-1"
            >×</button>
            <div className="font-mono text-[9px] uppercase tracking-widest text-ink-3 mb-0.5">
              {selectedNode.concept_type}
            </div>
            <div
              className={`text-lg font-medium mb-1.5 ${selectedNode.source_lang === 'ja' ? 'font-jp' : 'font-serif'}`}
            >
              {selectedNode.label}
            </div>
            <div className="flex gap-4 font-mono text-[10px] text-ink-3 mb-2">
              <span>activations <code className="text-ink-2">{selectedNode.activation_count}</code></span>
              <span>m_t avg <code className="text-ink-2">{selectedNode.mean_m_t?.toFixed(3)}</code></span>
              <span>dominance <code className="text-ink-2">{selectedNode.mean_dominance?.toFixed(3)}</code></span>
            </div>
            {selectedNode.text_summary && (
              <p className="font-serif text-xs text-ink-2 leading-relaxed max-w-2xl">
                {selectedNode.text_summary}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Stats sidebar */}
      <div className="w-56 flex-shrink-0 overflow-y-auto p-3 flex flex-col gap-3">

        {/* System stats */}
        {systemStats && (
          <StatBlock title="System">
            <Stat label="nodes"    value={systemStats.n_nodes} />
            <Stat label="edges"    value={systemStats.n_edges} />
            <Stat label="sessions" value={systemStats.n_sessions} />
            <Stat label="ingests"  value={systemStats.n_ingests} />
          </StatBlock>
        )}

        {showArmeHistory && <ArmePanel history={armeHistory} />}

        {/* Legend */}
        <StatBlock title="Concept types">
          <div className="flex flex-col gap-1 pt-1">
            {Object.entries(CONCEPT_COLORS).map(([type, color]) => (
              <div key={type} className="flex items-center gap-2 text-[11px] text-ink-2">
                <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                {type}
              </div>
            ))}
          </div>
        </StatBlock>

      </div>
    </div>
  )
}

function StatBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border border-border/60 rounded overflow-hidden">
      <div className="font-mono text-[9px] uppercase tracking-widest text-ink-3 px-2 py-1.5 bg-paper-1 border-b border-border/40">
        {title}
      </div>
      <div className="px-2 py-1.5">{children}</div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex justify-between py-0.5 border-b border-border/30 last:border-0">
      <span className="text-[11px] text-ink-3">{label}</span>
      <code className="text-[11px]">{value ?? '—'}</code>
    </div>
  )
}