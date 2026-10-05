import type {
  ChatResponse, GraphData, SystemStats, ArmeState, SecondBrainEvent,
  FadingConcept, PracticeExercise, PracticeResult,
} from './types'

const BASE = '/api'

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init.headers },
    ...init,
  })
  if (!res.ok) {
    const msg = await res.text()
    throw new Error(`${res.status}: ${msg}`)
  }
  return res.json() as Promise<T>
}

// ── Session ────────────────────────────────────────────────────────

export const startSession = () =>
  request<{ session_id: string }>('/session/start', { method: 'POST' })

export const endSession = (sessionId: string) =>
  request<{ ok: boolean }>('/session/end', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
  })

// ── Chat ───────────────────────────────────────────────────────────

export const sendMessage = (message: string, sessionId: string) =>
  request<ChatResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify({ message, session_id: sessionId }),
  })

// ── Graph ──────────────────────────────────────────────────────────

export interface GraphParams {
  minWeight?:       number
  limit?:           number
  layer?:           string
  includeIsolated?: boolean
}

export const getGraph = (params: GraphParams = {}) => {
  const q = new URLSearchParams({
    min_weight:       String(params.minWeight ?? 0.5),
    limit:            String(params.limit    ?? 200),
    layer:            params.layer           ?? 'hippocampal',
    include_isolated: String(params.includeIsolated ?? true),
  })
  return request<GraphData>(`/graph?${q}`)
}

export const getNodeDetail = (nodeId: string) =>
  request<GraphData['nodes'][0]>(`/graph/node/${nodeId}`)

// ── Stats ──────────────────────────────────────────────────────────

export const getSystemStats  = () => request<SystemStats>('/stats/system')
export const getArmeHistory  = (limit = 50) =>
  request<ArmeState[]>(`/stats/arme?limit=${limit}`)

// ── Retrieval practice (forgetting-curve exercises) ────────────────

export const getFadingConcepts = (limit = 10) =>
  request<FadingConcept[]>(`/practice/fading?limit=${limit}`)

export const nextPractice = (sessionId: string) =>
  request<PracticeExercise>('/practice/next', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
  })

export const submitPracticeAnswer = (sessionId: string, answer: string) =>
  request<PracticeResult>('/practice/answer', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, answer }),
  })

// ── Second brain (simulated BCI) ──────────────────────────────────────

export const startSecondBrainSession = () =>
  request<{ session_id: string }>('/second-brain/session/start', { method: 'POST' })

export const endSecondBrainSession = (sessionId: string) =>
  request<{ ok: boolean }>('/second-brain/session/end', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
  })

export const nextSecondBrainEvent = (sessionId: string) =>
  request<SecondBrainEvent>('/second-brain/next', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
  })

export const resetSecondBrainStream = () =>
  request<{ ok: boolean }>('/second-brain/reset', { method: 'POST' })