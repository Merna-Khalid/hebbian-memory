// types.ts — shared types for the Hebbian tutor UI.
// These mirror the JSON shapes served by server.py / TutorEngine.

// ── Concept types (mirror core/agents/concept_extractor.py) ─────────

export type ConceptType =
  | 'vocabulary' | 'grammar' | 'fact' | 'entity'
  | 'procedure' | 'example' | 'cultural' | 'question' | 'event'

export const CONCEPT_COLORS: Record<string, string> = {
  vocabulary: '#2d7a4f',
  grammar:    '#378ADD',
  fact:       '#888780',
  entity:     '#c84b2f',
  procedure:  '#7c5cbf',
  example:    '#c9a227',
  cultural:   '#d85a30',
  question:   '#b83280',
  event:      '#5a5749',
  // second-brain domain (fact/entity/question/event shared above)
  idea:       '#1d9e75',
  task:       '#378ADD',
  plan:       '#7c5cbf',
  feeling:    '#c84b2f',
}

// ── ARM-E ────────────────────────────────────────────────────────────

export type Quadrant = 'Q1' | 'Q2' | 'Q3' | 'Q4'

export const QUADRANT_META: Record<Quadrant, { label: string; color: string }> = {
  Q1: { label: 'curiosity', color: '#2d7a4f' },
  Q2: { label: 'anxiety',   color: '#c84b2f' },
  Q3: { label: 'boredom',   color: '#888780' },
  Q4: { label: 'calm',      color: '#378ADD' },
}

export interface ArmeState {
  quadrant:      Quadrant
  m_t:           number
  gated:         boolean
  valence:       number
  arousal:       number
  dominance:     number
  delta_w_mean?: number
  r_t?:          number
  timestamp?:    number
}

// ── Chat ─────────────────────────────────────────────────────────────

export interface ExtractedConcept {
  label:       string
  type:        string
  description: string
  lang:        string
}

export interface RetrievedConcept {
  node_id:      string
  label:        string
  text_summary: string
  concept_type: string
  score:        number
}

export interface Message {
  role:      'user' | 'assistant'
  content:   string
  arme?:     ArmeState
  concepts?: ExtractedConcept[]
  lang?:     string
}

export interface ChatResponse {
  response:  string
  arme:      ArmeState | null
  concepts:  ExtractedConcept[]
  lang:      string
  retrieved: RetrievedConcept[]
}

// ── Graph ────────────────────────────────────────────────────────────

export interface GraphNode {
  node_id:          string
  label:            string
  concept_type:     string
  source_type?:     string
  source_lang:      string
  activation_count: number
  mean_m_t?:        number
  mean_dominance?:  number
  text_summary?:    string
  created_at:       number
  // d3 force simulation mutates these onto the node objects
  x?:  number
  y?:  number
  fx?: number | null
  fy?: number | null
}

export interface GraphEdge {
  source:        string
  target:        string
  hebb_weight:   number
  eligibility?:  number
  causal_score?: number
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

// ── Stats ────────────────────────────────────────────────────────────

export interface SystemStats {
  n_nodes:    number
  n_edges:    number
  n_sessions: number
  n_ingests:  number
}

// ── Retrieval practice (forgetting-curve exercises) ─────────────────

export interface FadingConcept {
  node_id:      string
  label:        string
  concept_type: string
  text_summary: string
  avg_w:        number
  degree:       number
  strength:     number    // 0..1 — avg hippocampal edge weight / w_ceil
  staleness:    number    // 0..1 — time since last activation
  fading_score: number
}

export interface PracticeFadingTarget {
  label: string
  score: number
}

export type ExerciseType =
  | 'translation' | 'fill_in_blank' | 'sentence_production' | 'form_transform'

export interface PracticeExercise {
  none:          boolean | null   // true when nothing to practice yet
  exercise_type: ExerciseType | string
  prompt:        string
  hint:          string
  targets:       string[]
  fading:        PracticeFadingTarget[]
}

export interface PracticeResult {
  correct:    boolean
  feedback:   string
  answer:     string
  targets:    string[]
  reinforced: boolean      // true when memory edges were strengthened
}

// ── Second brain (simulated BCI) ────────────────────────────────────

export interface PhysioSample {
  hr_bpm:    number | null
  attention: number
  quadrant?: Quadrant
  m_t?:      number
  timestamp: number
}

export interface SecondBrainEvent {
  done:            boolean
  scenario_label?: string
  utterance?:      string
  hr_bpm?:         number | null
  attention?:      number
  arme?:           ArmeState | null
  concepts?:       ExtractedConcept[]
}
