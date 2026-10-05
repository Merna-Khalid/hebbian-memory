import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { GraphData, GraphNode, GraphEdge } from '@/types'
import { CONCEPT_COLORS } from '../types'

interface Props {
  data:           GraphData
  onSelectNode:   (node: GraphNode | null) => void
  selectedNode:   GraphNode | null
}

export default function ForceGraph({ data, onSelectNode, selectedNode }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) return
    const el     = ref.current
    const width  = el.clientWidth
    const height = el.clientHeight

    d3.select(el).selectAll('*').remove()

    const nodes: GraphNode[] = data.nodes.map(n => ({ ...n }))
    // D3 mutates edge objects to replace src/dst strings with node refs
    const edges = data.edges.map(e => ({ ...e })) as (GraphEdge & {
      source: GraphNode; target: GraphNode
    })[]

    const wExt   = d3.extent(edges, e => e.hebb_weight) as [number, number]
    const wScale = d3.scaleLinear().domain(wExt.every(isFinite) ? wExt : [0.5, 2]).range([0.5, 3.5])
    const wAlpha = d3.scaleLinear().domain(wExt.every(isFinite) ? wExt : [0.5, 2]).range([0.15, 0.7])
    const actExt = d3.extent(nodes, n => n.activation_count) as [number, number]
    const rScale = d3.scaleSqrt().domain(actExt.every(isFinite) ? actExt : [1, 10]).range([5, 18])

    const svg = d3.select(el).append('svg').attr('width', width).attr('height', height)
    const g   = svg.append('g')

    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.2, 4])
        .on('zoom', e => g.attr('transform', e.transform.toString()))
    )

    svg.append('defs').append('marker')
      .attr('id', 'arr').attr('viewBox', '0 -4 8 8')
      .attr('refX', 14).attr('refY', 0)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,-4L8,0L0,4').attr('fill', '#c0bba8')

    const sim = d3.forceSimulation<GraphNode>(nodes)
      .force('link', d3.forceLink<GraphNode, typeof edges[0]>(edges)
        .id(d => d.node_id)
        .distance(d => 80 + 40 / (d.hebb_weight || 1))
        .strength(d => Math.min(0.6, (d.hebb_weight || 1) / 3))
      )
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide<GraphNode>().radius(d => rScale(d.activation_count) + 6))

    const link = g.append('g').selectAll<SVGLineElement, typeof edges[0]>('line')
      .data(edges).join('line')
      .attr('stroke', '#c0bba8')
      .attr('stroke-width', d => wScale(d.hebb_weight))
      .attr('stroke-opacity', d => wAlpha(d.hebb_weight))
      .attr('marker-end', 'url(#arr)')

    const node = g.append('g').selectAll<SVGGElement, GraphNode>('g')
      .data(nodes).join('g')
      .style('cursor', 'pointer')
      .call(
        d3.drag<SVGGElement, GraphNode>()
          .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
          .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y })
          .on('end',   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null })
      )
      .on('click', (e, d) => { e.stopPropagation(); onSelectNode(d) })

    node.append('circle')
      .attr('r', d => rScale(d.activation_count))
      .attr('fill', d => CONCEPT_COLORS[d.concept_type] ?? '#888780')
      .attr('fill-opacity', 0.85)
      .attr('stroke', d => selectedNode?.node_id === d.node_id ? '#1a1814' : 'none')
      .attr('stroke-width', 2)

    node.append('text')
      .text(d => d.label.length > 14 ? d.label.slice(0, 12) + '…' : d.label)
      .attr('dy', d => rScale(d.activation_count) + 11)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('font-family', d => d.source_lang === 'ja' ? "'Noto Serif JP',serif" : "'IBM Plex Mono',monospace")
      .attr('fill', '#5a5749')
      .attr('pointer-events', 'none')

    svg.on('click', () => onSelectNode(null))

    sim.on('tick', () => {
      link
        .attr('x1', d => (d.source as GraphNode).x ?? 0)
        .attr('y1', d => (d.source as GraphNode).y ?? 0)
        .attr('x2', d => (d.target as GraphNode).x ?? 0)
        .attr('y2', d => (d.target as GraphNode).y ?? 0)
      node.attr('transform', d => `translate(${d.x ?? 0},${d.y ?? 0})`)
    })

    return () => { sim.stop(); d3.select(el).selectAll('*').remove() }
  }, [data, selectedNode, onSelectNode])

  return <div ref={ref} className="w-full h-full" />
}