import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { GraphData, GraphNode, ConceptType } from '@/types'
import { CONCEPT_COLORS } from '../types'

interface Props {
  data:         GraphData
  onSelectNode: (node: GraphNode | null) => void
  selectedNode: GraphNode | null
}

interface HierNode {
  name:         string
  concept_type?: string
  children?:    HierNode[]
  value?:       number
  // carry original node data for click
  _node?:       GraphNode
}

export default function HierarchyGraph({ data, onSelectNode, selectedNode }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) return
    const el     = ref.current
    const width  = el.clientWidth
    const height = el.clientHeight

    d3.select(el).selectAll('*').remove()

    const byType = d3.group(data.nodes, d => d.concept_type ?? 'event')

    const hierData: HierNode = {
      name: 'root',
      children: Array.from(byType.entries()).map(([type, nodes]) => ({
        name: type,
        children: nodes.map(n => ({
          name:        n.label,
          concept_type: n.concept_type,
          value:       Math.max(1, n.activation_count),
          _node:       n,
        })),
      })),
    }

    const root = d3.hierarchy<HierNode>(hierData)
      .sum(d => d.value ?? 1)
      .sort((a, b) => (b.value ?? 0) - (a.value ?? 0))

    const radius  = Math.min(width, height) / 2 - 60
    const cluster = d3.cluster<HierNode>().size([2 * Math.PI, radius])
    cluster(root)

    const svg = d3.select(el).append('svg').attr('width', width).attr('height', height)
    const g   = svg.append('g').attr('transform', `translate(${width / 2},${height / 2})`)

    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.3, 3])
        .on('zoom', e => g.attr('transform',
          `translate(${width / 2 + e.transform.x},${height / 2 + e.transform.y}) scale(${e.transform.k})`
        ))
    )

    const linkGen = d3.linkRadial<d3.HierarchyPointLink<HierNode>, d3.HierarchyPointNode<HierNode>>()
      .angle(d => d.x ?? 0)
      .radius(d => d.y ?? 0)

    g.append('g').selectAll<SVGPathElement, d3.HierarchyPointLink<HierNode>>('path')
      // d3 types root.links() as HierarchyLink, but cluster() has already
      // assigned x/y — safe to treat as point links
      .data(root.links() as d3.HierarchyPointLink<HierNode>[])
      .join('path')
      .attr('d', linkGen)
      .attr('fill', 'none')
      .attr('stroke', '#d0ccba')
      .attr('stroke-width', 0.8)
      .attr('stroke-opacity', 0.6)

    const node = g.append('g')
      .selectAll<SVGGElement, d3.HierarchyPointNode<HierNode>>('g')
      .data(root.descendants().filter(d => d.depth > 0))
      .join('g')
      .attr('transform', d =>
        `rotate(${((d.x ?? 0) * 180 / Math.PI) - 90}) translate(${d.y ?? 0},0)`)
      .style('cursor', d => d.depth === 2 ? 'pointer' : 'default')
      .on('click', (e, d) => {
        if (d.depth === 2 && d.data._node) {
          e.stopPropagation()
          onSelectNode(d.data._node)
        }
      })

    // Type labels
    node.filter(d => d.depth === 1)
      .append('text')
      .text(d => d.data.name)
      .attr('dy', 4)
      .attr('x', d => ((d.x ?? 0) < Math.PI ? 6 : -6))
      .attr('text-anchor', d => (d.x ?? 0) < Math.PI ? 'start' : 'end')
      .attr('transform', d => (d.x ?? 0) >= Math.PI ? 'rotate(180)' : null)
      .attr('font-size', 9)
      .attr('font-family', "'IBM Plex Mono',monospace")
      .attr('fill', d => CONCEPT_COLORS[d.data.name as ConceptType] ?? '#888780')

    // Leaf circles
    node.filter(d => d.depth === 2)
      .append('circle')
      .attr('r', d => Math.max(3, Math.sqrt(d.data.value ?? 1) * 2.5))
      .attr('fill', d => CONCEPT_COLORS[d.data.concept_type ?? 'event'] ?? '#888780')
      .attr('fill-opacity', 0.8)
      .attr('stroke', d => selectedNode?.node_id === d.data._node?.node_id ? '#1a1814' : 'none')
      .attr('stroke-width', 2)

    // Leaf labels
    node.filter(d => d.depth === 2)
      .append('text')
      .text(d => d.data.name.length > 10 ? d.data.name.slice(0, 9) + '…' : d.data.name)
      .attr('dy', 4)
      .attr('x', d => (d.x ?? 0) < Math.PI ? 8 : -8)
      .attr('text-anchor', d => (d.x ?? 0) < Math.PI ? 'start' : 'end')
      .attr('transform', d => (d.x ?? 0) >= Math.PI ? 'rotate(180)' : null)
      .attr('font-size', 9)
      .attr('font-family', d => d.data._node?.source_lang === 'ja'
        ? "'Noto Serif JP',serif"
        : "'IBM Plex Mono',monospace")
      .attr('fill', '#5a5749')

    svg.on('click', () => onSelectNode(null))

    return () => { d3.select(el).selectAll('*').remove() }
  }, [data, selectedNode, onSelectNode])

  return <div ref={ref} className="w-full h-full" />
}