import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { GraphData, GraphNode } from '../types'
import { CONCEPT_COLORS } from '../types'

interface Props {
  data:         GraphData
  onSelectNode: (node: GraphNode | null) => void
  selectedNode: GraphNode | null
}

interface Positioned extends GraphNode {
  _x: number
  _y: number
}

export default function TimelineGraph({ data, onSelectNode, selectedNode }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) return
    const el = ref.current
    const W  = el.clientWidth
    const H  = el.clientHeight
    const margin = { top: 40, right: 40, bottom: 50, left: 40 }
    const iW = W - margin.left - margin.right
    const iH = H - margin.top  - margin.bottom

    d3.select(el).selectAll('*').remove()

    const nodes = [...data.nodes]
      .filter(n => n.created_at)
      .sort((a, b) => a.created_at - b.created_at)

    if (!nodes.length) {
      d3.select(el).append('div')
        .style('display', 'flex')
        .style('align-items', 'center')
        .style('justify-content', 'center')
        .style('height', '100%')
        .style('color', '#8a8679')
        .style('font-size', '13px')
        .text('No timestamped nodes yet')
      return
    }

    const timeExt = d3.extent(nodes, n => n.created_at) as [number, number]
    const xScale  = d3.scaleTime()
      .domain([new Date(timeExt[0] * 1000), new Date(timeExt[1] * 1000)])
      .range([0, iW]).nice()

    const yByBucket: Record<number, number> = {}
    const positioned: Positioned[] = nodes.map(n => {
      const xRaw  = xScale(new Date(n.created_at * 1000))
      const bucket = Math.round(xRaw / 40)
      yByBucket[bucket] = (yByBucket[bucket] ?? 0)
      const row = yByBucket[bucket]
      const y   = iH / 2 + (row % 2 === 0 ? 1 : -1) * Math.ceil(row / 2) * 44
      yByBucket[bucket]++
      return { ...n, _x: xRaw, _y: y }
    })

    const svg = d3.select(el).append('svg').attr('width', W).attr('height', H)
    const g   = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.5, 5])
        .translateExtent([[-iW, -iH], [2 * iW, 2 * iH]])
        .on('zoom', e => g.attr('transform',
          `translate(${margin.left + e.transform.x},${margin.top + e.transform.y}) scale(${e.transform.k})`
        ))
    )

    g.append('line')
      .attr('x1', 0).attr('y1', iH / 2)
      .attr('x2', iW).attr('y2', iH / 2)
      .attr('stroke', '#d0ccba').attr('stroke-width', 1)

    g.append('g')
      .attr('transform', `translate(0,${iH - 10})`)
      .call(d3.axisBottom(xScale)
        .ticks(Math.min(8, nodes.length))
        .tickSize(3)
        .tickFormat(d => d3.timeFormat('%H:%M')(d as Date))
      )
      .call(ax => {
        ax.select('.domain').remove()
        ax.selectAll('line').attr('stroke', '#c0bba8')
        ax.selectAll('text').attr('font-size', 9).attr('font-family', "'IBM Plex Mono',monospace").attr('fill', '#8a8679')
      })

    g.selectAll('.vtick')
      .data(positioned).join('line')
      .attr('x1', d => d._x).attr('y1', iH / 2)
      .attr('x2', d => d._x).attr('y2', d => d._y)
      .attr('stroke', '#d0ccba').attr('stroke-width', 0.8).attr('stroke-dasharray', '2 2')

    const node = g.selectAll<SVGGElement, Positioned>('.ng')
      .data(positioned).join('g')
      .attr('class', 'ng')
      .attr('transform', d => `translate(${d._x},${d._y})`)
      .style('cursor', 'pointer')
      .on('click', (e, d) => { e.stopPropagation(); onSelectNode(d) })

    node.append('circle')
      .attr('r', d => Math.max(4, Math.sqrt(d.activation_count) * 3))
      .attr('fill', d => CONCEPT_COLORS[d.concept_type] ?? '#888780')
      .attr('fill-opacity', 0.85)
      .attr('stroke', d => selectedNode?.node_id === d.node_id ? '#1a1814' : 'none')
      .attr('stroke-width', 2)

    node.append('text')
      .text(d => d.label.length > 12 ? d.label.slice(0, 10) + '…' : d.label)
      .attr('dy', d => d._y < iH / 2 ? -10 : 16)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9)
      .attr('font-family', d => d.source_lang === 'ja' ? "'Noto Serif JP',serif" : "'IBM Plex Mono',monospace")
      .attr('fill', '#5a5749')

    svg.on('click', () => onSelectNode(null))

    return () => { d3.select(el).selectAll('*').remove() }
  }, [data, selectedNode, onSelectNode])

  return <div ref={ref} className="w-full h-full" />
}