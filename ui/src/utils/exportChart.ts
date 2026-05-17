/**
 * Export a container element's SVG charts to a PNG file.
 *
 * Collects every <svg> inside `container`, composes them vertically
 * onto an offscreen canvas at `scale`× resolution, prepends an optional
 * title, fills the background, and triggers a browser download.
 */

export interface ExportConfig {
  title:      string
  background: 'dark' | 'light'
  scale?:     number          // pixel-density multiplier, default 2 (retina)
}

const BG = {
  dark:  { fill: '#0a0a0a', text: '#e0e0e0', label: '#666666' },
  light: { fill: '#ffffff', text: '#111111', label: '#888888' },
}

const TITLE_H = 52    // px reserved above charts for the title row
const PAD     = 20    // horizontal padding

export async function downloadAsPng(
  container: HTMLElement,
  config: ExportConfig,
  filename: string,
): Promise<void> {
  const svgs = Array.from(container.querySelectorAll<SVGElement>('svg'))
  if (svgs.length === 0) return

  const scale  = config.scale ?? 2
  const colors = BG[config.background]
  const cRect  = container.getBoundingClientRect()

  const canvasW = Math.round(cRect.width)
  const canvasH = Math.round(cRect.height) + TITLE_H

  const canvas = document.createElement('canvas')
  canvas.width  = canvasW * scale
  canvas.height = canvasH * scale

  const ctx = canvas.getContext('2d')!
  ctx.scale(scale, scale)

  // Background
  ctx.fillStyle = colors.fill
  ctx.fillRect(0, 0, canvasW, canvasH)

  // Title
  if (config.title) {
    ctx.font      = `bold 14px "SF Mono", "Fira Code", monospace`
    ctx.fillStyle = colors.text
    ctx.fillText(config.title, PAD, 28)

    ctx.font      = `11px "SF Mono", "Fira Code", monospace`
    ctx.fillStyle = colors.label
    ctx.fillText('tform — Mars Terraforming Simulator', PAD, 44)
  }

  // Draw each SVG at its position relative to the container
  await Promise.all(svgs.map(async (svg) => {
    const r = svg.getBoundingClientRect()
    const x = Math.round(r.left - cRect.left)
    const y = Math.round(r.top  - cRect.top) + TITLE_H
    await _drawSvgAt(ctx, svg, x, y)
  }))

  // Draw HTML legends (Recharts renders them outside the SVG)
  _drawLegends(ctx, container, cRect, TITLE_H, canvasH)

  canvas.toBlob((blob) => {
    if (!blob) return
    const url = URL.createObjectURL(blob)
    const a   = Object.assign(document.createElement('a'), { href: url, download: filename })
    a.click()
    URL.revokeObjectURL(url)
  }, 'image/png')
}

/** Redraw Recharts HTML legends onto the canvas at their original positions. */
function _drawLegends(
  ctx: CanvasRenderingContext2D,
  container: HTMLElement,
  cRect: DOMRect,
  titleH: number,
  canvasH: number,
): void {
  const wrappers = Array.from(container.querySelectorAll<HTMLElement>('.recharts-legend-wrapper'))
  ctx.save()
  ctx.font = '10px "SF Mono", "Fira Code", monospace'

  for (const wrapper of wrappers) {
    const wRect = wrapper.getBoundingClientRect()
    const baseY = Math.round(wRect.top - cRect.top) + titleH
    if (baseY < 0 || baseY > canvasH) continue

    const items = Array.from(wrapper.querySelectorAll<HTMLElement>('li.recharts-legend-item'))
    if (items.length === 0) continue

    let ix = Math.round(wRect.left - cRect.left) + 8
    for (const item of items) {
      const pathEl = item.querySelector<SVGElement>('path, line')
      const color  = pathEl?.getAttribute('stroke') || '#888'
      const isDash = !!(pathEl?.getAttribute('stroke-dasharray'))

      ctx.strokeStyle = color
      ctx.lineWidth   = 1.5
      if (isDash) ctx.setLineDash([4, 2])
      ctx.beginPath()
      ctx.moveTo(ix,      baseY + 6)
      ctx.lineTo(ix + 14, baseY + 6)
      ctx.stroke()
      ctx.setLineDash([])

      const label = item.querySelector<HTMLElement>('.recharts-legend-item-text')?.textContent || ''
      ctx.fillStyle = '#888'
      ctx.fillText(label, ix + 18, baseY + 10)
      ix += Math.max(56, ctx.measureText(label).width + 34)
    }
  }

  ctx.restore()
}

/** Serialize one SVG element and draw it onto ctx at (x, y). */
async function _drawSvgAt(
  ctx: CanvasRenderingContext2D,
  svg: SVGElement,
  x: number,
  y: number,
): Promise<void> {
  let src = new XMLSerializer().serializeToString(svg)

  // Ensure the root <svg> has an explicit xmlns (required for img.src)
  if (!src.includes('xmlns=')) {
    src = src.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
  }

  const blob = new Blob([src], { type: 'image/svg+xml' })
  const url  = URL.createObjectURL(blob)

  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => { ctx.drawImage(img, x, y); URL.revokeObjectURL(url); resolve() }
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('SVG render failed')) }
    img.src = url
  })
}
