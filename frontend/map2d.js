// 2D map: base image + overlay + escape arrows, with pan/zoom and click picking.

const TIME_COLORS = [[0, '#2f9d5e'], [5, '#2f9d5e'], [10, '#d9b21f'], [15, '#e07b1a'], [30, '#c93a3a']];

export function timeColor(min) {
  for (let i = TIME_COLORS.length - 1; i >= 0; i--) if (min >= TIME_COLORS[i][0]) return TIME_COLORS[i][1];
  return TIME_COLORS[0][1];
}

export function drawVectors(ctx, W, H, data, scale = 1) {
  const { arrows = [], zones = [], bottlenecks = [], route = null, profile = null } = data;
  ctx.lineCap = 'round';
  for (const [u0, v0, u1, v1, t] of arrows) {
    const x0 = u0 * W, y0 = v0 * H, x1 = u1 * W, y1 = v1 * H;
    const ang = Math.atan2(y1 - y0, x1 - x0);
    const head = Math.max(5, Math.hypot(x1 - x0, y1 - y0) * 0.45);
    const path = () => {
      ctx.beginPath();
      ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
      ctx.moveTo(x1, y1); ctx.lineTo(x1 - head * Math.cos(ang - 0.5), y1 - head * Math.sin(ang - 0.5));
      ctx.moveTo(x1, y1); ctx.lineTo(x1 - head * Math.cos(ang + 0.5), y1 - head * Math.sin(ang + 0.5));
    };
    ctx.strokeStyle = 'rgba(0,0,0,.75)'; ctx.lineWidth = 4.5 * scale; path(); ctx.stroke();
    ctx.strokeStyle = timeColor(t); ctx.lineWidth = 2.2 * scale; path(); ctx.stroke();
  }
  if (route && route.length > 1) {
    ctx.beginPath();
    route.forEach(([u, v], i) => (i ? ctx.lineTo(u * W, v * H) : ctx.moveTo(u * W, v * H)));
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 8 * scale; ctx.stroke();
    ctx.strokeStyle = '#e01e78'; ctx.lineWidth = 4.5 * scale; ctx.stroke();
    const [su, sv] = route[0];
    ctx.beginPath(); ctx.arc(su * W, sv * H, 7 * scale, 0, 7);
    ctx.fillStyle = '#e01e78'; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2 * scale; ctx.stroke();
  }
  if (profile && profile.length === 2) {
    ctx.beginPath(); ctx.moveTo(profile[0][0] * W, profile[0][1] * H); ctx.lineTo(profile[1][0] * W, profile[1][1] * H);
    ctx.setLineDash([8 * scale, 6 * scale]); ctx.strokeStyle = '#fff'; ctx.lineWidth = 3 * scale; ctx.stroke(); ctx.setLineDash([]);
  }
  const r = 13 * scale;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.font = `700 ${13 * scale}px system-ui, sans-serif`;
  for (const z of zones) {
    const x = z.u * W, y = z.v * H;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7);
    ctx.fillStyle = '#fff'; ctx.fill();
    ctx.lineWidth = 3 * scale; ctx.strokeStyle = z.overloaded ? '#c93a3a' : (z.kind === 'Refuge building' ? '#119fb0' : '#2f9d5e'); ctx.stroke();
    ctx.fillStyle = '#17201b'; ctx.fillText(z.name, x, y + 0.5);
  }
  for (const b of bottlenecks) {
    const x = b.u * W, y = b.v * H;
    ctx.beginPath(); ctx.arc(x, y, r * 0.9, 0, 7);
    ctx.fillStyle = '#c93a3a'; ctx.fill(); ctx.lineWidth = 2 * scale; ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.fillText('!', x, y + 0.5);
  }
}

export class MapView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.base = null; this.overlay = null; this.overlayOpacity = 0.85;
    this.vectors = {};
    this.zoom = 1; this.cx = 0.5; this.cy = 0.5;
    this.onClick = null; this.onHover = null;
    this._bind();
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  get W() { return this.base ? this.base.width : 1; }
  get H() { return this.base ? this.base.height : 1; }

  setBase(img) { const first = !this.base; this.base = img; if (first) this.fit(); this.draw(); }
  setOverlay(img, opacity = 0.85) { this.overlay = img; this.overlayOpacity = opacity; this.draw(); }
  setVectors(v) { this.vectors = { ...this.vectors, ...v }; this.draw(); }

  fit() {
    const r = this.canvas.getBoundingClientRect();
    this.zoom = Math.min(r.width / this.W, r.height / this.H) * 0.96 || 1;
    this.cx = 0.5; this.cy = 0.5; this.draw();
  }
  zoomBy(f, px, py) {
    const r = this.canvas.getBoundingClientRect();
    px ??= r.width / 2; py ??= r.height / 2;
    const [u, v] = this.toUV(px, py);
    this.zoom = Math.min(Math.max(this.zoom * f, 0.05), 40);
    const [u2, v2] = this.toUV(px, py);
    this.cx += u - u2; this.cy += v - v2;
    this.draw();
  }
  toUV(px, py) {
    const r = this.canvas.getBoundingClientRect();
    return [this.cx + (px - r.width / 2) / (this.zoom * this.W), this.cy + (py - r.height / 2) / (this.zoom * this.H)];
  }

  draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const r = c.getBoundingClientRect();
    if (c.width !== Math.round(r.width * dpr) || c.height !== Math.round(r.height * dpr)) {
      c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
    }
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    if (!this.base) return;
    const s = this.zoom * dpr;
    ctx.setTransform(s, 0, 0, s, c.width / 2 - this.cx * this.W * s, c.height / 2 - this.cy * this.H * s);
    ctx.imageSmoothingEnabled = this.zoom < 2;
    ctx.drawImage(this.base, 0, 0, this.W, this.H);
    if (this.overlay) {
      ctx.globalAlpha = this.overlayOpacity;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.overlay, 0, 0, this.W, this.H);
      ctx.globalAlpha = 1;
    }
    drawVectors(ctx, this.W, this.H, this.vectors, 1 / this.zoom);
  }

  composite(maxDim = 2048) {
    const k = Math.min(1, maxDim / Math.max(this.W, this.H));
    const cv = document.createElement('canvas');
    cv.width = Math.round(this.W * k); cv.height = Math.round(this.H * k);
    const ctx = cv.getContext('2d');
    ctx.drawImage(this.base, 0, 0, cv.width, cv.height);
    if (this.overlay) { ctx.globalAlpha = this.overlayOpacity; ctx.drawImage(this.overlay, 0, 0, cv.width, cv.height); ctx.globalAlpha = 1; }
    drawVectors(ctx, cv.width, cv.height, this.vectors, Math.max(1, cv.width / 900));
    return cv;
  }

  _bind() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, y: e.clientY, cx: this.cx, cy: this.cy, moved: false };
    });
    c.addEventListener('pointermove', (e) => {
      const r = c.getBoundingClientRect();
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.hypot(dx, dy) > 4) drag.moved = true;
        if (drag.moved) {
          this.cx = drag.cx - dx / (this.zoom * this.W);
          this.cy = drag.cy - dy / (this.zoom * this.H);
          this.draw();
        }
      } else if (this.onHover && this.base) {
        const [u, v] = this.toUV(e.clientX - r.left, e.clientY - r.top);
        this.onHover(u, v, e.clientX - r.left, e.clientY - r.top);
      }
    });
    c.addEventListener('pointerup', (e) => {
      if (drag && !drag.moved && this.onClick && this.base) {
        const r = c.getBoundingClientRect();
        const [u, v] = this.toUV(e.clientX - r.left, e.clientY - r.top);
        if (u >= 0 && u <= 1 && v >= 0 && v <= 1) this.onClick(u, v);
      }
      drag = null;
    });
    c.addEventListener('pointerleave', () => this.onHover && this.onHover(null));
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      this.zoomBy(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
  }
}
