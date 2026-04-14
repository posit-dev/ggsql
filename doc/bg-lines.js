// bg-lines.js — Animated background line chart drawings
(function () {
  'use strict';

  const GRID = 40;
  const MAX_OPACITY = 0.25;
  const POINT_R = 2.5;
  const LINE_W = 1.25;
  const MAX_SERIES = 4;
  const DRAW_DURATION = 12000;   // ms to draw full line
  const HOLD_MS = 1500;
  const FADE_MS = 2500;
  const SPAWN_MIN = 500;
  const SPAWN_MAX = 2000;

  const COLORS = [
    [0, 95, 115],    // darkteal
    [10, 147, 150],  // teal
    [148, 210, 189], // lightteal
  ];

  const canvas = document.getElementById('bg-lines');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // --- Helpers ---
  function rand(lo, hi) { return lo + Math.random() * (hi - lo); }
  function randInt(lo, hi) { return Math.floor(rand(lo, hi + 1)); }
  function pick(arr) { return arr[randInt(0, arr.length - 1)]; }
  function rgba(c, a) { return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; }

  // --- Resize ---
  let W, H;
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  let resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 100);
  });
  resize();

  // --- Line generation ---
  const STEP_X = 40;  // fixed sampling interval along x

  function generatePoints() {
    const n = Math.ceil(W / STEP_X) + 1;
    const startY = rand(H * 0.15, H * 0.85);
    const volatility = rand(25, 60);

    const pts = [];
    let y = startY;
    for (let i = 0; i < n; i++) {
      pts.push({ x: i * STEP_X, y: y });
      y += (Math.random() - 0.5) * 2 * volatility;
      y = Math.max(GRID * 2, Math.min(H - GRID * 2, y));
    }
    return pts;
  }

  // --- Series ---
  function createSeries() {
    return {
      pts: generatePoints(),
      color: pick(COLORS),
      state: 'drawing',       // drawing | holding | fading | dead
      progress: 0,            // fractional segment index
      opacity: MAX_OPACITY,
      stateStart: 0,
    };
  }

  function updateSeries(s, now, dt) {
    switch (s.state) {
      case 'drawing':
        s.progress += (s.pts.length - 1) * (dt / DRAW_DURATION);
        if (s.progress >= s.pts.length - 1) {
          s.progress = s.pts.length - 1;
          s.state = 'holding';
          s.stateStart = now;
        }
        break;
      case 'holding':
        if (now - s.stateStart > HOLD_MS) {
          s.state = 'fading';
          s.stateStart = now;
        }
        break;
      case 'fading': {
        const t = (now - s.stateStart) / FADE_MS;
        s.opacity = MAX_OPACITY * (1 - t);
        if (s.opacity <= 0) {
          s.opacity = 0;
          s.state = 'dead';
        }
        break;
      }
    }
  }

  function drawSeries(s) {
    if (s.opacity <= 0) return;
    const col = rgba(s.color, s.opacity);
    const full = Math.floor(s.progress);

    // Line
    ctx.strokeStyle = col;
    ctx.lineWidth = LINE_W;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(s.pts[0].x, s.pts[0].y);
    for (let i = 1; i <= full; i++) {
      ctx.lineTo(s.pts[i].x, s.pts[i].y);
    }
    // Partial segment
    const frac = s.progress - full;
    if (frac > 0 && full + 1 < s.pts.length) {
      const a = s.pts[full];
      const b = s.pts[full + 1];
      ctx.lineTo(a.x + (b.x - a.x) * frac, a.y + (b.y - a.y) * frac);
    }
    ctx.stroke();

    // Points
    ctx.fillStyle = col;
    for (let i = 0; i <= full; i++) {
      ctx.beginPath();
      ctx.arc(s.pts[i].x, s.pts[i].y, POINT_R, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // --- Main loop ---
  let series = [];
  let nextSpawn = 0;
  let prev = 0;
  let paused = false;

  function tick(now) {
    if (paused) { requestAnimationFrame(tick); return; }
    const dt = prev ? now - prev : 16;
    prev = now;

    ctx.clearRect(0, 0, W, H);

    // Spawn
    if (series.length < MAX_SERIES && now > nextSpawn) {
      series.push(createSeries());
      nextSpawn = now + rand(SPAWN_MIN, SPAWN_MAX);
    }

    // Update & draw
    for (const s of series) {
      updateSeries(s, now, dt);
      drawSeries(s);
    }

    // Prune
    series = series.filter(function (s) { return s.state !== 'dead'; });

    requestAnimationFrame(tick);
  }

  // --- Visibility ---
  document.addEventListener('visibilitychange', function () {
    paused = document.hidden;
    if (!paused) prev = 0; // reset dt to avoid jump
  });

  // --- Start ---
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    requestAnimationFrame(tick);
  }
})();
