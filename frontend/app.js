import { MapView } from './map2d.js';

const $ = (id) => document.getElementById(id);
const api = async (path, opts = {}) => {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch { /* not json */ }
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return res;
};
const loadImage = (url) => new Promise((ok, fail) => {
  const img = new Image();
  img.onload = () => ok(img); img.onerror = fail; img.src = url;
});
const fmt = (n, d = 1) => (n === null || n === undefined || Number.isNaN(n) ? '–' : Number(n).toLocaleString(undefined, { maximumFractionDigits: d }));

const state = {
  hazards: {}, job: null, hazard: 'flood', level: 3, layer: 'image', tool: 'route',
  scenario: null, route: null, profilePts: [], view: 'map', three: null, lastRoutePoint: null,
};
const map = new MapView($('mapCanvas'));

// ------------------------------------------------------------------ setup
async function init() {
  try {
    const h = await (await api('/api/health')).json();
    state.hazards = h.hazards;
    const b = $('modelBadge');
    b.textContent = `v${h.version} · ` + (h.model_is_ai ? `AI: ${h.model}` : 'No AI model installed: demo heuristic');
    b.classList.toggle('fallback', !h.model_is_ai);
    configureSlider();
    const d = await (await api('/api/demos')).json();
    $('namchiBtn').classList.toggle('hidden', !d.namchi);
    await refreshHistory();
    const want = new URLSearchParams(location.search).get('job');
    if (want) openJob(want);
  } catch (e) {
    $('modelBadge').textContent = 'Server offline';
  }
}

// ------------------------------------------------------------------ previous results (?job=<id>)
async function refreshHistory() {
  const list = (await (await api('/api/jobs')).json()).filter((j) => j.status === 'done');
  $('historyBox').classList.toggle('hidden', !list.length);
  $('history').innerHTML = '<option value="">Open a previous result…</option>' + list.map((j) =>
    `<option value="${j.id}">${escapeHtml(j.name)} · ${j.created.replace('T', ' ').slice(0, 16)}</option>`).join('');
}
async function openJob(id) {
  try {
    const meta = await (await api(`/api/jobs/${id}`)).json();
    if (meta.status === 'done') await onReady(meta);
  } catch (e) {
    $('loadError').textContent = `Could not open that result: ${e.message}`;
    $('loadError').classList.remove('hidden');
  }
}
$('history').addEventListener('change', () => { if ($('history').value) openJob($('history').value); });

document.querySelectorAll('[data-set-mode]').forEach((btn) => btn.addEventListener('click', () => {
  document.body.dataset.mode = btn.dataset.setMode;
  document.querySelectorAll('[data-set-mode]').forEach((b) => b.classList.toggle('active', b === btn));
  if (btn.dataset.setMode === 'guided') { setTool('route'); setLayer('image'); }
}));

// ------------------------------------------------------------------ step 1: load
const imageFile = $('imageFile');
imageFile.addEventListener('change', () => {
  $('runBtn').disabled = !imageFile.files.length;
  $('dropText').textContent = imageFile.files.length ? imageFile.files[0].name : 'Drop PNG / JPG / GeoTIFF here';
});
const drop = $('drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length) { imageFile.files = e.dataTransfer.files; imageFile.dispatchEvent(new Event('change')); }
});

function numberOrNull(id) { const v = $(id).value; return v === '' ? null : Number(v); }

$('runBtn').addEventListener('click', async () => {
  const fd = new FormData();
  fd.append('image', imageFile.files[0]);
  if ($('demFile').files.length) fd.append('dem', $('demFile').files[0]);
  if ($('gcpFile').files.length) fd.append('gcps', $('gcpFile').files[0]);
  if ($('osmFile').files.length) fd.append('osm', $('osmFile').files[0]);
  const acquired = $('acquired').value;
  fd.append('params', JSON.stringify({
    gsd: numberOrNull('gsd'), sun_elevation: numberOrNull('sunEl'), sun_azimuth: numberOrNull('sunAz'),
    acquired: acquired ? `${acquired}:00Z` : null, geoid_offset_m: numberOrNull('geoid'),
    building_height_prior_m: numberOrNull('prior'), off_nadir: numberOrNull('offNadir'),
    view_azimuth: numberOrNull('viewAz'), dem_kind: $('demKind').value || null, osm_fetch: $('osmFetch').checked,
  }));
  startJob(() => api('/api/jobs', { method: 'POST', body: fd }));
});
$('demoBtn').addEventListener('click', () => startJob(() => api('/api/demo', { method: 'POST' })));
$('namchiBtn').addEventListener('click', () => startJob(() => api('/api/demo?scene=namchi', { method: 'POST' })));

async function startJob(request) {
  $('loadError').classList.add('hidden');
  $('progress').classList.remove('hidden');
  $('runBtn').disabled = true; $('demoBtn').disabled = true;
  try {
    let meta = await (await request()).json();
    while (meta.status === 'queued' || meta.status === 'running') {
      $('progressBar').style.width = `${Math.round(meta.progress * 100)}%`;
      $('progressMsg').textContent = meta.message;
      await new Promise((r) => setTimeout(r, 600));
      meta = await (await api(`/api/jobs/${meta.id}`)).json();
    }
    if (meta.status === 'error') throw new Error(meta.error);
    $('progressBar').style.width = '100%';
    $('progressMsg').textContent = 'Done';
    await onReady(meta);
    refreshHistory();
  } catch (e) {
    $('loadError').textContent = `Could not process: ${e.message}`;
    $('loadError').classList.remove('hidden');
  } finally {
    $('runBtn').disabled = !imageFile.files.length; $('demoBtn').disabled = false;
  }
}

// ------------------------------------------------------------------ step 2: model
async function onReady(meta) {
  state.job = meta; state.scenario = null; state.route = null; state.profilePts = []; state.lastRoutePoint = null;
  state.demOnly = false; $('demOnlyBtn').classList.remove('on');
  $('demOnlyBtn').classList.toggle('hidden', !meta.dem);
  history.replaceState(null, '', `?job=${meta.id}`);
  $('empty').classList.add('hidden');
  ['stepModel', 'stepHazard', 'stepExpert'].forEach((id) => $(id).classList.remove('hidden'));
  renderModelCard(meta);
  renderValidation(meta.validation);
  setExportLinks();
  state.layer = 'image';
  document.querySelectorAll('#layerSeg button').forEach((b) => b.classList.toggle('active', b.dataset.layer === 'image'));
  map.base = null;
  map.setBase(await loadImage(layerUrl('image')));
  state.three = null;  // heightmap reloaded lazily for the new job
  await runScenario();
  if (state.view === '3d') await show3d();
}

function layerUrl(name) { return `/api/jobs/${state.job.id}/layer/${name}.png?t=${Date.now()}`; }

const MODE_TEXT = {
  absolute: ['Absolute DSM: metres above sea level', ''],
  above_ground: ['Heights in metres above local ground', ''],
  terrain_only: ['Terrain in metres; building heights relative', 'relative'],
  relative: ['Relative DSM (0–1): no metres claimed', 'relative'],
};

function renderModelCard(m) {
  const [text, cls] = MODE_TEXT[m.mode];
  const pill = $('modePill');
  pill.textContent = text; pill.className = `mode-pill ${cls}`;
  const c = m.calibration;
  const tiles = [];
  tiles.push([fmt(c.buildings_detected, 0), 'buildings found']);
  if (c.buildings_measured_by_shadow !== undefined) tiles.push([fmt(c.buildings_measured_by_shadow, 0), 'heights measured from shadows']);
  if (c.shadow_measurement_spread_m) tiles.push([`± ${fmt(c.shadow_measurement_spread_m)} m`, 'shadow height spread']);
  else if (c.loo_rmse_m) tiles.push([`± ${fmt(c.loo_rmse_m)} m`, 'checked on held-out anchors']);
  if (m.building_height_p95 && m.height_unit === 'm') tiles.push([`${fmt(m.building_height_p95, 0)} m`, 'tall buildings (95th pct)']);
  if (m.gsd) tiles.push([`${fmt(m.gsd, 2)} m`, 'pixel size']);
  if (m.sun && m.sun.elevation !== null) tiles.push([`${fmt(m.sun.elevation, 0)}° / ${fmt(m.sun.azimuth, 0)}°`, `sun (${m.sun.source})`]);
  if (m.dem) tiles.unshift([m.dem.source.replace('copernicus_glo30', 'Copernicus 30 m').replace('cartodem', 'CartoDEM'), `DEM · ${m.dem.datum.split(' (')[0]}`]);
  if (m.view && m.view.off_nadir) tiles.push([`${fmt(m.view.off_nadir, 0)}°`, 'off-nadir (lean corrected)']);
  $('modelTiles').innerHTML = tiles.slice(0, 6).map(([b, s]) => `<div class="tile"><b>${b}</b><span>${s}</span></div>`).join('');
  const cs = c.confidence_share;
  $('confBar').innerHTML = `<i style="flex:${cs.green}"></i><i style="flex:${cs.amber}"></i><i style="flex:${cs.red}"></i>`;
  $('confBar').title = `Confidence: ${Math.round(cs.green * 100)}% high, ${Math.round(cs.amber * 100)}% medium, ${Math.round(cs.red * 100)}% low`;
  $('modelNotes').innerHTML = (c.notes || []).map((n) => `<li>${escapeHtml(n)}</li>`).join('');
  const anchors = c.anchors || [];
  $('anchors').innerHTML = anchors.length
    ? `<table class="vtable"><tr><th>source</th><th>height m</th><th>model</th><th>used</th></tr>${anchors.map((a) =>
      `<tr><td>${a.source}</td><td>${fmt(a.h)}</td><td>${fmt(a.r, 3)}</td><td>${a.inlier ? '✓' : '✗'}</td></tr>`).join('')}</table>`
    : '<p class="muted">No metric anchors in this image.</p>';
}

function renderValidation(v) {
  const box = $('validation');
  $('errorLayerBtn').classList.toggle('hidden', !v);
  if (!v) return;
  const row = (name, m, best) => `<tr><td>${name}</td><td class="${best === 'rmse' ? 'best' : ''}">${fmt(m.rmse, 2)}</td><td>${fmt(m.mae, 2)}</td><td>${fmt(m.bias, 2)}</td><td>${fmt(m.corr, 3)}</td></tr>`;
  const better = (a, b) => (a && b && a.rmse !== null && b.rmse !== null && a.rmse < b.rmse ? 'rmse' : '');
  let html = '<table class="vtable"><tr><th>Surface, m</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>r</th></tr>';
  html += row('DepthWizard (all)', v.depthwizard.all, better(v.depthwizard.all, v.dem_only?.all));
  if (v.dem_only) html += row('DEM only (all)', v.dem_only.all, '');
  html += row('DepthWizard (buildings & trees)', v.depthwizard.buildings, better(v.depthwizard.buildings, v.dem_only?.buildings));
  if (v.dem_only) html += row('DEM only (buildings & trees)', v.dem_only.buildings, '');
  html += '</table>';
  box.innerHTML = html;
}

// ------------------------------------------------------------------ step 3: hazards
document.querySelectorAll('#hazardSeg button').forEach((btn) => btn.addEventListener('click', () => {
  document.querySelectorAll('#hazardSeg button').forEach((b) => b.classList.toggle('active', b === btn));
  state.hazard = btn.dataset.hazard;
  configureSlider();
  runScenario();
}));

function configureSlider() {
  const spec = state.hazards[state.hazard];
  if (!spec) return;
  const s = $('level');
  s.min = spec.min; s.max = spec.max; s.step = state.hazard === 'flood' ? 0.5 : 0.05; s.value = spec.default;
  state.level = spec.default;
  $('levelLabel').textContent = spec.param;
  updateLevelOut();
}
function updateLevelOut() {
  const v = Number($('level').value);
  const metricFlood = state.hazard === 'flood' && (!state.job || state.job.has_dem);
  $('levelOut').textContent = state.hazard === 'flood' ? (metricFlood ? `${v} m` : `${v}/15`)
    : state.hazard === 'earthquake' ? `${v.toFixed(2)}×` : v.toFixed(2);
}
let levelTimer = null;
$('level').addEventListener('input', () => {
  updateLevelOut();
  clearTimeout(levelTimer);
  levelTimer = setTimeout(() => { state.level = Number($('level').value); runScenario(); }, 300);
});

let scenarioSeq = 0;
async function runScenario() {
  if (!state.job) return;
  const seq = ++scenarioSeq;
  $('summary').classList.add('loading');
  try {
    const p = await (await api(`/api/jobs/${state.job.id}/scenario`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hazard: state.hazard, level: Number($('level').value) }),
    })).json();
    if (seq !== scenarioSeq) return;
    state.scenario = p;
    state.route = null; $('routeBox').classList.add('hidden');
    const ov = await loadImage(layerUrl(state.layer === 'time' ? 'time' : 'hazard'));
    if (seq !== scenarioSeq) return;
    map.setOverlay(ov, state.layer === 'image' || state.layer === 'time' ? 0.85 : 0.6);
    map.setVectors({ arrows: p.arrows, zones: p.zones, bottlenecks: p.bottlenecks, route: null });
    renderScenario(p);
    if (state.lastRoutePoint) await routeFrom(...state.lastRoutePoint);
    refresh3dTexture();
  } catch (e) {
    $('summary').innerHTML = `<p class="error">${escapeHtml(e.message)}</p>`;
  } finally {
    if (seq === scenarioSeq) $('summary').classList.remove('loading');
  }
}

function renderScenario(p) {
  $('summary').innerHTML = p.summary.map((s) => `<p>${escapeHtml(s)}</p>`).join('');
  const st = p.stats;
  const metric = st.people_at_risk !== null;
  const tiles = [[`${st.buildings_at_risk}/${st.buildings_total}`, 'buildings at risk']];
  if (metric) tiles.push([fmt(st.people_at_risk, 0), 'people at risk (est.)']);
  if (metric && st.median_minutes !== null) tiles.push([`${fmt(st.median_minutes)} min`, 'median walk to safety']);
  tiles.push([st.safe_zones, 'safe zones']);
  $('hazardTiles').innerHTML = tiles.map(([b, s]) => `<div class="tile"><b>${b}</b><span>${s}</span></div>`).join('');
  $('zonesTable').innerHTML = '<tr><th>Zone</th><th>Type</th><th>People</th><th>Room</th></tr>' + p.zones.map((z) =>
    `<tr><td><b>${z.name}</b></td><td>${z.kind}</td><td>${fmt(z.assigned_people, 0)}</td><td class="${z.overloaded ? 'over' : ''}">${fmt(z.capacity, 0)}${z.overloaded ? ' ⚠' : ''}</td></tr>`).join('');
  $('hazardWarnings').innerHTML = (p.warnings || []).map((w) => `<li>${escapeHtml(w)}</li>`).join('');
  renderLegend();
  setExportLinks();
}

function renderLegend() {
  const L = $('legend');
  L.classList.remove('hidden');
  const sw = (c, t) => `<div><i style="background:${c}"></i>${t}</div>`;
  let html = '';
  if (state.layer === 'time') {
    html += '<b>Walking time to safety</b><div class="ramp" style="background:linear-gradient(90deg,#2f9d5e,#d9b21f,#e07b1a,#c93a3a)"></div><div class="ends"><span>0</span><span>10</span><span>30+ min</span></div>';
    html += sw('#5a0078', 'no safe path');
  } else if (state.layer === 'slope') {
    html += '<b>Slope</b><div class="ramp" style="background:linear-gradient(90deg,#2b856e,#f2cc5e,#d94a38)"></div><div class="ends"><span>0°</span><span>30°</span><span>60°+</span></div>';
  } else if (state.layer === 'confidence') {
    html += sw('#28aa5a', 'high confidence') + sw('#f0b428', 'medium') + sw('#d72828', 'low (water, shadow, uncertain)');
  } else if (state.layer === 'buildings') {
    html += '<b>Height above ground</b><div class="ramp" style="background:linear-gradient(90deg,#141828,#3c2878,#be3c6e,#fa963c,#faf5a0)"></div><div class="ends"><span>0</span><span>tallest</span></div>';
  } else if (state.layer === 'error') {
    html += '<b>DSM − reference</b><div class="ramp" style="background:linear-gradient(90deg,#2878dc,#f5f5f5,#d72828)"></div><div class="ends"><span>−6 m</span><span>0</span><span>+6 m</span></div>';
  } else {
    const hz = state.hazard;
    if (hz === 'flood') html += sw('#0f37a0', 'river / lake (detected)') + sw('#1e6ee6', 'flood water (darker = deeper)');
    if (hz === 'landslide') html += sw('#dc2828', 'landslide source') + sw('#f58c1e', 'run-out path');
    if (hz === 'earthquake') html += sw('#f58c1e', 'debris zone') + sw('#787878', 'buildings');
    html += sw('#28be5a', 'safe zone') + (hz === 'flood' ? sw('#14c8d2', 'refuge building (go up)') : '');
    html += '<div><i style="background:#c93a3a;border-radius:50%"></i>choke point</div>';
    html += '<div><i style="background:linear-gradient(90deg,#2f9d5e,#d9b21f,#c93a3a)"></i>arrow: way out (colour = minutes)</div>';
  }
  L.innerHTML = html;
}

// ------------------------------------------------------------------ map interactions
map.onClick = async (u, v) => {
  if (!state.job) return;
  if (state.tool === 'profile') {
    state.profilePts.push([u, v]);
    if (state.profilePts.length > 2) state.profilePts = [[u, v]];
    map.setVectors({ profile: state.profilePts.length === 2 ? state.profilePts : null });
    if (state.profilePts.length === 2) await drawProfile();
    return;
  }
  if (state.tool === 'inspect' || document.body.dataset.mode === 'expert') await inspect(u, v);
  if (state.tool !== 'inspect' && state.scenario) await routeFrom(u, v);
};

async function routeFrom(u, v) {
  const r = await (await api(`/api/jobs/${state.job.id}/route`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ u, v }),
  })).json();
  state.lastRoutePoint = [u, v];
  state.route = r;
  map.setVectors({ route: r.reachable ? r.path : null });
  const box = $('routeBox');
  box.classList.remove('hidden');
  if (!r.reachable) box.innerHTML = `<b>No safe walking route.</b> ${escapeHtml(r.message)}`;
  else {
    const dist = r.distance_m !== null ? `${fmt(r.distance_m, 0)} m, ` : '';
    box.innerHTML = r.minutes === 0 ? '<b>This spot is already safe.</b>'
      : `<b>Go to zone ${r.zone ?? '–'}</b> (${(r.zone_kind || '').toLowerCase()}): ${dist}about <b>${fmt(r.minutes)} min</b> on foot. Follow the pink line.`;
  }
  setExportLinks();
  refresh3dTexture();
}

async function inspect(u, v) {
  const p = await (await api(`/api/jobs/${state.job.id}/point?u=${u}&v=${v}`)).json();
  const unit = p.unit === 'm' ? ' m' : '';
  const rows = [
    ['Surface height', `${fmt(p.surface, 2)}${state.job.mode === 'relative' ? '' : ' m'}`],
    ['Height above ground', `${fmt(p.height_above_ground, 1)}${unit} ± ${fmt(p.uncertainty, 1)}`],
    ['Confidence', p.confidence], ['Slope', `${fmt(p.slope_deg)}°`],
    ['Above drainage', fmt(p.height_above_drainage, 1)],
  ];
  if (p.ground !== null) rows.splice(1, 0, ['Ground (DTM)', `${fmt(p.ground, 1)} m`]);
  if (p.minutes_to_safety !== undefined) rows.push(['Walk to safety', p.minutes_to_safety === null ? 'no route' : `${p.minutes_to_safety} min`]);
  if (p.lonlat) rows.push(['Lon, lat', `${p.lonlat[0]}, ${p.lonlat[1]}`]);
  const box = $('pointInfo');
  box.classList.remove('hidden');
  box.innerHTML = `<dl>${rows.map(([k, v2]) => `<dt>${k}</dt><dd>${v2}</dd>`).join('')}</dl>`;
}

async function drawProfile() {
  const [[u0, v0], [u1, v1]] = state.profilePts;
  const p = await (await api(`/api/jobs/${state.job.id}/profile?u0=${u0}&v0=${v0}&u1=${u1}&v1=${v1}`)).json();
  const cv = $('profileChart');
  cv.classList.remove('hidden');
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = 160;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  const series = [p.surface, p.ground].filter(Boolean);
  const all = series.flat();
  const lo = Math.min(...all), hi = Math.max(...all) || lo + 1;
  const pad = { l: 44, r: 8, t: 10, b: 22 };
  const X = (i) => pad.l + (i / (p.distance.length - 1)) * (W - pad.l - pad.r);
  const Y = (z) => H - pad.b - ((z - lo) / (hi - lo || 1)) * (H - pad.t - pad.b);
  ctx.font = '11px system-ui'; ctx.fillStyle = css.getPropertyValue('--muted');
  ctx.fillText(fmt(hi, 1), 4, pad.t + 8); ctx.fillText(fmt(lo, 1), 4, H - pad.b);
  ctx.fillText(`${fmt(p.distance.at(-1), 0)} ${p.distance_unit}`, W - 60, H - 6);
  const line = (arr, color, fill) => {
    ctx.beginPath(); arr.forEach((z, i) => (i ? ctx.lineTo(X(i), Y(z)) : ctx.moveTo(X(i), Y(z))));
    if (fill) { ctx.lineTo(X(arr.length - 1), H - pad.b); ctx.lineTo(X(0), H - pad.b); ctx.fillStyle = fill; ctx.fill(); }
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke();
  };
  line(p.surface, '#e07b1a', 'rgba(224,123,26,.18)');
  if (p.ground) line(p.ground, '#2f6fd6');
}

function setTool(t) {
  state.tool = t;
  $('profileBtn').classList.toggle('on', t === 'profile');
  $('inspectBtn').classList.toggle('on', t === 'inspect');
  if (t !== 'profile') { state.profilePts = []; map.setVectors({ profile: null }); }
}
$('profileBtn').addEventListener('click', () => setTool(state.tool === 'profile' ? 'route' : 'profile'));
$('inspectBtn').addEventListener('click', () => setTool(state.tool === 'inspect' ? 'route' : 'inspect'));

document.querySelectorAll('#layerSeg button').forEach((btn) => btn.addEventListener('click', () => setLayer(btn.dataset.layer)));
async function setLayer(name) {
  if (!state.job) return;
  state.layer = name;
  document.querySelectorAll('#layerSeg button').forEach((b) => b.classList.toggle('active', b.dataset.layer === name));
  const baseName = name === 'time' ? 'image' : name;
  map.setBase(await loadImage(layerUrl(baseName)));
  if (state.scenario) {
    const ov = await loadImage(layerUrl(name === 'time' ? 'time' : 'hazard'));
    const opacity = name === 'image' || name === 'time' ? 0.85 : (name === 'confidence' || name === 'error' ? 0 : 0.6);
    map.setOverlay(ov, opacity);
  }
  renderLegend();
  refresh3dTexture();
}

$('zoomIn').addEventListener('click', () => map.zoomBy(1.4));
$('zoomOut').addEventListener('click', () => map.zoomBy(1 / 1.4));
$('zoomFit').addEventListener('click', () => map.fit());

// ------------------------------------------------------------------ exports + validation
function setExportLinks() {
  if (!state.job) return;
  const base = `/api/jobs/${state.job.id}/export`;
  const has = !!state.scenario;
  for (const [id, kind] of [['geojsonBtn', 'evacuation.geojson'], ['kmlBtn', 'evacuation.kml']]) {
    $(id).href = `${base}/${kind}`;
    $(id).setAttribute('aria-disabled', String(!has || (kind.endsWith('kml') && !state.job.georeferenced)));
  }
  $('dsmBtn').href = `${base}/dsm.tif`;
  $('ndsmBtn').href = `${base}/ndsm.tif`;
  $('uncBtn').href = `${base}/uncertainty.tif`;
}
$('printBtn').addEventListener('click', () => {
  if (!state.scenario) return;
  const q = state.route && state.route.reachable && state.lastRoutePoint ? `?u=${state.lastRoutePoint[0]}&v=${state.lastRoutePoint[1]}` : '';
  window.open(`/api/jobs/${state.job.id}/export/report.html${q}`, '_blank');
});
$('refFile').addEventListener('change', async () => {
  if (!$('refFile').files.length || !state.job) return;
  const fd = new FormData();
  fd.append('reference', $('refFile').files[0]);
  try {
    const v = await (await api(`/api/jobs/${state.job.id}/validate`, { method: 'POST', body: fd })).json();
    state.job.validation = v;
    renderValidation(v);
  } catch (e) {
    $('validation').innerHTML = `<p class="error">${escapeHtml(e.message)}</p>`;
  }
});

// ------------------------------------------------------------------ 3D
document.querySelectorAll('[data-view]').forEach((btn) => btn.addEventListener('click', async () => {
  document.querySelectorAll('[data-view]').forEach((b) => b.classList.toggle('active', b === btn));
  state.view = btn.dataset.view;
  $('mapView').classList.toggle('hidden', state.view !== 'map');
  $('view3d').classList.toggle('hidden', state.view !== '3d');
  if (state.view === '3d') await show3d();
  else { if (viewer) viewer.visible = false; map.draw(); }
}));

let viewer = null;
async function show3d() {
  if (!state.job) return;
  if (!viewer) {
    const { Terrain3D } = await import('./view3d.js');
    viewer = new Terrain3D($('three'));
  }
  if (state.three !== state.job.id) {
    const res = await api(`/api/jobs/${state.job.id}/heightmap`);
    const gw = Number(res.headers.get('X-Width')), gh = Number(res.headers.get('X-Height'));
    const heights = new Float32Array(await res.arrayBuffer());
    const j = state.job;
    const ex = j.extent_m || [j.width, j.height];
    viewer.setTerrain(heights, gw, gh, ex[0], ex[1], j.mode !== 'relative');
    viewer.setExaggeration(Number($('exag').value));
    viewer.onFlyChange = (on) => {
      $('flyBtn').classList.toggle('on', on);
      $('hint3d').textContent = on ? 'Flying: W A S D / arrows to move · mouse to look · E up · Q down · Shift faster · Esc to stop'
        : 'Drag to rotate · right-drag to pan · scroll to zoom';
    };
    state.dsmHeights = heights; state.demHeights = null;
    state.three = j.id;
  }
  viewer.visible = true;
  viewer.resize();
  refresh3dTexture();
}
function refresh3dTexture() {
  if (viewer && state.three && state.view === '3d') viewer.setTexture(map.composite());
}
async function toggleDemOnly() {
  if (!viewer || !state.job || !state.job.dem) return;
  if (!state.demHeights) {
    const res = await api(`/api/jobs/${state.job.id}/heightmap?which=dem`);
    state.demHeights = new Float32Array(await res.arrayBuffer());
  }
  state.demOnly = !state.demOnly;
  viewer.setHeights(state.demOnly ? state.demHeights : state.dsmHeights);
  $('demOnlyBtn').classList.toggle('on', state.demOnly);
  $('demOnlyBtn').textContent = state.demOnly ? 'DEM only: ON (B)' : 'DEM only (B)';
}
$('demOnlyBtn').addEventListener('click', toggleDemOnly);
$('flyBtn').addEventListener('click', () => viewer && (viewer.fly.isLocked ? viewer.stopFly() : viewer.startFly()));
window.addEventListener('keydown', (e) => {
  if (state.view !== '3d' || e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'KeyB') toggleDemOnly();
  if (e.code === 'KeyF' && viewer && !viewer.fly.isLocked) viewer.startFly();
});

$('exag').addEventListener('input', () => {
  $('exagOut').textContent = $('exag').value;
  if (viewer) viewer.setExaggeration(Number($('exag').value));
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

init();
