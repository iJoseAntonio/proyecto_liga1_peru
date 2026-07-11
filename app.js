/* ═══════════════════════════════════════════════════════════════════════
   app.js  —  Liga 1 Perú Dashboard
   Lee tabla_liga1_peru.csv con PapaParse e inyecta los datos en el DOM.
   ═══════════════════════════════════════════════════════════════════════ */

'use strict';

// ── CONFIGURACIÓN ────────────────────────────────────────────────────────
const CSV_PATH         = 'tabla_liga1_peru.csv';
const MATCHES_CSV_PATH = 'partidos_liga1_2026.csv';
// Reemplaza con tu URL de Render una vez desplegado:
const API_URL          = 'https://api.data-sport.win';

// Map de nombres de equipos del CSV → ID de Sofascore para los escudos
const TEAM_IDS = {
  'Alianza Lima':       2311,
  'Los Chankas':        252254,
  'Cienciano':          2301,
  'Cusco':              63760,
  'Cusco FC':           63760,
  'Universitario':      2305,
  'Deportivo Garcilaso':458584,
  'Melgar':             2308,
  'Alianza Atlético':   2307,
  'Alianza Atletico':   2307,
  'Comerciantes Unidos':213609,
  'ADT':                335557,
  'Sporting Cristal':   2302,
  'Moquegua':           492848,
  'UTC':                87854,
  'Sport Boys':         2312,
  'Cajamarca':          1082002,
  'Atlético Grau':      282538,
  'Atletico Grau':      282538,
  'Sport Huancayo':     33895,
  'ADC Juan Pablo II':  511206,
  'CD Juan Pablo II':   511206,   // alias: nombre en tabla_liga1_peru.csv
};

// Normaliza nombres inconsistentes entre CSVs
const TEAM_NAME_MAP = {
  'CD Juan Pablo II': 'ADC Juan Pablo II',
};

// Zonas de la tabla (posiciones)
const PLAYOFF_POS    = [1];         // Amarillo
const RELEGATION_POS = [17, 18];    // Rojo

// Partidos cargados dinámicamente desde partidos_liga1_2026.csv
let MATCHES = {};

// ── ESTADO ───────────────────────────────────────────────────────────────
let currentRound  = 17;
let ROUND_MAX     = 17;
const ROUND_MIN   = 1;
let standingsData = [];
const predCache   = {};
let _statsLoaded  = false;
let _rendLoaded   = false;
let _edaLoaded    = false;

// ── DOM REFS ─────────────────────────────────────────────────────────────
const $standingsTable = () => document.getElementById('standings-table');
const $matchesList    = () => document.getElementById('matches-list');
const $roundSelect    = () => document.getElementById('round-select');
const $loading        = () => document.getElementById('table-loading');
const $error          = () => document.getElementById('table-error');
const $tableWrap      = () => document.getElementById('table-wrap');
const $countdown      = () => document.getElementById('countdown');

// ── HELPERS ───────────────────────────────────────────────────────────────
function logoUrl(idOrName) {
  const id = typeof idOrName === 'number' ? idOrName : TEAM_IDS[idOrName];
  return id
    ? `https://img.sofascore.com/api/v1/team/${id}/image`
    : '';
}

function getTeamId(name) {
  // Búsqueda exacta primero, luego parcial
  if (TEAM_IDS[name]) return TEAM_IDS[name];
  const key = Object.keys(TEAM_IDS).find(k =>
    name.toLowerCase().includes(k.toLowerCase()) ||
    k.toLowerCase().includes(name.toLowerCase())
  );
  return key ? TEAM_IDS[key] : null;
}

function formBox(letter) {
  // CSV usa: V=victoria, E=empate, D=derrota
  const map = { V:'v', E:'e', D:'d' };
  const labels = { V:'V', E:'E', D:'D' };
  const cls = map[letter.toUpperCase()] || 'd';
  return `<div class="fb ${cls}">${labels[letter.toUpperCase()] || letter}</div>`;
}

// ── MATCHES CSV LOADER ────────────────────────────────────────────────────
function normalizeDate(dateStr) {
  // Acepta "19/7/26" o "31/05/2026" → siempre devuelve "DD/MM/YYYY"
  const parts = (dateStr || '').trim().split('/');
  if (parts.length !== 3) return (dateStr || '').trim();
  let [d, m, y] = parts;
  d = d.padStart(2, '0');
  m = m.padStart(2, '0');
  if (y.length === 2) y = '20' + y;
  return `${d}/${m}/${y}`;
}

function parseMatchDate(dateStr) {
  const [d, m, y] = normalizeDate(dateStr).split('/');
  return new Date(parseInt(y), parseInt(m) - 1, parseInt(d));
}

function formatMatchDate(dateStr) {
  // "31/05/2026" → "31/5/26"
  const [d, m, y] = normalizeDate(dateStr).split('/');
  return `${parseInt(d)}/${parseInt(m)}/${y.slice(2)}`;
}

function isEmptyScore(v) {
  const s = (v ?? '').toString().trim().toUpperCase();
  return s === '' || s === 'N/A';
}

let ROUND_META = {};   // globalRound -> { stage, displayNum }

function loadMatchesCSV() {
  Papa.parse(MATCHES_CSV_PATH, {
    download: true,
    header: true,
    delimiter: ';',
    skipEmptyLines: true,
    complete: (results) => {
      if (!results.data || results.data.length === 0) return;

      // Agrupar por etapa + número de jornada (ej. "Apertura 17", "Clausura 1")
      const groups = {};
      results.data.forEach(row => {
        const jornada = (row['Jornada'] || '').trim();
        const m = jornada.match(/^(.*?)\s+(\d+)$/);
        if (!m) return;
        const stage = m[1].trim();
        const num   = parseInt(m[2]);
        const key   = `${stage}|${num}`;
        if (!groups[key]) groups[key] = { stage, num, rows: [] };
        groups[key].rows.push(row);
      });

      // Ordenar partidos dentro de cada grupo y hallar su fecha mínima
      Object.values(groups).forEach(g => {
        g.rows.sort((a, b) => parseMatchDate(a.fecha) - parseMatchDate(b.fecha));
        g.minDate = parseMatchDate(g.rows[0].fecha);
      });

      // Ordenar los grupos cronológicamente → numeración global secuencial
      const orderedGroups = Object.values(groups).sort((a, b) => a.minDate - b.minDate);

      MATCHES    = {};
      ROUND_META = {};
      orderedGroups.forEach((g, idx) => {
        const globalRound = idx + 1;
        ROUND_META[globalRound] = { stage: g.stage, displayNum: g.num };
        MATCHES[globalRound] = g.rows.map(row => {
          const rawDate  = normalizeDate(row['fecha']);
          const display  = rawDate ? formatMatchDate(rawDate) : '';

          const homeName  = (row['equipo_local']    || '').trim();
          const awayName  = (row['equipo_visitante'] || '').trim();
          const gl        = row['goles_local'];
          const gv        = row['goles_visitante'];
          const hasScore  = !isEmptyScore(gl) && !isEmptyScore(gv);

          const horaRaw   = (row['Hora'] || '').trim();
          const horaVal   = (!horaRaw || horaRaw.toLowerCase() === 'no') ? null : horaRaw;

          return {
            date:     display,
            rawDate:  rawDate,
            hour:     hasScore ? 'FT' : horaVal,
            homeId:   getTeamId(homeName),
            homeName,
            awayId:   getTeamId(awayName),
            awayName,
            sh:       hasScore ? parseInt(gl) : null,
            sa:       hasScore ? parseInt(gv) : null,
          };
        });
      });

      const rounds = Object.keys(MATCHES).map(Number);
      ROUND_MAX    = Math.max(...rounds);

      // Por defecto: la última ronda con al menos un partido ya finalizado
      currentRound = ROUND_MAX;
      for (let r = ROUND_MAX; r >= ROUND_MIN; r--) {
        if (MATCHES[r] && MATCHES[r].some(m => m.sh !== null)) {
          currentRound = r;
          break;
        }
      }

      buildRoundSelect();
      renderMatches(currentRound);
      renderDestacado(currentRound);
      if (isPredTabActive()) renderPredictionsTab(currentRound);
    },
  });
}

// ── CSV LOADER ────────────────────────────────────────────────────────────
function loadCSV() {
  $loading().style.display  = 'flex';
  $error().style.display    = 'none';
  $tableWrap().style.display = 'none';

  Papa.parse(CSV_PATH, {
    download: true,
    header:   true,
    skipEmptyLines: true,
    complete: (results) => {
      if (!results.data || results.data.length === 0) {
        showError();
        return;
      }
      standingsData = results.data;
      $loading().style.display   = 'none';
      $tableWrap().style.display = 'block';
      renderStandings(standingsData);
    },
    error: () => showError(),
  });
}

function showError() {
  $loading().style.display = 'none';
  $error().style.display   = 'flex';
}

// ── RENDER STANDINGS ──────────────────────────────────────────────────────
function renderStandings(data) {
  const el = $standingsTable();
  let html = '';

  data.forEach((row, i) => {
    const pos     = parseInt(row['Posicion'] || row['posicion'] || i + 1);
    const rawName = (row['Equipo'] || '').trim();
    const name    = TEAM_NAME_MAP[rawName] || rawName;
    const teamId = getTeamId(name);
    const logo   = teamId
      ? `https://img.sofascore.com/api/v1/team/${teamId}/image`
      : '';

    const nv  = v => (v !== undefined && v !== null && v !== '') ? v : 0;
    const pj  = nv(row['PJ']);
    const pg  = nv(row['PG']);
    const pe  = nv(row['PE']);
    const pp  = nv(row['PP']);
    const dif = nv(row['DIF']);
    const gls = nv(row['Goles']);
    const pts = nv(row['Puntos']);
    const forma = (row['Ultimos_5'] || '').trim();

    // Separadores de zona
    if (pos === 2)  html += `<div class="zone-sep"></div>`;
    if (pos === 17) html += `<div class="zone-sep"></div>`;

    const isPlayoff    = PLAYOFF_POS.includes(pos);
    const isRelegation = RELEGATION_POS.includes(pos);
    const rowClass = isPlayoff ? 'playoff-zone' : isRelegation ? 'relegation-zone' : '';
    const circleClass = isPlayoff ? 'playoff' : isRelegation ? 'relegation' : '';

    const formaHtml = forma.split('').map(formBox).join('');

    html += `
    <div class="team-row ${rowClass}" style="animation-delay:${i * 0.03}s">
      <span class="zone-indicator"></span>
      <div class="pos-circle ${circleClass}">${pos}</div>
      <div class="team-cell">
        ${logo
          ? `<img class="team-logo-sm" src="${logo}" alt="${name}"
               onerror="this.style.opacity=0.15">`
          : `<div style="width:20px;height:20px;flex-shrink:0"></div>`
        }
        <span class="team-name-cell">${name}</span>
      </div>
      <span class="td">${pj}</span>
      <span class="td">${pg}</span>
      <span class="td">${pe}</span>
      <span class="td">${pp}</span>
      <span class="td">${dif}</span>
      <span class="td">${gls}</span>
      <div class="form-mini">${formaHtml}</div>
      <span class="td pts">${pts}</span>
    </div>`;
  });

  el.innerHTML = html;
}

// ── RENDER MATCHES ────────────────────────────────────────────────────────
function renderMatches(round) {
  const matches = MATCHES[round] || [];
  const el = $matchesList();
  let html = '';

  matches.forEach((m, i) => {
    const finished = m.sh !== null;
    const homeWin  = finished && m.sh > m.sa;
    const awayWin  = finished && m.sa > m.sh;

    const dateHtml = m.date
      ? `<span class="match-date">${m.date}</span>` : '';
    const statusHtml = m.hour === 'FT'
      ? `<span class="match-ft">FT</span>`
      : m.hour
        ? `<span class="match-date">${m.hour}</span>`
        : `<span class="match-hour"></span>`;

    const scoreHtml = finished
      ? `<div class="match-scores">
           <span class="match-score ${homeWin ? 'winner' : ''}">${m.sh}</span>
           <span class="match-score ${awayWin ? 'winner' : ''}">${m.sa}</span>
         </div>`
      : `<div style="min-width:16px"></div>`;

    html += `
    <div class="match-row" style="animation-delay:${i * 0.04}s">
      <div class="match-time-cell">
        ${dateHtml}
        ${statusHtml}
      </div>
      <div class="match-teams">
        <div class="match-team-row">
          <img src="https://img.sofascore.com/api/v1/team/${m.homeId}/image/small"
               alt="${m.homeName}" onerror="this.style.opacity=0.15">
          <span class="match-team-name ${homeWin ? 'winner' : ''}">${m.homeName}</span>
        </div>
        <div class="match-team-row">
          <img src="https://img.sofascore.com/api/v1/team/${m.awayId}/image/small"
               alt="${m.awayName}" onerror="this.style.opacity=0.15">
          <span class="match-team-name ${awayWin ? 'winner' : ''}">${m.awayName}</span>
        </div>
      </div>
      ${scoreHtml}
    </div>`;

    if (i < matches.length - 1) {
      html += `<div class="match-sep"></div>`;
    }
  });

  el.innerHTML = html || '<div style="padding:20px;text-align:center;color:var(--text3);font-size:12px">Sin partidos para esta jornada</div>';
}

// ── PREDICCIONES API ──────────────────────────────────────────────────────
async function getMatchResult(m) {
  const key = `result|${m.homeName}|${m.awayName}`;
  if (predCache[key]) return predCache[key];
  try {
    const url = `${API_URL}/match-result` +
      `?home=${encodeURIComponent(m.homeName)}` +
      `&away=${encodeURIComponent(m.awayName)}`;
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    predCache[key] = data;
    return data;
  } catch (_) { return null; }
}

async function getPrediction(m) {
  const key = `${m.homeName}|${m.awayName}|${m.rawDate || ''}`;
  if (predCache[key]) return predCache[key];
  try {
    const url = `${API_URL}/predict-match` +
      `?home=${encodeURIComponent(m.homeName)}` +
      `&away=${encodeURIComponent(m.awayName)}` +
      (m.rawDate ? `&fecha=${encodeURIComponent(m.rawDate)}` : '');
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    predCache[key] = data;
    return data;
  } catch (_) { return null; }
}

// ── ROUND SELECT ──────────────────────────────────────────────────────────
function buildRoundSelect() {
  const sel   = $roundSelect();
  const stage = (ROUND_META[currentRound] || {}).stage || '';

  sel.innerHTML = '';
  Object.keys(ROUND_META)
    .map(Number)
    .filter(r => ROUND_META[r].stage === stage)
    .sort((a, b) => b - a)   // más reciente primero
    .forEach(r => {
      const opt = document.createElement('option');
      opt.value = r;
      opt.textContent = `${stage} Ronda ${ROUND_META[r].displayNum}`;
      if (r === currentRound) opt.selected = true;
      sel.appendChild(opt);
    });

  updateStageHeader(stage);
}

function updateStageHeader(stage) {
  const headerEl = document.querySelector('.match-group-header span');
  if (headerEl && stage) headerEl.textContent = `Liga 1, ${stage}`;
}

function changeRound(dir) {
  const next = currentRound + dir;
  if (next < ROUND_MIN || next > ROUND_MAX) return;
  currentRound = next;
  buildRoundSelect();   // por si el cambio cruza a otra etapa (Apertura ↔ Clausura)
  renderMatches(currentRound);
  renderDestacado(currentRound);
  if (isPredTabActive()) renderPredictionsTab(currentRound);
}

// ── DESTACADO ─────────────────────────────────────────────────────────────
function renderDestacado(round) {
  const el = document.getElementById('destacado-match');
  if (!el) return;

  const matches = MATCHES[round] || [];
  const played  = matches.filter(m => m.sh !== null);

  let featured, showScore;

  if (!played.length) {
    const next = matches.find(m => m.sh === null);
    if (!next) {
      el.innerHTML = `<div class="match-no-data">Sin partidos en esta jornada</div>`;
      return;
    }
    featured = next;
    showScore = false;
  } else {
    featured = played.reduce((a, b) => (a.sh + a.sa) >= (b.sh + b.sa) ? a : b);
    showScore = true;
  }

  el.innerHTML = buildDestacadoHTML(featured, showScore);
}

function buildDestacadoHTML(m, showScore) {
  const centerHTML = showScore
    ? `<div class="match-score-feat">${m.sh} - ${m.sa}</div>
       <div class="match-total-goals">${m.sh + m.sa} goles totales</div>`
    : `<div class="match-upcoming-time">${m.hour || '--:--'}</div>
       <div class="match-upcoming-label">${m.date || 'Próximo'}</div>`;

  return `
    <div class="team-feat">
      <img src="https://img.sofascore.com/api/v1/team/${m.homeId}/image"
           alt="${m.homeName}" onerror="this.style.opacity=0.15">
      <span>${m.homeName}</span>
    </div>
    <div class="match-center">${centerHTML}</div>
    <div class="team-feat">
      <img src="https://img.sofascore.com/api/v1/team/${m.awayId}/image"
           alt="${m.awayName}" onerror="this.style.opacity=0.15">
      <span>${m.awayName}</span>
    </div>`;
}


// ── PROGRESS BAR ─────────────────────────────────────────────────────────
function updateProgress() {
  const SEASON_START = new Date('2026-01-30');
  const SEASON_END   = new Date('2026-11-29');
  const now          = new Date();

  const total   = SEASON_END - SEASON_START;
  const elapsed = Math.min(Math.max(now - SEASON_START, 0), total);
  const pct     = (elapsed / total) * 100;

  const fill = document.querySelector('.progress-fill');
  if (fill) fill.style.width = `${pct.toFixed(1)}%`;

  const fmt = d => d.toLocaleDateString('es-PE', { day: 'numeric', month: 'short' });
  const spans = document.querySelectorAll('.progress-dates span');
  if (spans[0]) spans[0].textContent = fmt(SEASON_START);
  if (spans[1]) spans[1].textContent = fmt(SEASON_END);
}

// ── COUNTDOWN ─────────────────────────────────────────────────────────────
function updateCountdown() {
  const el = $countdown();
  if (!el) return;
  const now    = new Date();
  const target = new Date();
  target.setDate(target.getDate() + 1);
  target.setHours(13, 15, 0, 0);
  const diff = target - now;
  if (diff <= 0) { el.textContent = 'En curso'; return; }
  const h = String(Math.floor(diff / 3600000)).padStart(2, '0');
  const m = String(Math.floor((diff % 3600000) / 60000)).padStart(2, '0');
  const s = String(Math.floor((diff % 60000) / 1000)).padStart(2, '0');
  el.textContent = `${h}:${m}:${s}`;
}

// ── TABS ──────────────────────────────────────────────────────────────────
function setupMainTabs() {
  document.querySelectorAll('.main-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.main-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      const target = document.getElementById(`tab-${tab.dataset.tab}`);
      if (target) target.classList.add('active');
      const name = tab.dataset.tab;
      if (name === 'predicciones') renderPredictionsTab(currentRound);
      if (name === 'estadisticas' && !_statsLoaded) { renderEstadisticasTab(); _statsLoaded = true; }
      if (name === 'rendimiento'  && !_rendLoaded)  { renderRendimientoTab();  _rendLoaded  = true; }
      if (name === 'eda'          && !_edaLoaded)   { renderEdaTab();          _edaLoaded   = true; }
    });
  });
}

function isPredTabActive() {
  const t = document.querySelector('.main-tab.active');
  return t && t.dataset.tab === 'predicciones';
}

// ── PREDICCIONES TAB ──────────────────────────────────────────────────────
async function renderPredictionsTab(round) {
  const container = document.getElementById('pred-tab-content');
  const titleEl = document.getElementById('pred-tab-round');
  if (!container) return;

  const meta = ROUND_META[round] || {};
  if (titleEl) titleEl.textContent = `${meta.stage || 'Liga 1'} — Jornada ${meta.displayNum ?? round}`;

  const matches = MATCHES[round] || [];
  if (!matches.length) {
    container.innerHTML = '<div class="empty-tab">Sin partidos para esta jornada</div>';
    return;
  }

  container.innerHTML =
    `<div class="loading-state"><div class="spinner"></div><span>Cargando predicciones…</span></div>`;

  // Predicciones del modelo + resultados reales en paralelo
  const [preds, results] = await Promise.all([
    Promise.all(matches.map(m => getPrediction(m))),
    Promise.all(matches.map(m => m.sh !== null ? getMatchResult(m) : Promise.resolve(null))),
  ]);

  const html = matches.map((m, i) => {
    const pred = preds[i];
    if (!pred) {
      return `<div class="pred-card" id="pred-card-${round}-${i}" style="padding:14px;color:var(--text3);font-size:12px;
              text-align:center">${m.homeName} vs ${m.awayName} — sin datos del modelo</div>`;
    }
    return `<div class="pred-card" id="pred-card-${round}-${i}" style="animation-delay:${i * 0.05}s">
              ${buildPredCardHTML(m, pred, results[i])}
            </div>`;
  }).join('');

  container.innerHTML = html;

}

function buildPredCardHTML(m, data, result = null) {
  const finished = m.sh !== null;

  const hXG  = data.local.xg;
  const hTir = data.local.tiros;
  const hGol = data.local.goles;
  const aXG  = data.visitante.xg;
  const aTir = data.visitante.tiros;
  const aGol = data.visitante.goles;

  // ✓ si el modelo acertó (predijo alto y se cumplió, o predijo bajo y no se cumplió)
  function rChk(predicted, cumple) {
    if (cumple === undefined || cumple === null) return '';
    const ok = predicted === cumple;
    return `<span class="pred-check ${ok ? 'ok' : 'fail'}">${ok ? '✓' : '✗'}</span>`;
  }

  // 3 barras apiladas
  const BARS = [
    { label: 'XG', threshold: '≥ 1.5', key: 'xg'    },
    { label: 'GA', threshold: '≥ 2',   key: 'goles'  },
    { label: 'TP', threshold: '≥ 5',   key: 'tiros'  },
  ];

  function barsHome(d) {
    return BARS.map(b => {
      const m = d[b.key];
      return `
        <div class="pred-bar-row">
          <span class="pred-bar-label">
            <span class="pbl-abbr">${b.label}</span>
            <span class="pbl-thresh">${b.threshold}</span>
          </span>
          <div class="pred-bar-track">
            <div class="pred-bar-fill ${m.alto ? 'p-high' : 'p-low'}" style="width:${m.probabilidad}%"></div>
          </div>
          <span class="pred-bar-pct">${m.probabilidad}%</span>
        </div>`;
    }).join('');
  }

  function barsAway(d) {
    return BARS.map(b => {
      const m = d[b.key];
      return `
        <div class="pred-bar-row away">
          <span class="pred-bar-pct">${m.probabilidad}%</span>
          <div class="pred-bar-track away">
            <div class="pred-bar-fill ${m.alto ? 'p-high' : 'p-low'}" style="width:${m.probabilidad}%"></div>
          </div>
          <span class="pred-bar-label" style="text-align:right">
            <span class="pbl-abbr">${b.label}</span>
            <span class="pbl-thresh">${b.threshold}</span>
          </span>
        </div>`;
    }).join('');
  }

  // Stats reales del partido
  const hReal = result
    ? `<div class="pred-real">XG ${result.local.xg} ${rChk(hXG.alto, result.local.cumple_xg)} · GA ${result.local.goles} ${rChk(hGol.alto, result.local.cumple_goles)} · TP ${result.local.tiros_puerta} ${rChk(hTir.alto, result.local.cumple_tiros)}</div>` : '';
  const aReal = result
    ? `<div class="pred-real away">XG ${result.visitante.xg} ${rChk(aXG.alto, result.visitante.cumple_xg)} · GA ${result.visitante.goles} ${rChk(aGol.alto, result.visitante.cumple_goles)} · TP ${result.visitante.tiros_puerta} ${rChk(aTir.alto, result.visitante.cumple_tiros)}</div>` : '';

  const centerHtml = finished
    ? `<div class="pred-scorebox">${m.sh}<span>-</span>${m.sa}</div>
       <div class="pred-vs">FT</div>`
    : `<div class="pred-vs">VS</div>
       ${m.date ? `<div class="pred-matchdate">${m.date}</div>` : ''}`;

  return `
    <div class="pred-team home">
      <div class="pred-team-head">
        <img class="pred-logo" src="https://img.sofascore.com/api/v1/team/${m.homeId}/image"
             alt="${m.homeName}" onerror="this.style.opacity=0.15">
        <span class="pred-name">${m.homeName}</span>
      </div>
      <div class="pred-bars-stack">${barsHome(data.local)}</div>
      ${hReal}
    </div>

    <div class="pred-center">${centerHtml}</div>

    <div class="pred-team away">
      <div class="pred-team-head away">
        <span class="pred-name">${m.awayName}</span>
        <img class="pred-logo" src="https://img.sofascore.com/api/v1/team/${m.awayId}/image"
             alt="${m.awayName}" onerror="this.style.opacity=0.15">
      </div>
      <div class="pred-bars-stack">${barsAway(data.visitante)}</div>
      ${aReal}
    </div>`;
}

// ── ESTADÍSTICAS TAB ──────────────────────────────────────────────────────
let _statsData = null;
let _statsSort = { col: 'xg_avg', dir: 1 }; // 1 = asc, -1 = desc

function renderStatsTable() {
  const content = document.getElementById('stats-table-content');
  if (!content || !_statsData) return;

  const data = _statsData;
  const sorted = [...data].sort((a, b) => _statsSort.dir * (b[_statsSort.col] - a[_statsSort.col]));

  const maxXG  = Math.max(...data.map(t => t.xg_avg));
  const maxGol = Math.max(...data.map(t => t.goles_avg));
  const maxTot = Math.max(...data.map(t => t.tiros_tot_avg));
  const maxTir = Math.max(...data.map(t => t.tiros_avg));

  const arr = col => {
    if (_statsSort.col !== col) return `<span class="sort-arr">↕</span>`;
    return _statsSort.dir === -1
      ? `<span class="sort-arr on">↓</span>`
      : `<span class="sort-arr on">↑</span>`;
  };

  const COLS = [
    { key: 'xg_avg',        label: 'Goles Esperados' },
    { key: 'goles_avg',     label: 'Goles'           },
    { key: 'tiros_tot_avg', label: 'Tiros Totales'   },
    { key: 'tiros_avg',     label: 'Tiros a Puerta'  },
  ];

  let html = `
    <div class="stats-table-head">
      <span>#</span>
      <span>Equipo</span>
      <span style="text-align:center">PJ</span>
      ${COLS.map(c => `<span class="stats-sort-th" data-col="${c.key}">${c.label} ${arr(c.key)}</span>`).join('')}
    </div>`;

  sorted.forEach((team, i) => {
    const id  = getTeamId(team.equipo);
    const logo = id ? `https://img.sofascore.com/api/v1/team/${id}/image` : '';
    const xgW  = ((team.xg_avg        / maxXG)  * 100).toFixed(0);
    const golW = ((team.goles_avg      / maxGol) * 100).toFixed(0);
    const totW = ((team.tiros_tot_avg  / maxTot) * 100).toFixed(0);
    const tirW = ((team.tiros_avg      / maxTir) * 100).toFixed(0);

    html += `
      <div class="stats-row" style="animation-delay:${i * 0.03}s">
        <span class="stats-rank">${i + 1}</span>
        <div class="stats-team-cell">
          ${logo ? `<img src="${logo}" alt="${team.equipo}" onerror="this.style.opacity=0.15">` : '<div style="width:22px;flex-shrink:0"></div>'}
          <span>${team.equipo}</span>
        </div>
        <span class="stats-num-cell">${team.partidos}</span>
        <div class="stats-bar-cell">
          <div class="stats-bar-header"><span class="stats-val">${parseFloat(team.xg_avg).toFixed(2)}</span></div>
          <div class="stats-mini-bar-track"><div class="stats-mini-bar-fill xg-bar" style="width:${xgW}%"></div></div>
        </div>
        <div class="stats-bar-cell">
          <div class="stats-bar-header"><span class="stats-val">${parseFloat(team.goles_avg).toFixed(2)}</span></div>
          <div class="stats-mini-bar-track"><div class="stats-mini-bar-fill goles-bar" style="width:${golW}%"></div></div>
        </div>
        <div class="stats-bar-cell">
          <div class="stats-bar-header"><span class="stats-val">${parseFloat(team.tiros_tot_avg).toFixed(1)}</span></div>
          <div class="stats-mini-bar-track"><div class="stats-mini-bar-fill tiros-tot-bar" style="width:${totW}%"></div></div>
        </div>
        <div class="stats-bar-cell">
          <div class="stats-bar-header"><span class="stats-val">${parseFloat(team.tiros_avg).toFixed(1)}</span></div>
          <div class="stats-mini-bar-track"><div class="stats-mini-bar-fill tiros-bar" style="width:${tirW}%"></div></div>
        </div>
      </div>`;
  });

  content.innerHTML = html;

  content.querySelectorAll('.stats-sort-th').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      _statsSort = _statsSort.col === col
        ? { col, dir: _statsSort.dir * -1 }
        : { col, dir: -1 };
      renderStatsTable();
    });
  });
}

async function renderEstadisticasTab() {
  const tabEl = document.getElementById('tab-estadisticas');
  if (!tabEl) return;

  tabEl.innerHTML = `
    <div class="stats-tab-wrap">
      <div class="stats-tab-header">
        <span class="pred-tab-title">Promedio de estadísticas ofensivas</span>
      </div>
      <div class="stats-table-wrap">
        <div id="stats-table-content">
          <div class="loading-state"><div class="spinner"></div><span>Cargando ranking...</span></div>
        </div>
      </div>
    </div>`;

  try {
    const res = await fetch(`${API_URL}/team-rankings`);
    if (!res.ok) throw new Error();
    _statsData = await res.json();
    if (!_statsData.length) {
      document.getElementById('stats-table-content').innerHTML =
        '<div class="empty-tab">Sin datos disponibles</div>';
      return;
    }
    renderStatsTable();
  } catch (_) {
    const c = document.getElementById('stats-table-content');
    if (c) c.innerHTML = '<div class="empty-tab">No se pudo cargar las estadísticas</div>';
  }
}

// ── RENDIMIENTO TAB — helpers ─────────────────────────────────────────────
let _radarChart = null;
let _shapChart  = null;

const MODEL_COLORS = {
  'XGBoost':             '#a78bfa',
  'LightGBM':            '#38bdf8',
  'Random Forest':       '#4ade80',
  'Logistic Regression': '#fbbf24',
};

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function renderCompChart(varKey, metricsData) {
  if (_radarChart) { _radarChart.destroy(); _radarChart = null; }
  const canvas = document.getElementById('comp-radar');
  if (!canvas || !metricsData[varKey]) return;

  const { modelos } = metricsData[varKey];
  const labels   = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC'];
  const datasets = modelos.map(m => {
    const col = MODEL_COLORS[m.nombre] || '#888';
    return {
      label:               m.nombre,
      data:                [m.accuracy, m.precision, m.recall, m.f1, m.auc_roc],
      borderColor:         col,
      backgroundColor:     hexToRgba(col, 0.1),
      pointBackgroundColor: col,
      pointRadius:         4,
      borderWidth:         2,
    };
  });

  _radarChart = new Chart(canvas.getContext('2d'), {
    type: 'radar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0,
          max: 1,
          ticks: {
            stepSize: 0.2,
            color: '#ccc',
            font: { size: 9 },
            backdropColor: 'transparent',
          },
          grid:        { color: 'rgba(255,255,255,0.15)' },
          angleLines:  { color: 'rgba(255,255,255,0.15)' },
          pointLabels: { color: '#e0e0e0', font: { size: 11 } },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(1)}%`,
          },
        },
      },
    },
  });

  // Leyenda custom tipo checklist
  const legendEl = document.getElementById('comp-legend');
  if (legendEl) {
    legendEl.innerHTML = modelos.map((m, i) => {
      const col = MODEL_COLORS[m.nombre] || '#888';
      return `
        <div class="comp-legend-item" data-index="${i}">
          <div class="comp-legend-box" style="background:${col}; border-color:${col}"></div>
          <span>${m.nombre}</span>
        </div>`;
    }).join('');

    legendEl.querySelectorAll('.comp-legend-item').forEach(item => {
      item.addEventListener('click', () => {
        const idx  = parseInt(item.dataset.index);
        const meta = _radarChart.getDatasetMeta(idx);
        meta.hidden = !meta.hidden;
        item.classList.toggle('legend-off', meta.hidden);
        _radarChart.update();
      });
    });
  }

  const tbody = document.getElementById('comp-table-body');
  if (!tbody) return;
  tbody.innerHTML = modelos.map(m => {
    const col = MODEL_COLORS[m.nombre] || '#888';
    return `
      <div class="comp-table-row">
        <span class="comp-model-name" style="color:${col}">${m.nombre}</span>
        <span>${(m.accuracy  * 100).toFixed(1)}%</span>
        <span>${(m.precision * 100).toFixed(1)}%</span>
        <span>${(m.recall    * 100).toFixed(1)}%</span>
        <span>${(m.f1        * 100).toFixed(1)}%</span>
        <span>${m.auc_roc.toFixed(4)}</span>
      </div>`;
  }).join('');
}

// ── SHAP helpers ───────────────────────────────────────────────────────────
let _shapAsc = false;

const SHAP_VAR_NAMES = {
  'goles':                     'Goles',
  'Posesión de pelota':        'Posesión de Pelota',
  'Goles esperados (xG)':      'xG',
  'Tiros totales':             'Tiros Totales',
  'Tiros a puerta':            'Tiros a Puerta',
  'Disparos al palo':          'Disparos al Palo',
  'Tiros fuera':               'Tiros Fuera',
  'Tiros bloqueados':          'Tiros Bloqueados',
  'Tiros adentro del area':    'Tiros Dentro del Área',
  'Tiros desde fuera del area':'Tiros Fuera del Área',
  'Fueras de juego':           'Fueras de Juego',
  'Saques de banda':           'Saques de Banda',
  'Pases al ultimo tercio':    'Pases Último Tercio',
  'Entradas':                  'Entradas',
  'Intercepciones':            'Intercepciones',
  'Recuperaciones':            'Recuperaciones',
  'Despejes':                  'Despejes',
  'Corners':                   'Corners',
  'Faltas':                    'Faltas',
  'Tiros libres':              'Tiros Libres',
  'Tarjetas amarillas':        'Tarjetas Amarillas',
  'Tarjetas rojas':            'Tarjetas Rojas',
  'Atajadas':                  'Atajadas',
  'Saques de meta':            'Saques de Meta',
  'precision_pases':           'Precisión Pases',
  'precision_tiros':           'Precisión Tiros',
  'conversion_xg':             'Conversión xG',
  'ratio_area':                'Ratio Área',
  'Local':                     'Local',
};

function formatShapVar(raw) {
  const suffix = raw.endsWith('_prom_3') ? ' (prom. 3)'
               : raw.endsWith('_prom_5') ? ' (prom. 5)'
               : '';
  const base = raw.replace(/_prom_[35]$/, '');
  return (SHAP_VAR_NAMES[base] || base) + suffix;
}

function renderShapChart(target, shapData) {
  if (_shapChart) { _shapChart.destroy(); _shapChart = null; }
  const canvas = document.getElementById('shap-bar-chart');
  if (!canvas || !shapData || !shapData[target]) return;

  // desc (default): más importante arriba → invertir array para Chart.js
  const sorted = _shapAsc
    ? [...shapData[target]]
    : [...shapData[target]].reverse();

  const labels = sorted.map(d => formatShapVar(d.variable));
  const values = sorted.map(d => d.importancia);
  const colors = sorted.map(d => {
    const ratio = d.importancia > 0 ? Math.abs(d.direccion) / d.importancia : 0;
    if (ratio < 0.10) return 'rgba(56,189,248,0.72)';
    return d.direccion > 0 ? 'rgba(21,177,104,0.78)' : 'rgba(226,75,74,0.78)';
  });

  _shapChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.raw.toFixed(5)}  mean |SHAP|` }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.1)' },
          ticks: { color: '#f0f0f0', font: { size: 10 }, maxTicksLimit: 6 },
          border: { color: 'rgba(255,255,255,0.3)' },
        },
        y: {
          grid: { display: false },
          ticks: { color: '#f0f0f0', font: { size: 11 }, autoSkip: false },
          border: { color: 'rgba(255,255,255,0.3)' },
        }
      }
    }
  });

  // Actualizar texto del botón
  const btn = document.getElementById('shap-sort-btn');
  if (btn) btn.textContent = _shapAsc ? '↑ Asc' : '↓ Desc';
}

// ── EDA & CLUSTERING TAB ─────────────────────────────────────────────────
let _edaHistChart    = null;
let _edaElbowChart   = null;
let _edaSilChart     = null;
let _edaScatterChart = null;

// Paleta categórica validada (CVD-safe, banda de luminosidad para superficie oscura)
const CLUSTER_COLORS = ['#a8791f', '#2e6fb0', '#009E73', '#c0392b', '#9450c9', '#b8407a'];
const CLUSTER_SHAPES = ['circle', 'triangle', 'rect', 'rectRot', 'star', 'cross', 'rectRounded', 'crossRot'];
const clusterColor = i => CLUSTER_COLORS[i % CLUSTER_COLORS.length];
const clusterShape = i => CLUSTER_SHAPES[i % CLUSTER_SHAPES.length];

function renderEdaHistogram(varKey, edaData) {
  if (_edaHistChart) { _edaHistChart.destroy(); _edaHistChart = null; }
  const canvas = document.getElementById('eda-hist-chart');
  const v = edaData.variables[varKey];
  if (!canvas || !v) return;

  const bins   = v.histogram.bins;
  const labels = bins.slice(0, -1).map((b, i) => `${b}–${bins[i + 1]}`);

  _edaHistChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: v.histogram.counts,
        backgroundColor: hexToRgba('#6c63ff', 0.75),
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend:  { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.raw} partidos-equipo` } },
      },
      scales: {
        x: {
          grid:   { display: false },
          ticks:  { color: '#f0f0f0', font: { size: 9 }, maxRotation: 45, minRotation: 45 },
          border: { color: 'rgba(255,255,255,0.3)' },
        },
        y: {
          beginAtZero: true,
          grid:   { color: 'rgba(255,255,255,0.1)' },
          ticks:  { color: '#f0f0f0', font: { size: 10 } },
          border: { color: 'rgba(255,255,255,0.3)' },
        },
      },
    },
  });
}

function renderEdaBoxplot(varKey, edaData) {
  const wrap = document.getElementById('eda-boxplot-wrap');
  const v = edaData.variables[varKey];
  if (!wrap || !v) return;

  const b = v.boxplot;
  const lo   = Math.min(b.min, b.whisker_low);
  const hi   = Math.max(b.max, b.whisker_high);
  const span = (hi - lo) || 1;
  const pct  = x => ((x - lo) / span * 100).toFixed(1);

  wrap.innerHTML = `
    <div class="eda-boxplot">
      <div class="eda-boxplot-track" style="left:${pct(b.whisker_low)}%; right:${(100 - pct(b.whisker_high))}%"></div>
      <div class="eda-boxplot-cap" style="left:${pct(b.whisker_low)}%"></div>
      <div class="eda-boxplot-cap" style="left:${pct(b.whisker_high)}%"></div>
      <div class="eda-boxplot-box" style="left:${pct(b.q1)}%; right:${(100 - pct(b.q3))}%"></div>
      <div class="eda-boxplot-median" style="left:${pct(b.median)}%"></div>
    </div>
    <div class="eda-boxplot-labels">
      <span>Mín ${b.whisker_low}</span>
      <span>Q1 ${b.q1}</span>
      <span class="eda-boxplot-median-label">Mediana ${b.median}</span>
      <span>Q3 ${b.q3}</span>
      <span>Máx ${b.whisker_high}</span>
    </div>
    <div class="eda-boxplot-outliers">
      ⚠ ${b.outlier_count} valores atípicos (regla 1.5·IQR) — ${b.outlier_pct}% de las observaciones
    </div>`;
}

function renderEdaStats(varKey, edaData) {
  const row = document.getElementById('eda-stats-row');
  const v = edaData.variables[varKey];
  if (!row || !v) return;
  const s = v.stats;
  const items = [
    ['n', s.count], ['Media', s.mean], ['Desv. Est.', s.std], ['Mín', s.min],
    ['Q1', s.q1], ['Mediana', s.median], ['Q3', s.q3], ['Máx', s.max],
  ];
  row.innerHTML = items.map(([label, val]) => `
    <div class="eda-stat-item">
      <span class="eda-stat-value">${val}</span>
      <span class="eda-stat-label">${label}</span>
    </div>`).join('');
}

function renderEdaVariable(varKey, edaData) {
  renderEdaHistogram(varKey, edaData);
  renderEdaBoxplot(varKey, edaData);
  renderEdaStats(varKey, edaData);
}

function corrColor(v) {
  // Diverging: rojo (negativo) ↔ gris neutro (0) ↔ verde (positivo)
  const t = Math.min(Math.abs(v), 1);
  const neutral = [46, 46, 46];               // --surface3
  const pole = v >= 0 ? [21, 177, 104]         // --green
                       : [226, 75, 74];        // --red
  const rgb = neutral.map((c, i) => Math.round(c + (pole[i] - c) * t));
  return `rgb(${rgb.join(',')})`;
}

function renderCorrHeatmap(correlacion) {
  const wrap = document.getElementById('eda-corr-heatmap');
  if (!wrap) return;
  const { labels, matriz } = correlacion;
  const short = labels.map(l => formatShapVar(l));

  let html = `<div class="eda-corr-grid" style="grid-template-columns:110px repeat(${labels.length}, 1fr)">`;
  html += `<div></div>` + short.map(l => `<div class="eda-corr-head">${l}</div>`).join('');
  matriz.forEach((row, i) => {
    html += `<div class="eda-corr-head eda-corr-head-row">${short[i]}</div>`;
    html += row.map((val, j) => `
      <div class="eda-corr-cell" style="background:${corrColor(val)}" title="${short[i]} × ${short[j]}: ${val.toFixed(2)}">${val.toFixed(2)}</div>`).join('');
  });
  html += `</div>`;
  wrap.innerHTML = html;
}

function renderElbowChart(clusData) {
  if (_edaElbowChart) { _edaElbowChart.destroy(); _edaElbowChart = null; }
  const canvas = document.getElementById('eda-elbow-chart');
  if (!canvas) return;
  const curve = clusData.curva_codo;

  _edaElbowChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: curve.map(c => `k=${c.k}`),
      datasets: [{
        data: curve.map(c => c.inercia),
        borderColor: '#6c63ff',
        backgroundColor: hexToRgba('#6c63ff', 0.1),
        borderWidth: 2,
        pointRadius: curve.map(c => c.k === clusData.best_k ? 6 : 4),
        pointBackgroundColor: curve.map(c => c.k === clusData.best_k ? '#15b168' : '#6c63ff'),
        pointBorderColor: '#1e1e1e',
        pointBorderWidth: 2,
        tension: 0.25,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend:  { display: false },
        tooltip: { callbacks: { label: ctx => ` Inercia: ${ctx.raw}` } },
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#f0f0f0', font: { size: 10 } }, border: { color: 'rgba(255,255,255,0.3)' } },
        y: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#f0f0f0', font: { size: 10 } }, border: { color: 'rgba(255,255,255,0.3)' } },
      },
    },
  });
}

function renderSilhouetteChart(clusData) {
  if (_edaSilChart) { _edaSilChart.destroy(); _edaSilChart = null; }
  const canvas = document.getElementById('eda-sil-chart');
  if (!canvas) return;
  const curve = clusData.curva_codo;

  _edaSilChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: curve.map(c => `k=${c.k}`),
      datasets: [{
        data: curve.map(c => c.silueta),
        borderColor: '#15b168',
        backgroundColor: hexToRgba('#15b168', 0.1),
        borderWidth: 2,
        pointRadius: curve.map(c => c.k === clusData.best_k ? 6 : 4),
        pointBackgroundColor: curve.map(c => c.k === clusData.best_k ? '#15b168' : '#38bdf8'),
        pointBorderColor: '#1e1e1e',
        pointBorderWidth: 2,
        tension: 0.25,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend:  { display: false },
        tooltip: { callbacks: { label: ctx => ` Silueta: ${ctx.raw}` } },
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#f0f0f0', font: { size: 10 } }, border: { color: 'rgba(255,255,255,0.3)' } },
        y: { min: -1, max: 1, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#f0f0f0', font: { size: 10 } }, border: { color: 'rgba(255,255,255,0.3)' } },
      },
    },
  });
}

function renderClusterScatter(clusData) {
  if (_edaScatterChart) { _edaScatterChart.destroy(); _edaScatterChart = null; }
  const canvas = document.getElementById('eda-cluster-scatter');
  if (!canvas) return;

  const byCluster = {};
  clusData.teams.forEach(t => {
    (byCluster[t.cluster] ||= []).push(t);
  });

  const datasets = Object.entries(byCluster).map(([cl, teams]) => {
    const idx = parseInt(cl, 10);
    const col = clusterColor(idx);
    return {
      label:           teams[0].label,
      data:            teams.map(t => ({ x: t.x, y: t.y, equipo: t.equipo })),
      backgroundColor: hexToRgba(col, 0.85),
      borderColor:     '#1e1e1e',
      borderWidth:     2,
      pointStyle:      clusterShape(idx),
      radius:          6,
      hoverRadius:     8,
      hitRadius:       10,
    };
  });

  _edaScatterChart = new Chart(canvas, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display:  true,
          position: 'bottom',
          labels:   { color: '#f0f0f0', font: { size: 10 }, usePointStyle: true, boxWidth: 8 },
        },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.raw.equipo} · ${ctx.dataset.label}` },
        },
      },
      scales: {
        x: {
          title:  { display: true, text: 'Componente PCA 1', color: '#aaa', font: { size: 10 } },
          grid:   { color: 'rgba(255,255,255,0.08)' },
          ticks:  { color: '#aaa', font: { size: 9 } },
          border: { color: 'rgba(255,255,255,0.3)' },
        },
        y: {
          title:  { display: true, text: 'Componente PCA 2', color: '#aaa', font: { size: 10 } },
          grid:   { color: 'rgba(255,255,255,0.08)' },
          ticks:  { color: '#aaa', font: { size: 9 } },
          border: { color: 'rgba(255,255,255,0.3)' },
        },
      },
    },
  });
}

function renderClusterProfiles(clusData) {
  const wrap = document.getElementById('eda-cluster-profiles');
  if (!wrap) return;
  wrap.innerHTML = clusData.perfiles.map(p => {
    const col = clusterColor(p.cluster);
    return `
      <div class="eda-profile-card">
        <div class="eda-profile-head">
          <span class="eda-profile-dot" style="background:${col}"></span>
          <span class="eda-profile-label">${p.label}</span>
          <span class="eda-profile-size">${p.size} equipos</span>
        </div>
        <div class="eda-profile-metrics">
          <span>Goles/partido <b>${p.goles_avg}</b></span>
          <span>xG/partido <b>${p.xg_avg}</b></span>
          <span>Tiros a puerta <b>${p.tiros_avg}</b></span>
          <span>Faltas <b>${p.faltas_avg}</b></span>
        </div>
        <div class="eda-profile-teams">${p.equipos.join(' · ')}</div>
      </div>`;
  }).join('');
}

async function renderEdaTab() {
  const tabEl = document.getElementById('tab-eda');
  if (!tabEl) return;

  tabEl.innerHTML = `
    <div class="rend-tab-wrap">
      <div class="rend-sub-nav">
        <div class="eda-sub-tabs">
          <button class="eda-sub-tab active" data-section="vars">EDA</button>
          <button class="eda-sub-tab" data-section="cluster">Clustering</button>
        </div>
      </div>
      <div id="eda-tab-content">
        <div class="loading-state"><div class="spinner"></div><span>Calculando...</span></div>
      </div>
    </div>`;

  const content = document.getElementById('eda-tab-content');

  try {
    const [edaRes, clusRes] = await Promise.all([
      fetch(`${API_URL}/eda-summary`),
      fetch(`${API_URL}/clustering`),
    ]);
    if (!edaRes.ok) throw new Error();
    const edaData  = await edaRes.json();
    const clusData = clusRes.ok ? await clusRes.json() : null;

    const varKeys = Object.keys(edaData.variables);

    const varsHtml = `
      <div class="eda-var-tabs">
        ${varKeys.map((k, i) => `<button class="eda-var-tab${i === 0 ? ' active' : ''}" data-var="${k}">${formatShapVar(k)}</button>`).join('')}
      </div>
      <div class="comp-content">
        <div class="eda-grid">
          <div class="eda-card">
            <p class="shap-chart-title">Distribución (Histograma)</p>
            <div class="eda-chart-wrap"><canvas id="eda-hist-chart"></canvas></div>
          </div>
          <div class="eda-card">
            <p class="shap-chart-title">Boxplot · Outliers (1.5·IQR)</p>
            <div id="eda-boxplot-wrap" class="eda-boxplot-wrap"></div>
          </div>
        </div>
        <div id="eda-stats-row" class="eda-stats-row"></div>
        <div class="eda-card eda-corr-card">
          <p class="shap-chart-title">Mapa de Correlación</p>
          <div id="eda-corr-heatmap" class="eda-corr-heatmap"></div>
        </div>
      </div>
      <div class="rend-meta">${edaData.n_observaciones.toLocaleString('es-PE')} observaciones equipo-partido · Liga 1 Perú 2023–2026</div>`;

    let clusterHtml;
    if (clusData) {
      clusterHtml = `
        <div class="comp-content">
          <div class="eda-grid">
            <div class="eda-card">
              <p class="shap-chart-title">Método del Codo — Inercia por k</p>
              <div class="eda-chart-wrap eda-chart-wrap-sm"><canvas id="eda-elbow-chart"></canvas></div>
            </div>
            <div class="eda-card">
              <p class="shap-chart-title">Coeficiente de Silueta por k</p>
              <div class="eda-chart-wrap eda-chart-wrap-sm"><canvas id="eda-sil-chart"></canvas></div>
            </div>
          </div>
          <div class="rend-meta">k óptimo elegido por silueta: <b>${clusData.best_k}</b> (silueta = ${clusData.silueta_best}) · ${clusData.n_equipos} equipos vigentes</div>
          <div class="eda-card">
            <p class="shap-chart-title">Clusters de Equipos — Perfil Ofensivo (PCA 2D)</p>
            <div class="eda-chart-wrap"><canvas id="eda-cluster-scatter"></canvas></div>
          </div>
          <div class="eda-profile-grid" id="eda-cluster-profiles"></div>
        </div>`;
    } else {
      clusterHtml = `<div class="empty-tab">Sin datos de clustering disponibles</div>`;
    }

    content.innerHTML = `
      <div id="eda-section-vars"    class="eda-section comp-section">${varsHtml}</div>
      <div id="eda-section-cluster" class="eda-section comp-section" style="display:none">${clusterHtml}</div>`;

    renderEdaVariable(varKeys[0], edaData);
    renderCorrHeatmap(edaData.correlacion);

    tabEl.querySelectorAll('.eda-var-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        tabEl.querySelectorAll('.eda-var-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderEdaVariable(btn.dataset.var, edaData);
      });
    });

    tabEl.querySelectorAll('.eda-sub-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        tabEl.querySelectorAll('.eda-sub-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const target = btn.dataset.section;
        tabEl.querySelectorAll('.eda-section').forEach(s => {
          s.style.display = s.id === `eda-section-${target}` ? '' : 'none';
        });
        if (target === 'cluster' && clusData && !_edaElbowChart) {
          renderElbowChart(clusData);
          renderSilhouetteChart(clusData);
          renderClusterScatter(clusData);
          renderClusterProfiles(clusData);
        }
      });
    });
  } catch (_) {
    content.innerHTML = '<div class="empty-tab">No se pudo cargar el EDA</div>';
  }
}

// ── RENDIMIENTO TAB ────────────────────────────────────────────────────────
async function renderRendimientoTab() {
  const tabEl = document.getElementById('tab-rendimiento');
  if (!tabEl) return;

  tabEl.innerHTML = `
    <div class="rend-tab-wrap">
      <div class="rend-sub-nav">
        <div class="rend-sub-tabs">
          <button class="rend-sub-tab active" data-section="backtesting">Backtesting</button>
          <button class="rend-sub-tab" data-section="comp">Comparación de Algoritmos</button>
          <button class="rend-sub-tab" data-section="shap">Importancia de Variables</button>
        </div>
      </div>
      <div id="rend-tab-content">
        <div class="loading-state"><div class="spinner"></div><span>Calculando...</span></div>
      </div>
    </div>`;

  const content = document.getElementById('rend-tab-content');

  try {
    const [perfRes, metricsRes, shapRes] = await Promise.all([
      fetch(`${API_URL}/model-performance`),
      fetch(`${API_URL}/model-metrics`),
      fetch(`${API_URL}/shap-values`),
    ]);

    if (!perfRes.ok) throw new Error();
    const perfData    = await perfRes.json();
    const metricsData = metricsRes.ok ? await metricsRes.json() : null;
    const shapData    = shapRes.ok    ? await shapRes.json()    : null;

    function accColor(pct) {
      return pct >= 70 ? 'acc-green' : pct >= 50 ? 'acc-yellow' : 'acc-red';
    }

    // ── Sección Backtesting ──────────────────────────────────────────────
    let backHtml = '';
    if (!perfData.rounds || !perfData.rounds.length) {
      backHtml = `<div style="padding:24px 16px;color:var(--text3);font-size:13px">
        Sin datos post-entrenamiento. Modelos entrenados hasta el 27/04/2026.</div>`;
    } else {
      const { resumen, rounds } = perfData;
      const totalPred = rounds.reduce((s, r) => s + r.total, 0);
      backHtml = `
        <div class="acc-summary">
          <div class="acc-card">
            <span class="acc-card-label">Goles Esperados ≥ 1.5</span>
            <span class="acc-card-value ${accColor(resumen.xg_accuracy)}">${resumen.xg_accuracy}%</span>
            <span class="acc-card-sub">Accuracy global</span>
          </div>
          <div class="acc-card">
            <span class="acc-card-label">Tiros a Puerta ≥ 5</span>
            <span class="acc-card-value ${accColor(resumen.tiros_accuracy)}">${resumen.tiros_accuracy}%</span>
            <span class="acc-card-sub">Accuracy global</span>
          </div>
          <div class="acc-card">
            <span class="acc-card-label">Goles Anotados ≥ 2</span>
            <span class="acc-card-value ${accColor(resumen.goles_accuracy)}">${resumen.goles_accuracy}%</span>
            <span class="acc-card-sub">Accuracy global</span>
          </div>
        </div>
        <div class="rend-meta">${resumen.total_rondas} jornadas evaluadas · ${totalPred} predicciones</div>
        <div class="rend-table-wrap">
          <div class="rend-table-head">
            <span>Ronda</span>
            <span>Semana del</span>
            <span style="text-align:center">n</span>
            <span>Goles Esperados ≥ 1.5</span>
            <span>Tiros a Puerta ≥ 5</span>
            <span>Goles Anotados ≥ 2</span>
          </div>
          ${rounds.map((r, i) => `
            <div class="rend-table-row" style="animation-delay:${i * 0.05}s">
              <span class="rend-jornada">${r.jornada}</span>
              <span class="rend-fecha">${r.fecha}</span>
              <span class="rend-n">${r.total}</span>
              <span class="acc-badge ${accColor(r.xg_pct)}">${r.xg_pct}%</span>
              <span class="acc-badge ${accColor(r.tiros_pct)}">${r.tiros_pct}%</span>
              <span class="acc-badge ${accColor(r.goles_pct)}">${r.goles_pct}%</span>
            </div>`).join('')}
        </div>`;
    }

    // ── Sección Comparación ───────────────────────────────────────────────
    let compHtml = '';
    if (metricsData) {
      const varKeys = Object.keys(metricsData);
      compHtml = `
        <div class="comp-var-tabs" id="comp-var-tabs">
          ${varKeys.map((k, i) => `
            <button class="comp-var-tab${i === 0 ? ' active' : ''}" data-var="${k}">
              ${metricsData[k].label}
            </button>`).join('')}
        </div>
        <div class="comp-content">
          <div class="comp-chart-wrap">
            <canvas id="comp-radar"></canvas>
          </div>
          <div id="comp-legend" class="comp-legend"></div>
          <div class="comp-table-wrap">
            <div class="comp-table-head">
              <span>Modelo</span>
              <span>Accuracy</span>
              <span>Precision</span>
              <span>Recall</span>
              <span>F1</span>
              <span>AUC-ROC</span>
            </div>
            <div id="comp-table-body"></div>
          </div>
        </div>`;
    } else {
      compHtml = `<div style="padding:40px;color:var(--text3);font-size:13px;text-align:center">
        Sin datos de comparación disponibles.</div>`;
    }

    // ── Sección Importancia de Variables (SHAP) ──────────────────────────
    let shapHtml = '';
    if (shapData) {
      const shapTargets = {
        xg:    'Goles Esperados ≥ 1.5',
        tiros: 'Tiros a Puerta ≥ 5',
        goles: 'Goles Anotados ≥ 2',
      };
      shapHtml = `
        <div class="shap-header-row">
          <div class="comp-var-tabs" id="shap-var-tabs">
            ${Object.entries(shapTargets).map(([k, label], i) => `
              <button class="shap-var-tab${i === 0 ? ' active' : ''}" data-shap="${k}">${label}</button>
            `).join('')}
          </div>
          <button class="shap-sort-btn" id="shap-sort-btn">↓ Desc</button>
        </div>
        <div class="comp-content">
          <p class="shap-chart-title">Top 15 Variables más Influyentes · Valor SHAP Promedio</p>
          <div class="shap-chart-wrap">
            <canvas id="shap-bar-chart"></canvas>
          </div>
          <div class="shap-legend">
            <span class="shap-leg-pos">▬ Aumenta probabilidad</span>
            <span class="shap-leg-neg">▬ Disminuye probabilidad</span>
            <span class="shap-leg-neutral">▬ Efecto neutro</span>
            <span class="shap-leg-abbr"><span class="shap-abbr-key">prom. 3</span> = promedio últimos 3 partidos &nbsp;·&nbsp; <span class="shap-abbr-key">prom. 5</span> = promedio últimos 5 partidos</span>
          </div>
        </div>`;
    } else {
      shapHtml = `<div style="padding:40px;color:var(--text3);font-size:13px;text-align:center">
        Sin datos SHAP disponibles.</div>`;
    }

    content.innerHTML = `
      <div id="rend-section-backtesting" class="rend-section">${backHtml}</div>
      <div id="rend-section-comp"        class="rend-section comp-section" style="display:none">${compHtml}</div>
      <div id="rend-section-shap"        class="rend-section comp-section" style="display:none">${shapHtml}</div>`;

    // Sub-tab switching
    document.querySelectorAll('.rend-sub-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.rend-sub-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const target = btn.dataset.section;
        document.querySelectorAll('.rend-section').forEach(s => {
          s.style.display = s.id === `rend-section-${target}` ? '' : 'none';
        });
        // Lazy-init: charts solo cuando el canvas es visible
        if (target === 'comp' && metricsData && !_radarChart) {
          renderCompChart(Object.keys(metricsData)[0] || 'xg', metricsData);
        }
        if (target === 'shap' && shapData && !_shapChart) {
          renderShapChart('xg', shapData);
        }
      });
    });

    // Variable tabs dentro de Comparación
    if (metricsData) {
      document.querySelectorAll('.comp-var-tab').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('.comp-var-tab').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          renderCompChart(btn.dataset.var, metricsData);
        });
      });
    }

    // Target tabs dentro de Importancia de Variables
    if (shapData) {
      document.querySelectorAll('.shap-var-tab').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('.shap-var-tab').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          renderShapChart(btn.dataset.shap, shapData);
        });
      });

      // Botón de orden asc/desc
      const sortBtn = document.getElementById('shap-sort-btn');
      if (sortBtn) {
        sortBtn.addEventListener('click', () => {
          _shapAsc = !_shapAsc;
          const activeTab = document.querySelector('.shap-var-tab.active');
          const target = activeTab ? activeTab.dataset.shap : 'xg';
          renderShapChart(target, shapData);
        });
      }
    }
  } catch (_) {
    content.innerHTML = '<div class="empty-tab">No se pudo cargar el rendimiento del modelo</div>';
  }
}

function computeFilteredStandings(filter) {
  if (filter === 'all' || !Object.keys(MATCHES).length) return standingsData;

  const isHome = filter === 'home';
  const teams  = {};

  Object.values(MATCHES).forEach(roundMatches => {
    roundMatches.forEach(m => {
      if (m.sh === null) return; // partido no jugado

      const rawName = isHome ? m.homeName : m.awayName;
      const team    = TEAM_NAME_MAP[rawName] || rawName;
      const gf      = isHome ? m.sh : m.sa;
      const ga      = isHome ? m.sa : m.sh;

      if (!teams[team]) teams[team] = { pj:0, pg:0, pe:0, pp:0, gf:0, ga:0, forma:[] };
      const t = teams[team];
      t.pj++; t.gf += gf; t.ga += ga;
      if      (gf > ga)  { t.pg++; t.forma.push('V'); }
      else if (gf === ga) { t.pe++; t.forma.push('E'); }
      else               { t.pp++; t.forma.push('D'); }
    });
  });

  const result = Object.entries(teams).map(([name, s]) => {
    const dif = s.gf - s.ga;
    return {
      Equipo:    name,
      PJ:        s.pj,
      PG:        s.pg,
      PE:        s.pe,
      PP:        s.pp,
      DIF:       dif >= 0 ? `+${dif}` : `${dif}`,
      Goles:     `${s.gf}:${s.ga}`,
      Puntos:    s.pg * 3 + s.pe,
      Ultimos_5: s.forma.slice(-5).join(''),
      _dif:      dif,
    };
  });

  // Orden: Pts → DIF → Goles a favor
  result.sort((a, b) => b.Puntos - a.Puntos || b._dif - a._dif || 0);
  result.forEach((r, i) => { r.Posicion = i + 1; });
  return result;
}

function setupSubTabs() {
  document.querySelectorAll('.sub-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      renderStandings(computeFilteredStandings(tab.dataset.filter));
    });
  });
}


function setupRoundNav() {
  document.getElementById('btn-prev').addEventListener('click', () => changeRound(-1));
  document.getElementById('btn-next').addEventListener('click', () => changeRound(+1));
  $roundSelect().addEventListener('change', (e) => {
    currentRound = parseInt(e.target.value);
    renderMatches(currentRound);
    renderDestacado(currentRound);
    if (isPredTabActive()) renderPredictionsTab(currentRound);
  });
}

// ── INIT ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupMainTabs();
  setupSubTabs();
  buildRoundSelect();
  setupRoundNav();

  // Load matches from CSV (renders after load)
  loadMatchesCSV();

  // Load CSV for standings
  loadCSV();

  // Progress bar temporada
  updateProgress();

  // Countdown
  updateCountdown();
  setInterval(updateCountdown, 1000);
});
