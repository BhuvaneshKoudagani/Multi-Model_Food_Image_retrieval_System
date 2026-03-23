/* ═══════════════════════════════════════════════════════════════
   FOODLENS — app.js
   Fixes: details section stays hidden when not in details mode
          results section stays hidden when not in retrieval mode
═══════════════════════════════════════════════════════════════ */

const HOST        = window.location.hostname || '127.0.0.1';
const API         = `http://${HOST}:5000`;
const DETAILS_API = `http://${HOST}:5001`;

let selectedFile = null;
let currentMode  = 'image';

// ── Splash Screen ─────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  const splash = document.getElementById('splash');
  const app    = document.getElementById('app');

  // After 2.6 seconds, fade out splash and reveal app
  setTimeout(() => {
    splash.classList.add('exit');
    app.classList.remove('app-hidden');
    app.classList.add('app-visible');
  }, 2600);

  // Remove splash from DOM after transition
  setTimeout(() => {
    splash.style.display = 'none';
  }, 3400);
});

// ── Status Check ──────────────────────────────────────────────
async function checkStatus() {
  try {
    const r = await fetch(`${API}/api/status`, { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    document.getElementById('statusDot').classList.add('online');
    document.getElementById('statusText').textContent =
      `Online · ${d.indexed_images ?? 0} images`;
  } catch {
    document.getElementById('statusDot').classList.remove('online');
    document.getElementById('statusText').textContent = 'Offline';
  }
}
checkStatus();
setInterval(checkStatus, 15000);

// ── Mode Switching ─────────────────────────────────────────────
function switchMode(mode) {
  currentMode = mode;

  // Update nav tabs
  document.querySelectorAll('.nav-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.mode === mode);
  });

  // Show/hide panels
  document.querySelectorAll('.panel').forEach(p => {
    p.classList.toggle('active', p.id === `panel-${mode}`);
  });

  // *** KEY FIX: Hide both result sections when switching modes ***
  hideAllResults();
  clearAllErrors();
}

function hideAllResults() {
  document.getElementById('resultsSection').classList.remove('show');
  document.getElementById('detailsSection').classList.remove('show');
}

function clearAllErrors() {
  ['image', 'text', 'generate', 'details'].forEach(clearError);
}

// ── File Select & Upload ──────────────────────────────────────
function onFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;
  setFile(file);
}

function setFile(file) {
  selectedFile = file;
  const placeholder = document.getElementById('uploadPlaceholder');
  const preview     = document.getElementById('uploadPreview');
  const img         = document.getElementById('previewImg');
  const nameEl      = document.getElementById('previewName');

  img.src = URL.createObjectURL(file);
  nameEl.textContent = file.name;
  placeholder.style.display = 'none';
  preview.style.display     = 'block';
  document.getElementById('btnImageSearch').disabled = false;
}

function clearImage(e) {
  e.stopPropagation();
  selectedFile = null;
  document.getElementById('imageFile').value = '';
  document.getElementById('uploadPlaceholder').style.display = 'block';
  document.getElementById('uploadPreview').style.display     = 'none';
  document.getElementById('btnImageSearch').disabled = true;
}

// Drag & drop
const zone = document.getElementById('uploadZone');
zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('dragover'); });
zone.addEventListener('dragleave', ()  => zone.classList.remove('dragover'));
zone.addEventListener('drop', e => {
  e.preventDefault();
  zone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) setFile(file);
});

// ── Suggestion Chips ──────────────────────────────────────────
function fillText(val) {
  document.getElementById('textInput').value = val;
  document.getElementById('textInput').focus();
}
function fillDetails(val) {
  document.getElementById('detailsInput').value = val;
  document.getElementById('detailsInput').focus();
}

// ── Search by Image ───────────────────────────────────────────
async function searchByImage() {
  if (!selectedFile) return;
  clearError('image');
  showLoading('Encoding image with CLIP ViT…');
  try {
    const fd = new FormData();
    fd.append('image', selectedFile);
    const r = await fetch(`${API}/api/retrieve/image`, { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Server error');
    renderResults({
      type: 'image',
      queryImg: `data:image/jpeg;base64,${d.query_image}`,
      queryLabel: selectedFile.name,
      results: d.results
    });
  } catch (e) {
    showError('image', e.message);
  } finally {
    hideLoading();
  }
}

// ── Search by Text ────────────────────────────────────────────
async function searchByText() {
  const text = document.getElementById('textInput').value.trim();
  if (!text) { showError('text', 'Please enter a food description.'); return; }
  clearError('text');
  showLoading(`Searching for "${text}"…`);
  try {
    const r = await fetch(`${API}/api/retrieve/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Server error');
    renderResults({ type: 'text', queryLabel: text, results: d.results });
  } catch (e) {
    showError('text', e.message);
  } finally {
    hideLoading();
  }
}

// ── Generate & Retrieve ───────────────────────────────────────
async function generateAndSearch() {
  const food = document.getElementById('generateInput').value.trim();
  if (!food) { showError('generate', 'Please enter a food name.'); return; }
  clearError('generate');
  showLoading(`Generating "${food}" with FLUX… (may take ~30s)`);
  try {
    const r = await fetch(`${API}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ food_name: food })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Server error');
    renderResults({
      type: 'generate',
      queryImg: `data:image/jpeg;base64,${d.generated_image}`,
      queryLabel: `Generated: ${food}`,
      results: d.results
    });
  } catch (e) {
    showError('generate', e.message);
  } finally {
    hideLoading();
  }
}

// ── Render Retrieval Results ──────────────────────────────────
function renderResults({ type, queryImg, queryLabel, results }) {
  // Hide details section — only show retrieval results
  document.getElementById('detailsSection').classList.remove('show');

  // Query card
  const qd = document.getElementById('queryDisplay');
  if (type === 'text') {
    qd.innerHTML = `
      <div class="query-icon-badge">📝</div>
      <div class="query-meta">
        <div class="query-meta-label">Text Query</div>
        <div class="query-meta-val">"${escHtml(queryLabel)}"</div>
      </div>`;
  } else {
    const tag = type === 'generate' ? 'Generated Image' : 'Query Image';
    qd.innerHTML = `
      <img src="${queryImg}" alt="query" onclick="openLightbox('${queryImg}')"/>
      <div class="query-meta">
        <div class="query-meta-label">${tag}</div>
        <div class="query-meta-val">${escHtml(queryLabel)}</div>
      </div>`;
  }

  // Grid
  const grid   = document.getElementById('resultsGrid');
  grid.innerHTML = '';
  const medals = ['🥇', '🥈', '🥉'];

  results.forEach((item, i) => {
    const card   = document.createElement('div');
    card.className = 'result-card';
    const imgSrc = `data:image/jpeg;base64,${item.image_b64}`;
    card.onclick = () => openLightbox(imgSrc);
    const pct    = Math.round(item.score * 100);
    card.innerHTML = `
      <img src="${imgSrc}" alt="${escHtml(item.category)}" loading="lazy"/>
      <div class="result-meta">
        <div class="result-rank-row">
          <span class="result-rank-badge">Rank ${item.rank}</span>
          <span class="result-medal">${medals[i] || ''}</span>
        </div>
        <div class="result-category">${item.category.replace(/_/g,' ')}</div>
        <div class="score-bar-wrap">
          <div class="score-bar" style="width:0%" data-pct="${pct}"></div>
        </div>
        <div class="score-val">Similarity: ${item.score}</div>
      </div>`;
    grid.appendChild(card);
  });

  document.getElementById('resultsCount').textContent = `${results.length} results`;
  document.getElementById('resultsSection').classList.add('show');

  setTimeout(() => {
    document.querySelectorAll('.score-bar').forEach(b => {
      b.style.width = b.dataset.pct + '%';
    });
  }, 120);

  document.getElementById('resultsSection')
    .scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Food Details ──────────────────────────────────────────────
async function getFoodDetails() {
  const food = document.getElementById('detailsInput').value.trim();
  if (!food) { showError('details', 'Please enter a food name.'); return; }
  clearError('details');

  // Hide retrieval results when doing food details
  document.getElementById('resultsSection').classList.remove('show');
  document.getElementById('detailsSection').classList.remove('show');

  showLoading(`Generating "${food}" image and analysing with AI… (may take ~30s)`);

  try {
    const r = await fetch(`${DETAILS_API}/api/food-details`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ food_name: food })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Server error');
    renderFoodDetails(d);
  } catch (e) {
    showError('details', e.message);
  } finally {
    hideLoading();
  }
}

function renderFoodDetails({ image_b64, details }) {
  const imgSrc = `data:image/jpeg;base64,${image_b64}`;

  // Image
  document.getElementById('detailsImg').src = imgSrc;

  // Header
  document.getElementById('detailsName').textContent    = details.name || '—';
  document.getElementById('detailsCuisine').textContent = details.cuisine || '';
  document.getElementById('detailsDesc').textContent    = details.description || '';

  // Stats
  document.getElementById('statCalories').textContent =
    details.calories?.per_serving ? `${details.calories.per_serving} kcal` : '—';
  document.getElementById('statPrice').textContent     = details.price?.restaurant || '—';
  document.getElementById('statHomemade').textContent  = details.price?.homemade   || '—';
  document.getElementById('statTime').textContent      = details.prep_time         || '—';

  if (details.calories?.serving_size) {
    document.getElementById('servingSize').textContent = `per ${details.calories.serving_size}`;
  }

  // Nutrition
  const nutr = details.nutrition || {};
  const nutritionGrid = document.getElementById('nutritionGrid');
  const nutrItems = [
    { label: 'Protein', val: nutr.protein,      unit: 'g' },
    { label: 'Carbs',   val: nutr.carbohydrates, unit: 'g' },
    { label: 'Fat',     val: nutr.fat,           unit: 'g' },
    { label: 'Fiber',   val: nutr.fiber,         unit: 'g' },
    { label: 'Sugar',   val: nutr.sugar,         unit: 'g' },
  ];
  nutritionGrid.innerHTML = nutrItems.map(n => `
    <div class="nutr-item">
      <div class="nutr-val">${n.val ?? '—'}${n.val ? n.unit : ''}</div>
      <div class="nutr-lbl">${n.label}</div>
    </div>`).join('');

  // Ingredients
  const ingList = document.getElementById('ingredientsList');
  ingList.innerHTML = (details.main_ingredients || [])
    .map(i => `<span class="ing-chip">${escHtml(i)}</span>`).join('');

  // Allergens
  const algList = document.getElementById('allergensList');
  const algCard = document.getElementById('allergensCard');
  algList.innerHTML = '';
  if (details.allergens?.length) {
    algList.innerHTML = details.allergens
      .map(a => `<span class="alg-chip">⚠️ ${escHtml(a)}</span>`).join('');
    algCard.style.display = 'block';
  } else {
    algCard.style.display = 'none';
  }

  // Tags
  const tagsEl = document.getElementById('detailsTags');
  tagsEl.innerHTML = '';
  (details.health_tags || []).forEach(t => {
    tagsEl.innerHTML += `<span class="det-tag green">${escHtml(t)}</span>`;
  });
  if (details.course || details.cuisine) {
    tagsEl.innerHTML += `<span class="det-tag">${escHtml(details.course || details.cuisine)}</span>`;
  }

  // Fun fact
  document.getElementById('funFact').textContent = details.fun_fact || '';

  // *** Show details section, ensure retrieval results are hidden ***
  document.getElementById('resultsSection').classList.remove('show');
  document.getElementById('detailsSection').classList.add('show');
  document.getElementById('detailsSection')
    .scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── UI Helpers ────────────────────────────────────────────────
function showLoading(msg) {
  document.getElementById('loadingMsg').textContent = msg || 'Processing…';
  document.getElementById('loading').classList.add('show');
}
function hideLoading() {
  document.getElementById('loading').classList.remove('show');
}
function showError(mode, msg) {
  const el = document.getElementById(`err-${mode}`);
  el.textContent = '⚠️  ' + msg;
  el.classList.add('show');
}
function clearError(mode) {
  const el = document.getElementById(`err-${mode}`);
  if (!el) return;
  el.textContent = '';
  el.classList.remove('show');
}
function escHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

// ── Lightbox ──────────────────────────────────────────────────
function openLightbox(src) {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightbox').classList.add('show');
  document.body.style.overflow = 'hidden';
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('show');
  document.body.style.overflow = '';
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
});