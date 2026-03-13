const API = 'http://127.0.0.1:5000';
let selectedFile = null;

// ── Status check ──────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    const d = await r.json();
    document.getElementById('statusDot').className = 'status-dot online';
    document.getElementById('statusText').textContent =
      `Online · ${d.indexed_images} images indexed`;
  } catch {
    document.getElementById('statusDot').className = 'status-dot';
    document.getElementById('statusText').textContent = 'Backend offline';
  }
}
checkStatus();
setInterval(checkStatus, 15000);

// ── Mode switching ────────────────────────────────────────────────────────────
function switchMode(mode) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
  document.getElementById(`panel-${mode}`).classList.add('active');
  hideResults();
}

// ── File select ───────────────────────────────────────────────────────────────
function onFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;
  selectedFile = file;
  const prev = document.getElementById('previewImg');
  prev.src = URL.createObjectURL(file);
  prev.style.display = 'block';
  document.getElementById('btnImageSearch').disabled = false;
}

// Drag & drop
const zone = document.getElementById('uploadZone');
zone.addEventListener('dragover', e => {
  e.preventDefault();
  zone.classList.add('dragover');
});
zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
zone.addEventListener('drop', e => {
  e.preventDefault();
  zone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    selectedFile = file;
    const prev = document.getElementById('previewImg');
    prev.src = URL.createObjectURL(file);
    prev.style.display = 'block';
    document.getElementById('btnImageSearch').disabled = false;
  }
});

// ── Search by Image ───────────────────────────────────────────────────────────
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

// ── Search by Text ────────────────────────────────────────────────────────────
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

// ── Generate & Retrieve ───────────────────────────────────────────────────────
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

// ── Render Results ────────────────────────────────────────────────────────────
function renderResults({ type, queryImg, queryLabel, results }) {
  const qd = document.getElementById('queryDisplay');

  if (type === 'text') {
    qd.innerHTML = `
      <div class="query-text-badge">📝</div>
      <div class="query-info">
        <div class="query-label">Text Query</div>
        <div class="query-value">"${queryLabel}"</div>
      </div>`;
  } else {
    const tag = type === 'generate' ? 'Generated Image' : 'Query Image';
    qd.innerHTML = `
      <img src="${queryImg}" alt="query"
           onclick="openLightbox('${queryImg}')"
           style="cursor:zoom-in"/>
      <div class="query-info">
        <div class="query-label">${tag}</div>
        <div class="query-value">${queryLabel}</div>
      </div>`;
  }

  const grid = document.getElementById('resultsGrid');
  grid.innerHTML = '';
  const medals = ['🥇', '🥈', '🥉'];

  results.forEach((item, i) => {
    const card = document.createElement('div');
    card.className = 'result-card';
    const imgSrc = `data:image/jpeg;base64,${item.image_b64}`;
    card.onclick = () => openLightbox(imgSrc);
    const pct = Math.round(item.score * 100);
    card.innerHTML = `
      <img src="${imgSrc}" alt="${item.category}" loading="lazy"/>
      <div class="result-meta">
        <div class="result-rank">${medals[i] || `#${item.rank}`} Rank ${item.rank}</div>
        <div class="result-category">${item.category.replace(/_/g, ' ')}</div>
        <div class="score-bar-wrap">
          <div class="score-bar" style="width:0%" data-pct="${pct}"></div>
        </div>
        <div class="score-val">Similarity: ${item.score}</div>
      </div>`;
    grid.appendChild(card);
  });

  document.getElementById('resultsCount').textContent = `${results.length} results`;
  document.getElementById('resultsSection').classList.add('show');

  // Animate score bars after paint
  setTimeout(() => {
    document.querySelectorAll('.score-bar').forEach(bar => {
      bar.style.width = bar.dataset.pct + '%';
    });
  }, 100);

  document.getElementById('resultsSection')
    .scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function showLoading(msg) {
  document.getElementById('loadingMsg').textContent = msg;
  document.getElementById('loading').classList.add('show');
  hideResults();
}
function hideLoading() {
  document.getElementById('loading').classList.remove('show');
}
function hideResults() {
  document.getElementById('resultsSection').classList.remove('show');
}
function showError(mode, msg) {
  const el = document.getElementById(`err-${mode}`);
  el.textContent = '⚠️ ' + msg;
  el.classList.add('show');
}
function clearError(mode) {
  const el = document.getElementById(`err-${mode}`);
  el.textContent = '';
  el.classList.remove('show');
}

// ── Lightbox ──────────────────────────────────────────────────────────────────
function openLightbox(src) {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightbox').classList.add('show');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('show');
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
});