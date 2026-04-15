/**
 * DeepGuard AI — Popup Controller
 * 
 * Handles:
 *   - Loading/saving settings from chrome.storage
 *   - Displaying real-time scan state and results
 *   - Triggering manual scans
 *   - Rendering the detection list
 */

// ─── DOM References ──────────────────────────────────────────────────────────

const elements = {
  // Stats
  statScanned: document.getElementById('statScanned'),
  statSafe: document.getElementById('statSafe'),
  statFlagged: document.getElementById('statFlagged'),

  // Status
  headerStatus: document.getElementById('headerStatus'),

  // Progress
  progressSection: document.getElementById('progressSection'),
  progressFill: document.getElementById('progressFill'),
  progressPercent: document.getElementById('progressPercent'),

  // Buttons
  btnScan: document.getElementById('btnScan'),
  btnClear: document.getElementById('btnClear'),

  // Detections
  detectionList: document.getElementById('detectionList'),

  // Settings
  settingsToggle: document.getElementById('settingsToggle'),
  settingsPanel: document.getElementById('settingsPanel'),
  settingAutoScan: document.getElementById('settingAutoScan'),
  settingOverlays: document.getElementById('settingOverlays'),
  settingScanVideos: document.getElementById('settingScanVideos'),
  settingSensitivity: document.getElementById('settingSensitivity'),
  sensitivityValue: document.getElementById('sensitivityValue'),
};

// ─── State ───────────────────────────────────────────────────────────────────

let currentTabId = null;
let pollInterval = null;

// ─── Initialize ──────────────────────────────────────────────────────────────

async function init() {
  // Get current tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  currentTabId = tab?.id;

  // Load settings
  await loadSettings();

  // Load current state
  await refreshState();

  // Start polling for updates while popup is open
  pollInterval = setInterval(refreshState, 1000);

  // Bind event listeners
  bindEvents();
}

// ─── Settings ────────────────────────────────────────────────────────────────

async function loadSettings() {
  try {
    const response = await chrome.runtime.sendMessage({ type: 'GET_SETTINGS' });
    if (response) {
      elements.settingAutoScan.checked = response.autoScan ?? false;
      elements.settingOverlays.checked = response.showOverlays ?? true;
      elements.settingScanVideos.checked = response.scanVideos ?? true;
      elements.settingSensitivity.value = response.sensitivity ?? 0.5;
      elements.sensitivityValue.textContent = (response.sensitivity ?? 0.5).toFixed(2);
    }
  } catch (e) {
    console.warn('Failed to load settings:', e);
  }
}

async function saveSettings() {
  const settings = {
    autoScan: elements.settingAutoScan.checked,
    showOverlays: elements.settingOverlays.checked,
    scanVideos: elements.settingScanVideos.checked,
    sensitivity: parseFloat(elements.settingSensitivity.value),
  };

  try {
    await chrome.runtime.sendMessage({ type: 'SAVE_SETTINGS', settings });
  } catch (e) {
    console.warn('Failed to save settings:', e);
  }
}

// ─── State Refresh ───────────────────────────────────────────────────────────

async function refreshState() {
  if (!currentTabId) return;

  try {
    const state = await chrome.runtime.sendMessage({
      type: 'GET_STATE',
      tabId: currentTabId,
    });

    if (!state) return;

    // Update stats
    elements.statScanned.textContent = state.scanned || 0;
    elements.statSafe.textContent = (state.scanned || 0) - (state.flagged || 0);
    elements.statFlagged.textContent = state.flagged || 0;

    // Update status indicator
    updateStatusIndicator(state);

    // Update progress bar
    if (state.scanning && state.total > 0) {
      elements.progressSection.style.display = 'block';
      const pct = Math.round((state.scanned / state.total) * 100);
      elements.progressFill.style.width = `${pct}%`;
      elements.progressPercent.textContent = `${pct}%`;
    } else {
      elements.progressSection.style.display = 'none';
    }

    // Update detection list
    renderDetections(state.results || []);

  } catch (e) {
    // Tab might not have our content script
  }
}

function updateStatusIndicator(state) {
  const statusDot = elements.headerStatus.querySelector('.status-dot');
  const statusText = elements.headerStatus.querySelector('.status-text');

  statusDot.className = 'status-dot';

  if (state.scanning) {
    statusDot.classList.add('status-scanning');
    statusText.textContent = 'Scanning...';
  } else if (state.flagged > 0) {
    statusDot.classList.add('status-alert');
    statusText.textContent = `${state.flagged} flagged`;
  } else if (state.scanned > 0) {
    statusDot.classList.add('status-complete');
    statusText.textContent = 'All clear';
  } else {
    statusDot.classList.add('status-idle');
    statusText.textContent = 'Ready';
  }
}

// ─── Detection List ──────────────────────────────────────────────────────────

function renderDetections(results) {
  if (!results || results.length === 0) {
    elements.detectionList.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🛡️</div>
        <p>No media scanned yet</p>
        <p class="empty-hint">Click "Scan This Page" to begin analysis</p>
      </div>
    `;
    return;
  }

  // Show most recent first, limit to 20
  const recent = [...results].reverse().slice(0, 20);

  elements.detectionList.innerHTML = recent.map((item) => {
    const isAI = item.isAI;
    const icon = isAI ? '⚠' : '✓';
    const iconClass = isAI ? 'danger' : 'safe';
    const scoreClass = isAI ? 'danger' : 'safe';
    const label = isAI ? 'AI Generated' : 'Authentic';
    const typeLabel = item.mediaType === 'video' ? '🎬 Video' : '🖼️ Image';

    // Extract filename from src
    let name = 'Unknown';
    try {
      const url = new URL(item.src);
      name = url.pathname.split('/').pop() || url.hostname;
    } catch {
      name = item.src?.substring(0, 30) || item.elementId || 'Media';
    }
    if (name.length > 28) name = name.substring(0, 28) + '…';

    return `
      <div class="detection-item">
        <div class="detection-item-icon ${iconClass}">${icon}</div>
        <div class="detection-item-info">
          <div class="detection-item-name" title="${item.src || ''}">${name}</div>
          <div class="detection-item-meta">${typeLabel} · ${label}</div>
        </div>
        <div class="detection-item-score ${scoreClass}">${item.confidence}%</div>
      </div>
    `;
  }).join('');
}

// ─── Event Binding ───────────────────────────────────────────────────────────

function bindEvents() {
  // Scan button
  elements.btnScan.addEventListener('click', async () => {
    if (!currentTabId) return;

    elements.btnScan.disabled = true;
    elements.btnScan.textContent = 'Scanning...';

    try {
      await chrome.runtime.sendMessage({
        type: 'TRIGGER_SCAN',
        tabId: currentTabId,
      });
    } catch (e) {
      console.warn('Scan trigger failed:', e);
    }

    // Re-enable after delay
    setTimeout(() => {
      elements.btnScan.disabled = false;
      elements.btnScan.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M1 4V1h3M12 1h3v3M15 12v3h-3M4 15H1v-3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="8" cy="8" r="3" stroke="currentColor" stroke-width="1.5"/>
        </svg>
        Scan This Page
      `;
    }, 1500);
  });

  // Clear button
  elements.btnClear.addEventListener('click', () => {
    elements.statScanned.textContent = '0';
    elements.statSafe.textContent = '0';
    elements.statFlagged.textContent = '0';
    renderDetections([]);

    updateStatusIndicator({ scanning: false, scanned: 0, flagged: 0 });
  });

  // Settings toggle
  elements.settingsToggle.addEventListener('click', () => {
    const panel = elements.settingsPanel;
    const chevron = elements.settingsToggle.querySelector('.chevron');
    panel.classList.toggle('open');
    chevron.classList.toggle('open');
  });

  // Settings inputs
  elements.settingAutoScan.addEventListener('change', saveSettings);
  elements.settingOverlays.addEventListener('change', saveSettings);
  elements.settingScanVideos.addEventListener('change', saveSettings);
  elements.settingSensitivity.addEventListener('input', () => {
    elements.sensitivityValue.textContent = parseFloat(elements.settingSensitivity.value).toFixed(2);
  });
  elements.settingSensitivity.addEventListener('change', saveSettings);
}

// ─── Cleanup ─────────────────────────────────────────────────────────────────

window.addEventListener('unload', () => {
  if (pollInterval) clearInterval(pollInterval);
});

// ─── Boot ────────────────────────────────────────────────────────────────────
init();
