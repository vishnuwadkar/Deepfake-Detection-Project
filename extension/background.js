/**
 * DeepGuard AI — Background Service Worker
 * 
 * Responsibilities:
 *   - Manages offscreen document lifecycle for TF.js inference
 *   - Routes messages between content script ↔ offscreen document
 *   - Updates badge icon/text with scan results
 *   - Manages per-tab scan state
 *   - Handles user settings via chrome.storage
 */

// ─── Constants ───────────────────────────────────────────────────────────────

const OFFSCREEN_DOCUMENT_PATH = 'offscreen/offscreen.html';
const DEFAULT_SETTINGS = {
  autoScan: false,
  sensitivity: 0.5,         // decision threshold (0=flag everything, 1=flag nothing)
  showOverlays: true,
  minImageSize: 64,          // skip images smaller than 64×64
  maxConcurrent: 3,          // max concurrent inferences
  scanVideos: true,          // enable video frame scanning
  videoFrameInterval: 2000,  // ms between video frame captures
};

// Per-tab scan state: { tabId: { total, scanned, flagged, results[] } }
const tabState = new Map();

// ─── Offscreen Document Management ──────────────────────────────────────────

let creatingOffscreen = null;

async function ensureOffscreenDocument() {
  // Check if already exists
  const exists = await chrome.offscreen.hasDocument?.() || false;
  if (exists) return;

  // Prevent race conditions with multiple create calls
  if (creatingOffscreen) {
    await creatingOffscreen;
    return;
  }

  creatingOffscreen = chrome.offscreen.createDocument({
    url: OFFSCREEN_DOCUMENT_PATH,
    reasons: ['DOM_SCRAPING', 'LOCAL_STORAGE'],
    justification: 'TensorFlow.js model inference requires DOM and WebGL context',
  });

  await creatingOffscreen;
  creatingOffscreen = null;
  console.log('[DeepGuard] Offscreen document created');
}

// ─── Settings ────────────────────────────────────────────────────────────────

async function getSettings() {
  const result = await chrome.storage.local.get('settings');
  return { ...DEFAULT_SETTINGS, ...(result.settings || {}) };
}

async function saveSettings(settings) {
  await chrome.storage.local.set({ settings: { ...DEFAULT_SETTINGS, ...settings } });
}

// ─── Tab State ───────────────────────────────────────────────────────────────

function getTabState(tabId) {
  if (!tabState.has(tabId)) {
    tabState.set(tabId, {
      total: 0,
      scanned: 0,
      flagged: 0,
      results: [],
      scanning: false,
    });
  }
  return tabState.get(tabId);
}

function updateBadge(tabId) {
  const state = getTabState(tabId);
  
  if (state.scanning) {
    chrome.action.setBadgeText({ text: '...', tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#6C63FF', tabId });
  } else if (state.flagged > 0) {
    chrome.action.setBadgeText({ text: String(state.flagged), tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#FF3D71', tabId });
  } else if (state.scanned > 0) {
    chrome.action.setBadgeText({ text: '✓', tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#00D68F', tabId });
  } else {
    chrome.action.setBadgeText({ text: '', tabId });
  }
}

// ─── Message Handling ────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const tabId = sender.tab?.id || message.tabId;

  switch (message.type) {

    // ── Content script found media to scan ───────────────────────────────
    case 'SCAN_IMAGE':
    case 'SCAN_VIDEO_FRAME':
      handleScanRequest(message, tabId).then(sendResponse);
      return true; // async response

    // ── Content script reporting scan start ──────────────────────────────
    case 'SCAN_START':
      {
        const state = getTabState(tabId);
        state.total = message.count;
        state.scanned = 0;
        state.flagged = 0;
        state.results = [];
        state.scanning = true;
        updateBadge(tabId);
        sendResponse({ ok: true });
      }
      break;

    // ── Content script reporting scan complete ───────────────────────────
    case 'SCAN_COMPLETE':
      {
        const state = getTabState(tabId);
        state.scanning = false;
        updateBadge(tabId);
        sendResponse({ ok: true });
      }
      break;

    // ── Popup requesting current state ───────────────────────────────────
    case 'GET_STATE':
      {
        const state = getTabState(tabId);
        sendResponse({ ...state });
      }
      break;

    // ── Popup requesting settings ────────────────────────────────────────
    case 'GET_SETTINGS':
      getSettings().then(sendResponse);
      return true;

    // ── Popup saving settings ────────────────────────────────────────────
    case 'SAVE_SETTINGS':
      saveSettings(message.settings).then(() => {
        // Notify all content scripts of settings change
        chrome.tabs.query({}, (tabs) => {
          tabs.forEach((tab) => {
            chrome.tabs.sendMessage(tab.id, {
              type: 'SETTINGS_UPDATED',
              settings: message.settings,
            }).catch(() => {}); // ignore tabs without content script
          });
        });
        sendResponse({ ok: true });
      });
      return true;

    // ── Popup triggering manual scan ─────────────────────────────────────
    case 'TRIGGER_SCAN':
      chrome.tabs.sendMessage(tabId, { type: 'START_SCAN' }).catch(() => {});
      sendResponse({ ok: true });
      break;

    // ── Offscreen document returning inference result ─────────────────────
    case 'INFERENCE_RESULT':
      // This comes from offscreen.js — route back via stored callback
      // Handled via the Promise in handleScanRequest
      break;

    default:
      break;
  }
});

// ─── Inference Request Handler ───────────────────────────────────────────────

async function handleScanRequest(message, tabId) {
  try {
    await ensureOffscreenDocument();

    const state = getTabState(tabId);
    state.scanning = true;
    updateBadge(tabId);

    // Forward to offscreen document and wait for result
    const result = await chrome.runtime.sendMessage({
      type: 'RUN_INFERENCE',
      imageData: message.imageData,      // base64 or ArrayBuffer
      imageWidth: message.imageWidth,
      imageHeight: message.imageHeight,
      mediaType: message.type === 'SCAN_VIDEO_FRAME' ? 'video' : 'image',
      elementId: message.elementId,
    });

    // Update tab state
    state.scanned++;
    if (result.isAI) {
      state.flagged++;
    }
    state.results.push({
      elementId: message.elementId,
      score: result.score,
      isAI: result.isAI,
      confidence: result.confidence,
      mediaType: message.type === 'SCAN_VIDEO_FRAME' ? 'video' : 'image',
      timestamp: Date.now(),
      src: message.src || '',
    });

    updateBadge(tabId);

    return {
      elementId: message.elementId,
      score: result.score,
      isAI: result.isAI,
      confidence: result.confidence,
    };
  } catch (err) {
    console.error('[DeepGuard] Inference error:', err);
    return {
      elementId: message.elementId,
      score: -1,
      isAI: false,
      confidence: 0,
      error: err.message,
    };
  }
}

// ─── Tab Cleanup ─────────────────────────────────────────────────────────────

chrome.tabs.onRemoved.addListener((tabId) => {
  tabState.delete(tabId);
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === 'loading') {
    tabState.delete(tabId);
    updateBadge(tabId);
  }
});

// ─── Installation ────────────────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(async () => {
  // Initialize default settings
  const existing = await chrome.storage.local.get('settings');
  if (!existing.settings) {
    await saveSettings(DEFAULT_SETTINGS);
  }
  console.log('[DeepGuard AI] Extension installed — v1.0.0');
});
