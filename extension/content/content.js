/**
 * DeepGuard AI — Content Script
 * 
 * Injected into every web page. Responsibilities:
 *   1. Scan all visible images on the page
 *   2. Watch for dynamically added images via MutationObserver
 *   3. Extract video frames at intervals for analysis
 *   4. Send media data to the service worker for inference
 *   5. Inject visual overlay badges on scanned elements
 *   6. Manage a processing queue with throttling
 */

// ─── State ───────────────────────────────────────────────────────────────────

const processedElements = new WeakSet();
const resultCache = new Map();  // url-hash → { score, isAI, confidence }
const processingQueue = [];
let isProcessing = false;
let activeInferences = 0;
let deepguardIdCounter = 0;

let settings = {
  autoScan: false,
  sensitivity: 0.5,
  showOverlays: true,
  minImageSize: 64,
  maxConcurrent: 3,
  scanVideos: true,
  videoFrameInterval: 2000,
};

// Track active video observers
const videoIntervals = new Map();

// ─── Initialize ──────────────────────────────────────────────────────────────

async function initialize() {
  // Load settings
  try {
    const response = await chrome.runtime.sendMessage({ type: 'GET_SETTINGS' });
    if (response) {
      settings = { ...settings, ...response };
    }
  } catch (e) {
    // Extension context may not be ready yet
  }

  // Auto-scan if enabled
  if (settings.autoScan) {
    setTimeout(() => scanPage(), 1000); // Delay to let page settle
  }

  // Set up MutationObserver for dynamic content
  setupMutationObserver();
}

// ─── Unique ID Generator ────────────────────────────────────────────────────

function getElementId(el) {
  if (!el.dataset.deepguardId) {
    el.dataset.deepguardId = `dg-${++deepguardIdCounter}`;
  }
  return el.dataset.deepguardId;
}

// ─── Image Discovery ────────────────────────────────────────────────────────

function discoverMedia() {
  const media = [];

  // 1. Standard <img> elements
  document.querySelectorAll('img').forEach((img) => {
    if (isValidImageTarget(img)) {
      media.push({ element: img, type: 'image' });
    }
  });

  // 2. <picture> elements (grab their <img> child)
  document.querySelectorAll('picture img').forEach((img) => {
    if (isValidImageTarget(img)) {
      media.push({ element: img, type: 'image' });
    }
  });

  // 3. <video> elements
  if (settings.scanVideos) {
    document.querySelectorAll('video').forEach((video) => {
      if (isValidVideoTarget(video)) {
        media.push({ element: video, type: 'video' });
      }
    });
  }

  // 4. Elements with CSS background-image
  document.querySelectorAll('[style*="background-image"]').forEach((el) => {
    const bg = getComputedStyle(el).backgroundImage;
    if (bg && bg !== 'none' && !processedElements.has(el)) {
      const url = extractUrlFromCss(bg);
      if (url) {
        media.push({ element: el, type: 'background', src: url });
      }
    }
  });

  return media;
}

function isValidImageTarget(img) {
  if (processedElements.has(img)) return false;
  if (!img.complete || img.naturalWidth === 0) return false;
  if (img.naturalWidth < settings.minImageSize || img.naturalHeight < settings.minImageSize) return false;
  if (img.closest('.deepguard-overlay')) return false; // Skip our own overlays
  
  // Skip data URIs that are tiny (likely tracking pixels)
  const src = img.src || img.currentSrc || '';
  if (src.startsWith('data:') && src.length < 200) return false;
  
  return true;
}

function isValidVideoTarget(video) {
  if (processedElements.has(video)) return false;
  if (video.videoWidth < settings.minImageSize || video.videoHeight < settings.minImageSize) return false;
  if (video.closest('.deepguard-overlay')) return false;
  return true;
}

function extractUrlFromCss(cssValue) {
  const match = cssValue.match(/url\(["']?(.*?)["']?\)/);
  return match ? match[1] : null;
}

// ─── Image Capture ───────────────────────────────────────────────────────────

async function captureImageAsBase64(element, type, src) {
  try {
    if (type === 'image') {
      return await imageElementToBase64(element);
    } else if (type === 'video') {
      return videoFrameToBase64(element);
    } else if (type === 'background' && src) {
      return await urlToBase64(src);
    }
  } catch (err) {
    console.warn('[DeepGuard] Capture failed:', err.message);
    return null;
  }
}

function imageElementToBase64(img) {
  return new Promise((resolve, reject) => {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      resolve(canvas.toDataURL('image/jpeg', 0.85));
    } catch (err) {
      // CORS: try fetching the image instead
      if (img.src && !img.src.startsWith('data:')) {
        urlToBase64(img.src).then(resolve).catch(reject);
      } else {
        reject(err);
      }
    }
  });
}

function videoFrameToBase64(video) {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.85);
}

async function urlToBase64(url) {
  try {
    const response = await fetch(url, { mode: 'cors' });
    const blob = await response.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } catch {
    // If CORS fails, skip this element
    return null;
  }
}

// ─── Processing Queue ────────────────────────────────────────────────────────

function enqueueMedia(mediaItem) {
  processingQueue.push(mediaItem);
  processQueue();
}

async function processQueue() {
  if (activeInferences >= settings.maxConcurrent) return;
  if (processingQueue.length === 0) {
    if (activeInferences === 0) {
      chrome.runtime.sendMessage({ type: 'SCAN_COMPLETE' }).catch(() => {});
    }
    return;
  }

  const item = processingQueue.shift();
  activeInferences++;

  try {
    await processMediaItem(item);
  } catch (err) {
    console.warn('[DeepGuard] Process error:', err.message);
  } finally {
    activeInferences--;
    processQueue();
  }
}

async function processMediaItem(item) {
  const { element, type, src } = item;
  const elementId = getElementId(element);

  // Check cache
  const cacheKey = (type === 'image' ? (element.src || element.currentSrc) : src) || elementId;
  if (resultCache.has(cacheKey)) {
    const cached = resultCache.get(cacheKey);
    applyOverlay(element, cached);
    return;
  }

  // Mark as processed
  processedElements.add(element);

  // Capture image data
  const base64 = await captureImageAsBase64(element, type, src);
  if (!base64) return;

  // Send to background → offscreen for inference
  try {
    const result = await chrome.runtime.sendMessage({
      type: type === 'video' ? 'SCAN_VIDEO_FRAME' : 'SCAN_IMAGE',
      imageData: base64,
      elementId: elementId,
      src: cacheKey,
    });

    if (result && result.score >= 0) {
      resultCache.set(cacheKey, result);
      applyOverlay(element, result);
    }
  } catch (err) {
    console.warn('[DeepGuard] Message error:', err.message);
  }
}

// ─── Video Frame Scanner ─────────────────────────────────────────────────────

function startVideoScanning(video) {
  const videoId = getElementId(video);
  if (videoIntervals.has(videoId)) return;

  const interval = setInterval(() => {
    if (video.paused || video.ended || !document.contains(video)) {
      clearInterval(interval);
      videoIntervals.delete(videoId);
      return;
    }

    // Remove from processed to allow re-scan of new frame
    processedElements.delete(video);
    enqueueMedia({ element: video, type: 'video' });
  }, settings.videoFrameInterval);

  videoIntervals.set(videoId, interval);
}

// ─── Overlay Injection ───────────────────────────────────────────────────────

function applyOverlay(element, result) {
  if (!settings.showOverlays) return;
  
  const elementId = getElementId(element);
  
  // Remove any existing overlay
  const existing = document.querySelector(`.deepguard-overlay[data-for="${elementId}"]`);
  if (existing) existing.remove();

  // Create overlay badge
  const overlay = document.createElement('div');
  overlay.className = 'deepguard-overlay';
  overlay.dataset.for = elementId;

  const isAI = result.isAI;
  const confidence = result.confidence;
  const score = result.score;

  overlay.classList.add(isAI ? 'deepguard-danger' : 'deepguard-safe');

  overlay.innerHTML = `
    <div class="deepguard-badge">
      <div class="deepguard-badge-icon">${isAI ? '⚠' : '✓'}</div>
      <div class="deepguard-badge-label">${isAI ? 'AI Generated' : 'Authentic'}</div>
    </div>
    <div class="deepguard-detail">
      <div class="deepguard-detail-row">
        <span class="deepguard-detail-label">Confidence</span>
        <span class="deepguard-detail-value">${confidence}%</span>
      </div>
      <div class="deepguard-detail-row">
        <span class="deepguard-detail-label">Raw Score</span>
        <span class="deepguard-detail-value">${score.toFixed(4)}</span>
      </div>
      <div class="deepguard-bar-container">
        <div class="deepguard-bar" style="width: ${Math.round(score * 100)}%"></div>
      </div>
      <div class="deepguard-detail-footer">
        <span>🛡️ DeepGuard AI</span>
      </div>
    </div>
  `;

  // Position overlay relative to the element
  const parent = element.parentElement;
  if (parent) {
    const parentPosition = getComputedStyle(parent).position;
    if (parentPosition === 'static') {
      parent.style.position = 'relative';
    }
    parent.appendChild(overlay);
  }
}

// ─── Page Scanner ────────────────────────────────────────────────────────────

function scanPage() {
  const media = discoverMedia();
  
  if (media.length === 0) return;

  // Notify background of scan start
  chrome.runtime.sendMessage({
    type: 'SCAN_START',
    count: media.length,
  }).catch(() => {});

  // Enqueue all discovered media
  media.forEach((item) => {
    enqueueMedia(item);

    // Start continuous scanning for videos
    if (item.type === 'video' && settings.scanVideos) {
      startVideoScanning(item.element);
    }
  });
}

// ─── MutationObserver ────────────────────────────────────────────────────────

let mutationDebounceTimer = null;
const pendingMutationNodes = new Set();

function setupMutationObserver() {
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType !== Node.ELEMENT_NODE) continue;

        // Check if the node itself is a media element
        if (node.tagName === 'IMG' && isValidImageTarget(node)) {
          pendingMutationNodes.add(node);
        } else if (node.tagName === 'VIDEO' && settings.scanVideos && isValidVideoTarget(node)) {
          pendingMutationNodes.add(node);
        }

        // Check children
        if (node.querySelectorAll) {
          node.querySelectorAll('img').forEach((img) => {
            if (isValidImageTarget(img)) pendingMutationNodes.add(img);
          });
          if (settings.scanVideos) {
            node.querySelectorAll('video').forEach((video) => {
              if (isValidVideoTarget(video)) pendingMutationNodes.add(video);
            });
          }
        }
      }
    }

    // Debounce: batch process after 300ms of inactivity
    if (pendingMutationNodes.size > 0) {
      clearTimeout(mutationDebounceTimer);
      mutationDebounceTimer = setTimeout(() => {
        if (!settings.autoScan) return; // Only auto-process if auto-scan is on

        pendingMutationNodes.forEach((node) => {
          const type = node.tagName === 'VIDEO' ? 'video' : 'image';
          enqueueMedia({ element: node, type });
          if (type === 'video') startVideoScanning(node);
        });
        pendingMutationNodes.clear();
      }, 300);
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });
}

// ─── Message Listener (from popup/background) ───────────────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.type) {
    case 'START_SCAN':
      scanPage();
      sendResponse({ ok: true });
      break;

    case 'SETTINGS_UPDATED':
      settings = { ...settings, ...message.settings };
      
      // Toggle overlays visibility
      document.querySelectorAll('.deepguard-overlay').forEach((overlay) => {
        overlay.style.display = settings.showOverlays ? '' : 'none';
      });
      
      sendResponse({ ok: true });
      break;

    case 'GET_PAGE_STATS':
      {
        const images = document.querySelectorAll('img').length;
        const videos = document.querySelectorAll('video').length;
        const overlays = document.querySelectorAll('.deepguard-overlay').length;
        sendResponse({ images, videos, overlays, cached: resultCache.size });
      }
      break;
  }
});

// ─── Handle images that load after initial scan ──────────────────────────────

document.addEventListener('load', (e) => {
  if (e.target.tagName === 'IMG' && settings.autoScan && isValidImageTarget(e.target)) {
    enqueueMedia({ element: e.target, type: 'image' });
  }
}, true);

// ─── Boot ────────────────────────────────────────────────────────────────────
initialize();
