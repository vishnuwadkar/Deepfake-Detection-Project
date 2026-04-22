/**
 * DeepGuard AI — Offscreen Document (Message Relay)
 * 
 * This offscreen document acts as a bridge between the background
 * service worker and the sandboxed iframe where TF.js runs.
 * 
 * Flow:
 *   1. Background sends RUN_INFERENCE via chrome.runtime.onMessage
 *   2. This script forwards the request to the sandbox iframe via postMessage
 *   3. Sandbox runs TF.js inference and sends result back via postMessage
 *   4. This script relays the result back to background via sendResponse
 */

// ─── State ───────────────────────────────────────────────────────────────────

const pendingRequests = new Map();  // requestId → { sendResponse, timeout }
let requestCounter = 0;
let sandboxReady = false;
let sandboxIframe = null;

// ─── Sandbox Iframe Setup ────────────────────────────────────────────────────

function createSandbox() {
  sandboxIframe = document.createElement('iframe');
  sandboxIframe.src = chrome.runtime.getURL('sandbox/sandbox.html');
  sandboxIframe.style.display = 'none';
  document.body.appendChild(sandboxIframe);
  console.log('[Offscreen] Sandbox iframe created');
}

// ─── Handle messages from sandbox iframe ─────────────────────────────────────

window.addEventListener('message', (event) => {
  const data = event.data;
  
  if (data.type === 'SANDBOX_READY') {
    sandboxReady = true;
    console.log('[Offscreen] Sandbox is ready');
    return;
  }
  
  if (data.type === 'INFERENCE_RESULT') {
    const pending = pendingRequests.get(data.requestId);
    if (pending) {
      clearTimeout(pending.timeout);
      pendingRequests.delete(data.requestId);
      
      if (data.success) {
        const score = data.score;
        const threshold = pending.threshold || 0.5;
        const isAI = score >= threshold;
        const confidence = isAI ? score : (1 - score);
        
        console.log('[Offscreen] Inference result: score=', score);
        
        pending.resolve({
          score: Math.round(score * 10000) / 10000,
          isAI,
          confidence: Math.round(confidence * 100),
          elementId: pending.elementId,
        });
      } else {
        console.error('[Offscreen] Inference failed:', data.error);
        pending.resolve({
          score: -1,
          isAI: false,
          confidence: 0,
          error: data.error,
          elementId: pending.elementId,
        });
      }
    }
  }
});

// ─── Wait for sandbox to be ready ────────────────────────────────────────────

function waitForSandbox(timeoutMs = 15000) {
  if (sandboxReady) return Promise.resolve();
  
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const check = setInterval(() => {
      if (sandboxReady) {
        clearInterval(check);
        resolve();
      } else if (Date.now() - start > timeoutMs) {
        clearInterval(check);
        reject(new Error('Sandbox initialization timeout'));
      }
    }, 100);
  });
}

// ─── Message Listener (from background service worker) ───────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== 'RUN_INFERENCE') return;

  const imageData = message.imageData;
  const elementId = message.elementId;

  (async () => {
    try {
      // Wait for sandbox to be ready
      await waitForSandbox();
      
      // Get threshold from settings
      let threshold = 0.5;
      try {
        const result = await chrome.storage.local.get('settings');
        threshold = result?.settings?.sensitivity || 0.5;
      } catch (e) { /* use default */ }
      
      const requestId = ++requestCounter;
      
      // Create a promise that resolves when sandbox responds
      const resultPromise = new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          pendingRequests.delete(requestId);
          reject(new Error('Inference timeout (30s)'));
        }, 30000);
        
        pendingRequests.set(requestId, {
          resolve,
          timeout,
          elementId,
          threshold,
        });
      });
      
      // Send to sandbox iframe
      sandboxIframe.contentWindow.postMessage({
        type: 'INFERENCE_REQUEST',
        requestId,
        imageData,
      }, '*');
      
      const result = await resultPromise;
      sendResponse(result);
    } catch (err) {
      console.error('[Offscreen] Error:', err);
      sendResponse({
        score: -1,
        isAI: false,
        confidence: 0,
        error: err.message,
        elementId,
      });
    }
  })();

  return true; // async response
});

// ─── Initialize ──────────────────────────────────────────────────────────────
createSandbox();
