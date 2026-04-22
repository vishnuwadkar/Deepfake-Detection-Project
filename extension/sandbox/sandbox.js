/**
 * DeepGuard AI — Sandbox Inference Engine
 * 
 * Runs inside a sandboxed iframe with relaxed CSP (eval allowed).
 * TF.js can fully initialize here including WebGL shader compilation.
 * 
 * Communicates with parent (offscreen document) via postMessage.
 */

let model = null;
let isLoading = false;
const INPUT_SIZE = 224;

// ─── Model Loading ───────────────────────────────────────────────────────────

async function loadModel() {
  if (model) return model;
  if (isLoading) return null;
  
  isLoading = true;
  console.log('[Sandbox] Loading TF.js model...');
  
  try {
    // TF.js can use setBackend here because sandbox allows eval
    try {
      await tf.setBackend('webgl');
      await tf.ready();
    } catch (e) {
      console.warn('[Sandbox] WebGL failed, trying CPU:', e.message);
      await tf.setBackend('cpu');
      await tf.ready();
    }
    console.log('[Sandbox] Backend:', tf.getBackend());

    // Model path relative to sandbox.html
    model = await tf.loadGraphModel('../model/model.json');
    console.log('[Sandbox] Model loaded successfully');
    
    // Warm up
    const warmup = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
    const result = model.predict(warmup);
    const data = await result.data();
    console.log('[Sandbox] Warmup score:', data[0]);
    result.dispose();
    warmup.dispose();
    
    isLoading = false;
    return model;
  } catch (err) {
    console.error('[Sandbox] Model load failed:', err);
    isLoading = false;
    throw err;
  }
}

// ─── Image Preprocessing ─────────────────────────────────────────────────────

function preprocessBase64(base64Data) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      try {
        const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);
        const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
        
        const tensor = tf.tidy(() => {
          const raw = tf.browser.fromPixels(imageData);
          return raw.toFloat().expandDims(0);  // [1, 224, 224, 3] in [0, 255]
        });
        
        resolve(tensor);
      } catch (err) {
        reject(err);
      }
    };
    img.onerror = () => reject(new Error('Failed to decode image'));
    img.src = base64Data;
  });
}

// ─── Inference ───────────────────────────────────────────────────────────────

async function runInference(base64Data) {
  const mdl = await loadModel();
  if (!mdl) throw new Error('Model not loaded');
  
  const inputTensor = await preprocessBase64(base64Data);
  
  try {
    const prediction = mdl.predict(inputTensor);
    const data = await prediction.data();
    const score = data[0];
    prediction.dispose();
    
    return { score };
  } finally {
    inputTensor.dispose();
  }
}

// ─── Message Handler (from parent offscreen document) ────────────────────────

window.addEventListener('message', async (event) => {
  const { type, requestId, imageData } = event.data;
  
  if (type !== 'INFERENCE_REQUEST') return;
  
  try {
    const result = await runInference(imageData);
    
    // Send result back to parent
    parent.postMessage({
      type: 'INFERENCE_RESULT',
      requestId,
      score: result.score,
      success: true,
    }, '*');
  } catch (err) {
    console.error('[Sandbox] Inference error:', err);
    parent.postMessage({
      type: 'INFERENCE_RESULT',
      requestId,
      score: -1,
      success: false,
      error: err.message,
    }, '*');
  }
});

// Signal ready
parent.postMessage({ type: 'SANDBOX_READY' }, '*');

// Pre-load model
loadModel().catch(err => {
  console.warn('[Sandbox] Pre-load failed:', err.message);
});
