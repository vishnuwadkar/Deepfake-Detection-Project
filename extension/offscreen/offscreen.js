/**
 * DeepGuard AI — Offscreen Inference Engine
 * 
 * Runs inside an offscreen document (hidden from user).
 * Loads the TF.js model and performs GPU-accelerated inference
 * on image data received from the service worker.
 * 
 * Flow:
 *   1. Service worker sends RUN_INFERENCE with imageData
 *   2. This script preprocesses the image (resize to 224×224, normalize)
 *   3. Runs model.predict()
 *   4. Returns { score, isAI, confidence }
 */

// ─── State ───────────────────────────────────────────────────────────────────

let model = null;
let isLoadingModel = false;
let modelLoadPromise = null;

const MODEL_PATH = chrome.runtime.getURL('model/model.json');
const INPUT_SIZE = 224;

// ─── Model Loading ───────────────────────────────────────────────────────────

async function loadModel() {
  if (model) return model;
  if (modelLoadPromise) return modelLoadPromise;

  isLoadingModel = true;
  console.log('[DeepGuard] Loading TF.js model...');

  modelLoadPromise = (async () => {
    try {
      // Set backend — prefer WebGL for GPU acceleration
      await tf.setBackend('webgl');
      await tf.ready();
      console.log('[DeepGuard] TF.js backend:', tf.getBackend());

      // Load the converted model
      model = await tf.loadGraphModel(MODEL_PATH);
      
      // Warm up with a dummy tensor to compile shaders
      const warmup = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
      const warmupResult = model.predict(warmup);
      warmupResult.dispose();
      warmup.dispose();

      console.log('[DeepGuard] Model loaded and warmed up');
      isLoadingModel = false;
      return model;
    } catch (err) {
      console.error('[DeepGuard] Model load failed:', err);
      isLoadingModel = false;
      modelLoadPromise = null;
      throw err;
    }
  })();

  return modelLoadPromise;
}

// ─── Image Preprocessing ─────────────────────────────────────────────────────

/**
 * Preprocesses a base64 image string into a tensor.
 * EfficientNetV2 expects inputs in [0, 255] range (internal preprocessing).
 */
async function preprocessImage(base64Data) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      try {
        // Draw to offscreen canvas at model input size
        const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);
        
        // Get pixel data and create tensor
        const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
        
        // Create tensor: [1, 224, 224, 3] float32 in [0, 255]
        const tensor = tf.tidy(() => {
          const raw = tf.browser.fromPixels(imageData);  // [224, 224, 3] uint8
          const float = raw.toFloat();                     // [224, 224, 3] float32
          return float.expandDims(0);                      // [1, 224, 224, 3]
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

/**
 * Preprocesses raw ImageData (from canvas capture) into a tensor.
 */
function preprocessImageData(pixelData, width, height) {
  return tf.tidy(() => {
    // Create tensor from raw pixel array
    const raw = tf.tensor3d(new Uint8Array(pixelData), [height, width, 4]);  // RGBA
    const rgb = raw.slice([0, 0, 0], [-1, -1, 3]);      // Drop alpha → [H, W, 3]
    const resized = tf.image.resizeBilinear(rgb, [INPUT_SIZE, INPUT_SIZE]);
    const float = resized.toFloat();
    return float.expandDims(0);                            // [1, 224, 224, 3]
  });
}

// ─── Inference ───────────────────────────────────────────────────────────────

async function runInference(imageInput) {
  const mdl = await loadModel();
  
  let inputTensor;
  
  if (typeof imageInput === 'string') {
    // base64 encoded image
    inputTensor = await preprocessImage(imageInput);
  } else if (imageInput.pixelData) {
    // Raw pixel data from canvas
    inputTensor = preprocessImageData(
      imageInput.pixelData, 
      imageInput.width, 
      imageInput.height
    );
  } else {
    throw new Error('Invalid image input format');
  }
  
  try {
    // Run prediction
    const prediction = mdl.predict(inputTensor);
    const score = (await prediction.data())[0];
    
    prediction.dispose();
    
    // Get current threshold from settings
    const result = await chrome.storage.local.get('settings');
    const threshold = result?.settings?.sensitivity || 0.5;
    
    const isAI = score >= threshold;
    const confidence = isAI ? score : (1 - score);
    
    return {
      score: Math.round(score * 10000) / 10000,
      isAI,
      confidence: Math.round(confidence * 100),
    };
  } finally {
    inputTensor.dispose();
  }
}

// ─── Message Listener ────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== 'RUN_INFERENCE') return;

  const imageInput = message.imageData;

  runInference(imageInput)
    .then((result) => {
      sendResponse({
        ...result,
        elementId: message.elementId,
      });
    })
    .catch((err) => {
      console.error('[DeepGuard] Inference error:', err);
      sendResponse({
        score: -1,
        isAI: false,
        confidence: 0,
        error: err.message,
        elementId: message.elementId,
      });
    });

  return true; // async response
});

// ─── Pre-load model on document ready ────────────────────────────────────────
loadModel().catch(err => {
  console.warn('[DeepGuard] Pre-load failed (will retry on first request):', err.message);
});
