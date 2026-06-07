const video = document.getElementById('webcam');
const canvas = document.getElementById('output');
const ctx = canvas.getContext('2d');

const gestureDisplay = document.getElementById('gesture-display');
const confidenceDisplay = document.getElementById('confidence-display');
const confidenceBar = document.getElementById('confidence-bar');
const methodDisplay = document.getElementById('method-display');
const languageSelect = document.getElementById('language-select');
const statusText = document.getElementById('status-text');
const statusDot = document.querySelector('.status-dot');

let model;
let lastPredictionTime = 0;
const PREDICTION_INTERVAL_MS = 150; // Throttle requests to ~7 FPS for smooth performance
let isPredicting = false;

// Initialize camera stream
async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 640,
        height: 480,
        facingMode: 'user'
      },
      audio: false
    });
    video.srcObject = stream;
    return new Promise(resolve => {
      video.onloadedmetadata = () => {
        resolve(video);
      };
    });
  } catch (err) {
    console.error("Camera access denied or unavailable:", err);
    gestureDisplay.textContent = "Camera Error";
    confidenceDisplay.textContent = "0.0%";
    methodDisplay.textContent = "Enable camera permissions";
    throw err;
  }
}

// Load TensorFlow.js Handpose model
async function loadModel() {
  try {
    statusText.textContent = "Loading Handpose Model...";
    statusDot.style.background = "var(--warning)";
    
    model = await handpose.load();
    
    statusText.textContent = "Systems Ready";
    statusDot.style.background = "var(--success)";
    console.log('TFJS Handpose model loaded successfully.');
  } catch (err) {
    console.error("Failed to load Handpose model:", err);
    statusText.textContent = "Model Load Failed";
    statusDot.style.background = "#ef4444";
  }
}

// Send landmarks to backend Python API
async function queryBackend(landmarks) {
  const selectedLanguage = languageSelect.value;
  
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        keypoints: landmarks,
        language: selectedLanguage
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    updateUI(result);
    
    // Maintain active server status indicator
    statusText.textContent = "Connected (API Active)";
    statusDot.style.background = "var(--success)";
  } catch (err) {
    console.warn("Backend server connection failed, showing client status:", err);
    statusText.textContent = "Server Offline (Using Fallback)";
    statusDot.style.background = "var(--warning)";
    
    // In case server is offline, we run a client-side simple rule-based mock
    // to keep the UI interactive and impressive!
    runLocalFallbackClassification(landmarks);
  }
}

// Draw skeletons on output canvas overlay
function drawHand(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (!landmarks || landmarks.length === 0) return;
  
  // Joint connection edges mapping
  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // Index
    [0, 9], [9, 10], [10, 11], [11, 12],   // Middle
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17]              // Palm cross connections
  ];

  // Draw connections (bones)
  ctx.strokeStyle = 'rgba(59, 130, 246, 0.6)';
  ctx.lineWidth = 3;
  connections.forEach(([i, j]) => {
    const p1 = landmarks[i];
    const p2 = landmarks[j];
    ctx.beginPath();
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    ctx.stroke();
  });

  // Draw landmarks (joints)
  landmarks.forEach(([x, y, z], idx) => {
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    
    // Accent tip joints differently
    if ([4, 8, 12, 16, 20].includes(idx)) {
      ctx.fillStyle = 'var(--info)';
    } else {
      ctx.fillStyle = 'var(--accent)';
    }
    
    ctx.shadowBlur = 8;
    ctx.shadowColor = 'rgba(59, 130, 246, 0.8)';
    ctx.fill();
    ctx.shadowBlur = 0; // reset
  });
}

// Update UI dashboard values
function updateUI(prediction) {
  const { gesture, confidence, method } = prediction;
  
  // Animate text if gesture changes
  if (gestureDisplay.textContent !== gesture) {
    gestureDisplay.style.transform = 'scale(0.85)';
    setTimeout(() => {
      gestureDisplay.textContent = gesture;
      gestureDisplay.style.transform = 'scale(1)';
    }, 100);
  }
  
  confidenceDisplay.textContent = `${confidence.toFixed(1)}%`;
  confidenceBar.style.width = `${confidence}%`;
  methodDisplay.textContent = method;
  
  // Color code based on confidence
  if (confidence > 85) {
    confidenceDisplay.style.color = 'var(--success)';
  } else if (confidence > 65) {
    confidenceDisplay.style.color = 'var(--info)';
  } else {
    confidenceDisplay.style.color = 'var(--warning)';
  }

  // Highlight corresponding gesture in the help library
  const libraryItems = document.querySelectorAll('.library-item');
  libraryItems.forEach(item => {
    const gestureKey = item.getAttribute('data-gesture');
    if (gestureKey && gesture.toLowerCase().includes(gestureKey.toLowerCase())) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });
}

// Simple rule-based calculation local callback if Python backend is offline
function runLocalFallbackClassification(landmarks) {
  // Translate landmarks relative to wrist (index 0)
  const wrist = landmarks[0];
  const relative = landmarks.map(p => [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]]);
  
  // Wrist-to-MCP scale
  const scale = Math.hypot(relative[9][0], relative[9][1], relative[9][2]);
  if (scale < 0.1) return;
  
  // Normalize keypoints
  const normalized = relative.map(p => [p[0] / scale, p[1] / scale, p[2] / scale]);
  
  const dist = (p1, p2) => Math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
  const distFromWrist = p => Math.hypot(p[0], p[1], p[2]);
  
  const d_thumb = distFromWrist(normalized[4]);
  const d_index = distFromWrist(normalized[8]);
  const d_middle = distFromWrist(normalized[12]);
  const d_ring = distFromWrist(normalized[16]);
  const d_pinky = distFromWrist(normalized[20]);
  
  const thumb_extended = d_thumb > distFromWrist(normalized[3]) + 0.15;
  const index_extended = d_index > distFromWrist(normalized[6]) + 0.2;
  const middle_extended = d_middle > distFromWrist(normalized[10]) + 0.2;
  const ring_extended = d_ring > distFromWrist(normalized[14]) + 0.2;
  const pinky_extended = d_pinky > distFromWrist(normalized[18]) + 0.2;
  
  let gesture = "Unknown Gesture";
  let confidence = 50;

  if (!index_extended && !middle_extended && !ring_extended && !pinky_extended && !thumb_extended) {
    gesture = "Fist";
    confidence = 90;
  } else if (index_extended && middle_extended && ring_extended && pinky_extended) {
    gesture = thumb_extended ? "Open Palm" : "High Four (Open Hand)";
    confidence = 95;
  } else if (index_extended && middle_extended && !ring_extended && !pinky_extended) {
    gesture = "Peace / V Sign";
    confidence = 92;
  } else if (thumb_extended && !index_extended && !middle_extended && !ring_extended && !pinky_extended) {
    if (normalized[4][1] < normalized[2][1] - 0.2) {
      gesture = "Thumbs Up";
      confidence = 94;
    } else if (normalized[4][1] > normalized[2][1] + 0.2) {
      gesture = "Thumbs Down";
      confidence = 94;
    }
  } else if (dist(normalized[4], normalized[8]) < 0.45 && middle_extended && ring_extended && pinky_extended) {
    gesture = "OK Sign";
    confidence = 90;
  } else if (index_extended && pinky_extended && !middle_extended && !ring_extended) {
    gesture = "Rock On (Sign of Horns)";
    confidence = 92;
  } else if (index_extended && !middle_extended && !ring_extended && !pinky_extended) {
    gesture = "Pointing (Index)";
    confidence = 88;
  }
  
  updateUI({
    gesture: gesture,
    confidence: confidence,
    method: 'Local Javascript Fallback'
  });
}

// Primary prediction loops
async function predict() {
  if (!model) {
    requestAnimationFrame(predict);
    return;
  }
  
  try {
    const predictions = await model.estimateHands(video);
    
    if (predictions.length > 0) {
      const landmarks = predictions[0].landmarks; // 21 keypoints
      
      // Draw skeleton layer overlay
      drawHand(landmarks);
      
      // Query backend with throttling
      const now = Date.now();
      if (now - lastPredictionTime > PREDICTION_INTERVAL_MS) {
        lastPredictionTime = now;
        queryBackend(landmarks);
      }
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      updateUI({
        gesture: "No Hand Detected",
        confidence: 0,
        method: "Engine Standby"
      });
    }
  } catch (err) {
    console.error("Prediction loop exception:", err);
  }
  
  requestAnimationFrame(predict);
}

// Application startup entry
async function main() {
  try {
    await setupCamera();
    
    // Set canvas dimensions matching native video feed
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    await loadModel();
    predict();
  } catch (err) {
    console.error("Initialization failed:", err);
    statusText.textContent = "Error Loading Feed";
    statusDot.style.background = "#ef4444";
  }
}

main();
