/**
 * STT Microphone Client (v4.1.0)
 * Features:
 * - Real-time WebSocket Audio Streaming (Float32)
 * - Biometric 256-dim Vector Visualization (Canvas)
 * - Unique Call ID Handshake & Display
 * - Robust Error Handling & Cleanup
 */

// ==========================================
// 1. DOM Elements
// ==========================================
const statusDiv = document.getElementById('status');
const outputDiv = document.getElementById('output');
const languageSelect = document.getElementById('language');
const sampleRateSelect = document.getElementById('sample-rate');
const wordLevelCheckbox = document.getElementById('word-level');

const initBtn = document.getElementById('init-btn');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const cleanupBtn = document.getElementById('cleanup-btn');

// ==========================================
// 2. Global State Variables
// ==========================================
let websocket = null;
let audioContext = null;
let mediaStream = null;
let processor = null;
let isRecording = false;
let currentCallId = null;

// ==========================================
// 3. UI Helper Functions
// ==========================================

/**
 * Updates the status banner with appropriate styling
 * Uses classes: status-info, status-success, status-warning, status-error
 */
function updateStatus(message, type = 'info') {
    statusDiv.textContent = message;
    // Reset to base class then add specific type
    statusDiv.className = 'status'; 
    statusDiv.classList.add(`status-${type}`);
    console.log(`[${type.toUpperCase()}] ${message}`);
}

/**
 * Visualizes the 256-dimensional speaker embedding as a bar chart.
 * @param {HTMLCanvasElement} canvas - The canvas to draw on.
 * @param {string} b64String - The Base64 encoded float32 array.
 */
function drawEmbedding(canvas, b64String) {
    const ctx = canvas.getContext('2d');
    
    // Set resolution to match display size for sharpness
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Decode Base64 -> Float32Array
    const binaryString = atob(b64String);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    const floatArray = new Float32Array(bytes.buffer);

    // Clear Canvas
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#667eea'; // Match your primary CSS gradient color
    
    const barWidth = width / floatArray.length;
    
    // Draw 256 dimensions
    for (let i = 0; i < floatArray.length; i++) {
        const val = floatArray[i]; 
        // Scale height for visibility (abs value * height * factor)
        const barHeight = Math.abs(val) * (height / 2) * 5; 
        
        // Center bars vertically
        const y = height / 2 - (val > 0 ? barHeight : 0);
        
        ctx.fillRect(i * barWidth, y, Math.max(1, barWidth), barHeight);
    }
}

/**
 * Creates the visual card for transcription results & biometrics
 */
function createResultCard(data) {
    const card = document.createElement('div');
    card.className = 'result-box';

    // 1. Header (Call ID & Time)
    const time = new Date().toLocaleTimeString();
    // Extract short ID for display (e.g., "ws-a1b2...")
    const displayId = data.call_id ? data.call_id.substring(0, 18) + '...' : 'N/A';
    
    let html = `
        <div style="display:flex; justify-content:space-between; color:#666; font-size:0.85em; margin-bottom:10px; border-bottom:1px solid #eee; padding-bottom:5px;">
            <span title="${data.call_id}"><strong>ID:</strong> ${displayId}</span>
            <span>${time}</span>
        </div>
    `;
    
    // 2. Transcription Text
    html += `<div style="font-size:1.1em; color:#333; margin-bottom:15px; line-height:1.6;">
        ${data.transcription || "<em>No speech detected.</em>"}
    </div>`;

    // 3. Biometric Visualization (Canvas)
    if (data.voice_embedding_exists && data.voice_embedding_b64) {
        html += `
            <div style="background:#f8fafc; padding:15px; border-radius:8px; border:1px solid #e2e8f0; margin-top:10px;">
                <div style="display:inline-flex; align-items:center; gap:5px; font-size:0.8em; background:#e0e7ff; color:#3730a3; padding:4px 8px; border-radius:4px; font-weight:600; margin-bottom:8px;">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"></path>
                        <path d="M12 6a6 6 0 1 0 6 6 6 6 0 0 0-6-6zm0 10a4 4 0 1 1 4-4 4 4 0 0 1-4 4z"></path>
                    </svg>
                    Speaker Identity Vector (256-dim)
                </div>
                <canvas class="embedding-viz" style="width:100%; height:60px; background:#1e293b; border-radius:4px; display:block;"></canvas>
            </div>
        `;
    }

    // 4. Statistics Footer
    html += `
        <div class="info-text">
            Saved as: ${data.saved_file || 'Not saved'}
        </div>
    `;

    card.innerHTML = html;
    
    // Insert new result at the top of the output list (Stack LIFO)
    outputDiv.insertBefore(card, outputDiv.firstChild);

    // 5. Render Canvas (must be done after element is inserted into DOM)
    if (data.voice_embedding_exists) {
        const canvas = card.querySelector('canvas');
        if(canvas) drawEmbedding(canvas, data.voice_embedding_b64);
    }
}

// ==========================================
// 4. Audio Processing Logic
// ==========================================

function setupAudioProcessing() {
    try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        // Force 16000Hz to match backend requirement
        audioContext = new AudioContextClass({ sampleRate: 16000 }); 
        
        const inputSource = audioContext.createMediaStreamSource(mediaStream);

        // ScriptProcessor is deprecated but reliable for raw Float32 access. 
        // Buffer Size 4096 = ~250ms latency
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (e) => {
            if (!isRecording || websocket?.readyState !== WebSocket.OPEN) return;

            // Get Raw Float32 Data (Mono channel)
            const inputData = e.inputBuffer.getChannelData(0);
            
            // Send binary data directly to the Python server
            websocket.send(inputData.buffer);
        };

        // Connect the audio nodes: Source -> Processor -> Destination
        inputSource.connect(processor);
        processor.connect(audioContext.destination); // Required for processor to run
        
    } catch (err) {
        updateStatus(`Audio Setup Error: ${err.message}`, 'error');
        stopRecordingInternal();
    }
}

function stopRecordingInternal() {
    isRecording = false;

    // 1. Send "End" Signal to Server
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ type: "end" }));
    }

    // 2. Disconnect Audio Nodes
    if (processor) {
        processor.disconnect();
        processor = null;
    }
    
    // 3. Close Audio Context
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
        audioContext = null;
    }
}

// ==========================================
// 5. Event Listeners
// ==========================================

// --- Initialize Microphone ---
initBtn.addEventListener('click', async () => {
    try {
        updateStatus('Requesting microphone access...', 'info');
        
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Browser API not supported.");
        }

        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        updateStatus('Microphone Ready. Click Start Recording.', 'success');
        
        // Update Buttons
        initBtn.disabled = true;
        initBtn.classList.replace('btn-primary', 'btn-secondary'); // Grey out
        initBtn.textContent = "Mic Initialized";
        
        startBtn.disabled = false;
        startBtn.classList.replace('btn-secondary', 'btn-primary'); // Enable Start
        
    } catch (err) {
        updateStatus(`Mic Access Denied: ${err.message}`, 'error');
    }
});

// --- Start Recording ---
startBtn.addEventListener('click', () => {
    if (!mediaStream) {
        updateStatus('Error: Microphone stream lost. Reload page.', 'error');
        return;
    }

    // 1. Connect WebSocket
    // Ensure this matches your Python Server address
    websocket = new WebSocket("ws://127.0.0.1:8000/ws/transcribe-stream");

    websocket.onopen = () => {
        isRecording = true;
        updateStatus('Handshaking with server...', 'warning');

        // 2. Send Configuration
        const config = {
            sample_rate: 16000,
            language: languageSelect.value,
            word_level: wordLevelCheckbox.checked
        };
        websocket.send(JSON.stringify(config));

        // 3. Start Audio Stream
        setupAudioProcessing();

        // UI Updates
        startBtn.disabled = true;
        startBtn.classList.replace('btn-primary', 'btn-secondary');
        startBtn.textContent = "Recording...";
        
        stopBtn.disabled = false;
        stopBtn.classList.replace('btn-secondary', 'btn-danger'); // Red Stop Button
    };

    websocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.status === 'ready') {
                currentCallId = data.call_id;
                // Show Call ID in status
                updateStatus(`Recording Active [ID: ${currentCallId.split('-')[1]}...]`, 'warning');
            
            } else if (data.status === 'chunk_processed') {
                // Optional: Update timer/counter here
            
            } else if (data.status === 'complete') {
                createResultCard(data);
                updateStatus('Processing Complete.', 'success');
                websocket.close(); // Clean close
                
                // Reset UI for next recording
                startBtn.disabled = false;
                startBtn.classList.replace('btn-secondary', 'btn-primary');
                startBtn.textContent = "Start Recording";
                stopBtn.disabled = true;
                stopBtn.classList.replace('btn-danger', 'btn-secondary');
            
            } else if (data.status === 'error') {
                updateStatus(`Server Error: ${data.error}`, 'error');
                stopRecordingInternal();
            }
        } catch (e) {
            console.error("Parse error:", event.data);
        }
    };

    websocket.onerror = (e) => {
        updateStatus('Connection Failed. Is the server running?', 'error');
        stopRecordingInternal();
    };
    
    websocket.onclose = () => {
        if(isRecording) stopRecordingInternal();
    };
});

// --- Stop Recording ---
stopBtn.addEventListener('click', () => {
    updateStatus('Finalizing results...', 'info');
    stopRecordingInternal();
    
    // UI Updates immediately
    stopBtn.disabled = true;
    stopBtn.classList.replace('btn-danger', 'btn-secondary');
});

// --- Cleanup ---
cleanupBtn.addEventListener('click', () => {
    // Force stop everything
    stopRecordingInternal();
    
    // Stop tracks
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    mediaStream = null;
    
    // Clear UI
    outputDiv.innerHTML = '';
    
    // Reload page to get clean state
    window.location.reload();
});