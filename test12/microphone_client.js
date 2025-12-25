/**
 * STT Microphone Client (v3.2 - Base64 & WebSocket Optimized)
 * Features: Robust connection handling, detailed error logging, and clean UI state management.
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
let inputSource = null;
let processor = null;
let isRecording = false;

// ==========================================
// 3. UI Helper Functions
// ==========================================

// Updates the status banner with color coding
function updateStatus(message, type = 'info') {
    statusDiv.textContent = message;
    statusDiv.className = 'status'; // Reset class
    statusDiv.classList.add(`status-${type}`); // Add specific type
    console.log(`[${type.toUpperCase()}] ${message}`);
}

// Formats error objects into readable strings
function formatError(err) {
    if (typeof err === 'string') return err;
    if (err.message) return err.message;
    return "Unknown error occurred";
}

// Creates the visual card for transcription results
function createResultCard(data) {
    const card = document.createElement('div');
    card.className = 'result-box';

    let html = `<div class="section-title">Transcription Result</div>`;
    
    // A. Transcription Text
    if (data.transcription && data.transcription.trim().length > 0) {
        html += `<p><strong>Text:</strong> ${data.transcription}</p>`;
    } else {
        html += `<p><em>No speech detected.</em></p>`;
    }

    // B. Base64 Fingerprint
    // ... inside createResultCard(data) ...

    // 2. Base64 Fingerprint & Audio Player
    if (data.voice_fingerprint_sample_b64) {
        const shortHash = data.voice_fingerprint_sample_b64.substring(0, 50) + "...";
        
        // Convert Base64 to a playable Audio Blob
        const byteCharacters = atob(data.voice_fingerprint_sample_b64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        
        // Create WAV header (simplified) or raw PCM blob
        // Since it's raw Float32, we need a WAV container to play it easily in browser, 
        // OR we can just tell the user it's raw data.
        // For simplicity, let's just show the data, as raw PCM playback in HTML requires more code.
        
        html += `
            <div style="margin-top: 15px; border-top: 1px solid #ddd; padding-top: 10px;">
                <p><strong>Captured Audio Fingerprint:</strong></p>
                <div style="background:#f9f9f9; padding:10px; border-radius:4px; margin-bottom:5px;">
                    <span style="font-size:12px; color:#555;">(Raw Audio Data Captured)</span>
                </div>
                
                <p style="margin-top:5px; font-size:12px;"><strong>Base64 Preview:</strong></p>
                <code style="display:block; background:#fff; padding:8px; border:1px solid #ddd; border-radius:4px; word-break:break-all; font-size:11px; color:#555;">
                    ${shortHash}
                </code>
            </div>
        `;
    }

    // C. Statistics
    html += `
        <div class="info-text" style="margin-top: 10px;">
            Duration: ${data.total_duration_seconds}s | Chunks: ${data.chunks_processed}
        </div>
    `;

    card.innerHTML = html;
    
    // Insert new result at the top of the list
    outputDiv.insertBefore(card, outputDiv.firstChild);
}

// ==========================================
// 4. Audio Processing Logic
// ==========================================

function setupAudioProcessing() {
    try {
        const desiredSampleRate = parseInt(sampleRateSelect.value);
        
        // Cross-browser AudioContext creation
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        audioContext = new AudioContextClass({
            sampleRate: desiredSampleRate
        });

        // Create Source from the microphone stream
        inputSource = audioContext.createMediaStreamSource(mediaStream);

        // Create ScriptProcessor (Buffer Size: 4096 = approx 0.1s latency)
        // Note: ScriptProcessor is deprecated but allows raw binary access without extra worklet files
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (e) => {
            // Only send data if we are recording and the socket is open
            if (!isRecording || !websocket || websocket.readyState !== WebSocket.OPEN) return;

            // Get Raw Float32 Data (Mono channel)
            const inputData = e.inputBuffer.getChannelData(0);
            
            // Send binary data directly to the Python server
            websocket.send(inputData.buffer);
        };

        // Connect the audio nodes
        inputSource.connect(processor);
        processor.connect(audioContext.destination); // Required destination for the processor to run
        
    } catch (err) {
        updateStatus(`Audio Setup Error: ${formatError(err)}`, 'error');
        stopRecordingInternal();
    }
}

// Internal function to stop audio flow without resetting the whole UI
function stopRecordingInternal() {
    if (!isRecording) return;
    isRecording = false;

    // 1. Send "End" Signal to Server
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ type: "end" }));
    }

    // 2. Disconnect Audio Nodes
    if (processor) {
        processor.disconnect();
        inputSource.disconnect();
    }
    
    // 3. Close Audio Context to release hardware
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
        audioContext = null;
    }
}

// Helper to reset button states
function resetRecordingUI() {
    startBtn.disabled = false;
    startBtn.classList.replace('btn-secondary', 'btn-primary');
    
    stopBtn.disabled = true;
    stopBtn.classList.replace('btn-danger', 'btn-secondary');
}

// ==========================================
// 5. Event Listeners (Button Clicks)
// ==========================================

// --- Initialize Microphone ---
initBtn.addEventListener('click', async () => {
    try {
        updateStatus('Requesting microphone access...', 'info');
        
        // Check for browser support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Your browser does not support audio access. Please use Chrome, Edge, or Firefox.");
        }

        // Request microphone access
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        updateStatus('Microphone initialized. Ready to record.', 'success');
        
        // Enable Start Button
        initBtn.disabled = true;
        initBtn.classList.replace('btn-primary', 'btn-secondary');
        startBtn.disabled = false;
        cleanupBtn.disabled = false;
        
    } catch (err) {
        updateStatus(`Mic Error: ${formatError(err)}`, 'error');
    }
});

// --- Start Recording ---
startBtn.addEventListener('click', () => {
    if (!mediaStream) {
        updateStatus('Error: Microphone stream lost. Please re-initialize.', 'error');
        return;
    }

    // UI Updates
    isRecording = true;
    startBtn.disabled = true;
    startBtn.classList.replace('btn-primary', 'btn-secondary');
    stopBtn.disabled = false;
    stopBtn.classList.replace('btn-secondary', 'btn-danger'); // Red stop button
    updateStatus('Connecting to server...', 'warning');

    // Create WebSocket Connection
    try {
        // ENSURE THIS MATCHES YOUR PYTHON SERVER ADDRESS
        websocket = new WebSocket("ws://127.0.0.1:8000/ws/transcribe-stream");
    } catch (e) {
        updateStatus(`WebSocket Creation Error: ${formatError(e)}`, 'error');
        return;
    }

    // WebSocket: Connected
    websocket.onopen = () => {
        updateStatus('Connected! Streaming audio...', 'warning');
        
        // Send Configuration JSON
        const config = {
            sample_rate: parseInt(sampleRateSelect.value),
            language: languageSelect.value,
            word_level: wordLevelCheckbox.checked
        };
        websocket.send(JSON.stringify(config));

        // Begin Audio Stream
        setupAudioProcessing();
    };

    // WebSocket: Message Received
    websocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.status === 'chunk_processed') {
                // Live status update
                let fpStatus = data.fingerprint_extracted ? " | 🆔 Fingerprint Captured" : "";
                updateStatus(`Recording... (${data.total_duration_seconds}s)${fpStatus}`, 'warning');
            } 
            else if (data.status === 'complete') {
                // Final result received
                createResultCard(data);
                updateStatus('Transcription complete.', 'success');
                resetRecordingUI();
                
                // Close socket nicely
                websocket.close();
            } 
            else if (data.status === 'error') {
                updateStatus(`Server Error: ${data.error}`, 'error');
                stopRecordingInternal();
                resetRecordingUI();
            }
        } catch (e) {
            console.error("Failed to parse WebSocket message:", event.data);
        }
    };

    // WebSocket: Connection Error
    websocket.onerror = (e) => {
        console.error("WebSocket Error Object:", e);
        updateStatus('Connection Failed. Is "main.py" running?', 'error');
        stopRecordingInternal();
        resetRecordingUI();
    };

    // WebSocket: Closed
    websocket.onclose = (e) => {
        if (isRecording) {
            stopRecordingInternal();
            resetRecordingUI();
            updateStatus(`Connection closed (Code: ${e.code})`, 'info');
        }
    };
});

// --- Stop Recording ---
stopBtn.addEventListener('click', () => {
    updateStatus('Finalizing results...', 'info');
    stopRecordingInternal();
});

// --- Cleanup / Reset ---
cleanupBtn.addEventListener('click', () => {
    // Close WebSocket
    if (websocket) websocket.close();
    
    // Close Audio Context
    if (audioContext && audioContext.state !== 'closed') audioContext.close();
    
    // Stop all microphone tracks (turns off the red light on your mic/tab)
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    
    // Reset variables
    mediaStream = null;
    audioContext = null;
    websocket = null;
    isRecording = false;

    // Clear output area
    outputDiv.innerHTML = '';

    // Reset Buttons to initial state
    initBtn.disabled = false;
    initBtn.classList.replace('btn-secondary', 'btn-primary');
    
    startBtn.disabled = true;
    startBtn.classList.replace('btn-secondary', 'btn-primary');
    
    stopBtn.disabled = true;
    stopBtn.classList.replace('btn-danger', 'btn-secondary');

    updateStatus('Cleaned up. Click Initialize to start again.', 'info');
});