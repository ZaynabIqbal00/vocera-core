import io
import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the unified engine from spoof_detector.py
from spoof_detector import VoceraUnifiedEngine

app = FastAPI(title="Vocera Voice Biometrics & Anti-Spoofing API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine globally on startup
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    # Point to your downloaded checkpoint path
    engine = VoceraUnifiedEngine(antispoof_ckpt_path="vocera_robust_musan_eer.pt")
    print("🚀 Vocera Engine loaded and ready for API requests.")

@app.get("/")
def health_check():
    return {"status": "online", "service": "Vocera Biometrics Core", "eer_target": "< 5%"}

# Rest API Endpoint for File-based Audio Verification
@app.post("/api/v1/verify-audio")
async def verify_audio_file(file: UploadFile = File(...)):
    contents = await file.read()
    audio_stream = io.BytesIO(contents)
    
    wf, sr = torchaudio.load(audio_stream)
    wf_padded = engine.preprocess_audio(wf, sr)
    
    prob, risk_score, is_real = engine.predict_spoof_risk(wf_padded)
    
    return {
        "filename": file.filename,
        "is_real_human": is_real,
        "bonafide_probability": round(prob, 4),
        "deepfake_risk_score": f"{risk_score:.2f}%",
        "action_required": "PROCEED" if is_real else "TRIGGER_LOW_CONFIDENCE_MFA"
    }

# WebSocket Endpoint for Live Audio Stream Inference
@app.websocket("/ws/stream-verify")
async def stream_verify(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket Client Connected")
    try:
        while True:
            # Receive raw PCM audio chunk bytes from client
            data = await websocket.receive_bytes()
            
            # Process bytes into waveform tensor
            audio_stream = io.BytesIO(data)
            wf, sr = torchaudio.load(audio_stream)
            wf_padded = engine.preprocess_audio(wf, sr)
            
            # Predict risk score
            prob, risk_score, is_real = engine.predict_spoof_risk(wf_padded)
            
            # Stream JSON assessment back to client
            await websocket.send_json({
                "is_real_human": is_real,
                "confidence": round(prob, 4),
                "deepfake_risk_score": round(risk_score, 2),
                "status": "PASS" if is_real else "ALERT_SPOOF_DETECTED"
            })
    except WebSocketDisconnect:
        print("❌ WebSocket Client Disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)