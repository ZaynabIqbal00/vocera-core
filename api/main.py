from contextlib import asynccontextmanager
import os
import shutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from spoof_detector import AntiSpoofDetector
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

EMBEDDINGS_DIR = r"D:\vocera-core\data\enrolled_embeddings"
TEMP_DIR = r"D:\vocera-core\data\temp"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


class VoceraEngine:

  def __init__(self):
    print("Initializing Vocera CAM++ Vector Extractor...")
    self.cam_pipeline = pipeline(
        task=Tasks.speaker_verification,
        model="damo/speech_campplus_sv_en_voxceleb_16k",
        model_revision="v1.0.2",
    )
    self.verification_threshold = 0.2673
    self.anti_spoof = AntiSpoofDetector()


engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
  global engine
  engine = VoceraEngine()
  yield


app = FastAPI(
    title="Vocera Voice Verification API",
    version="1.1.0",
    description=(
        "Enterprise Multi-Tenant Speaker Verification Engine & Anti-Spoofing"
        " Guard"
    ),
    lifespan=lifespan,
)


def _save_uploaded_file(upload_file: UploadFile, save_path: str):
  with open(save_path, "wb") as buffer:
    shutil.copyfileobj(upload_file.file, buffer)


@app.post("/enroll")
async def enroll_speaker(
    user_id: str = Form(...),
    tenant_id: str = Form("allied_bank"),
    file: UploadFile = File(...),
):
  try:
    tenant_dir = os.path.join(EMBEDDINGS_DIR, tenant_id)
    os.makedirs(tenant_dir, exist_ok=True)

    ext = os.path.splitext(file.filename)[1] or ".wav"
    enrolled_path = os.path.join(tenant_dir, f"{user_id}{ext}")

    _save_uploaded_file(file, enrolled_path)

    return {
        "status": "success",
        "message": (
            f"User '{user_id}' successfully enrolled under tenant '{tenant_id}'."
        ),
        "user_id": user_id,
        "tenant_id": tenant_id,
        "saved_path": enrolled_path,
    }
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Enrollment failed: {str(e)}"
    )


@app.post("/detect-spoof")
async def detect_spoof(file: UploadFile = File(...)):
  """Standalone endpoint to evaluate if an audio sample is genuine or AI-generated deepfake."""
  try:
    ext = os.path.splitext(file.filename)[1] or ".wav"
    temp_path = os.path.join(TEMP_DIR, f"spoof_check{ext}")
    _save_uploaded_file(file, temp_path)

    result = engine.anti_spoof.predict(temp_path)

    if os.path.exists(temp_path):
      os.remove(temp_path)

    return {"status": "success", "result": result}
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Anti-spoof analysis failed: {str(e)}"
    )


@app.post("/verify")
async def verify_speaker(
    user_id: str = Form(...),
    tenant_id: str = Form("allied_bank"),
    file: UploadFile = File(...),
):
  try:
    tenant_dir = os.path.join(EMBEDDINGS_DIR, tenant_id)

    enrolled_files = [
        f
        for f in os.listdir(tenant_dir)
        if os.path.isfile(os.path.join(tenant_dir, f))
        and f.startswith(f"{user_id}.")
    ]

    if not enrolled_files:
      raise HTTPException(
          status_code=404,
          detail=f"User '{user_id}' is not enrolled under tenant '{tenant_id}'.",
      )

    enrolled_path = os.path.join(tenant_dir, enrolled_files[0])

    ext = os.path.splitext(file.filename)[1] or ".wav"
    verify_path = os.path.join(TEMP_DIR, f"verify_{user_id}{ext}")
    _save_uploaded_file(file, verify_path)

    # Stage 1: Anti-Spoofing Liveness Check
    spoof_res = engine.anti_spoof.predict(verify_path)
    if not spoof_res["is_bonafide"]:
      if os.path.exists(verify_path):
        os.remove(verify_path)
      return {
          "status": "rejected",
          "user_id": user_id,
          "tenant_id": tenant_id,
          "reason": "SPOOF_DETECTED",
          "liveness_check": spoof_res,
          "verified": False,
          "decision": "ACCESS_DENIED",
      }

    # Stage 2: Speaker Verification via CAM++
    res = engine.cam_pipeline([enrolled_path, verify_path])
    score = float(res["score"])

    if os.path.exists(verify_path):
      os.remove(verify_path)

    verified = score >= engine.verification_threshold

    return {
        "status": "success",
        "user_id": user_id,
        "tenant_id": tenant_id,
        "liveness_check": spoof_res,
        "confidence_score": round(score, 4),
        "threshold": engine.verification_threshold,
        "verified": verified,
        "decision": "ACCESS_GRANTED" if verified else "ACCESS_DENIED",
    }
  except HTTPException as he:
    raise he
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Verification failed: {str(e)}"
    )


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)