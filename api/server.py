"""
NeuroVitals rPPG API — Production Server
=========================================
Full endpoint suite for the NeuroVitals Mental Health Phenotyping Engine.

Endpoints
---------
GET  /                   Health check
POST /analyze            Full rPPG analysis pipeline → risk output
POST /enroll             Enroll a face embedding for a subject
POST /verify             Verify identity against enrolled embedding
GET  /governance/report  Governance / drift / bias monitoring report
"""

import os
from starlette.middleware.base import BaseHTTPMiddleware

# Configurable limits via environment variables (defaults provided)
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "150"))  # Number of video frames to process
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "52428800"))  # 50 MB default upload limit

import sys

# MONKEYPATCH: Prevent mediapipe -> sounddevice -> PortAudio initialization which hangs on some Windows systems
try:
    import sounddevice
    def mock_init(*args, **kwargs): pass
    sounddevice._initialize = mock_init
    print("[NeuroVitals] [PATCH] sounddevice._initialize bypassed successfully.")
except Exception:
    pass

import math
import time
import tempfile
import traceback
import numpy as np

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sdk.face_processor import FaceProcessor, extract_frames_from_video
from sdk.feature_engineer import (
    extract_all_features, detect_peaks, compute_sqi,
)
from sdk.rppg_extractor import RPPGExtractor, cv2_mean_rgb
from sdk.anomaly_model import WaveformValidator
from sdk.bayesian_engine import (
    BayesianMentalHealthEngine, MoodInferenceEngine
)
from sdk.identity_verifier import (
    EmbeddingModel, verify_identity, cosine_similarity, IdentityVerifier,
    compute_liveness_score,
)
from sdk.logger import (
    configure_audit_logger, audit_analysis, audit_identity,
    audit_consent, audit_governance, audit_event,
)
from sdk.security import (
    generate_consent_token, verify_consent_token,
)
from sdk.governance import GovernanceMonitor
from api.schemas import (
    HealthResponse,
    AnalysisResponse,
    ExplainabilityDetail,
    BayesianResult,
    ClinicalMetricSet,
    EnrollResponse,
    VerifyResponse,
    GovernanceReport,
    MoodResult,
    GenderDetectionResponse,
    ClinicalTrendSet,
)


# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------

TRACE_LOG = os.path.join(os.path.dirname(__file__), "_logs", "analysis_trace.log")
os.makedirs(os.path.dirname(TRACE_LOG), exist_ok=True)

def trace_log(message: str):
    """Helper to write to both terminal and a dedicated debug file."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{ts}] {message}\n"
    print(message, flush=True)  # keep terminal output
    with open(TRACE_LOG, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()

app = FastAPI(
    title="NeuroVitals rPPG API",
    description=(
        "AI-powered contactless facial identification and mental health "
        "phenotyping engine using remote photoplethysmography (rPPG)."
    ),
    version="1.0.0",
)

class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_CONTENT_LENGTH:
            return JSONResponse(status_code=413, content={"detail": f"Payload too large – limit is {MAX_CONTENT_LENGTH} bytes"})
        return await call_next(request)

app.add_middleware(LimitUploadSizeMiddleware)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    err_msg = traceback.format_exc()
    trace_log(f"[GLOBAL_FATAL] Unhandled exception:\n{err_msg}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Clinical Engine Internal Error: {str(exc)}",
            "traceback": err_msg
        }
    )

logger = configure_audit_logger()

# In-memory stores (replace with database in production)
_enrolled_embeddings: dict = {}       # subject_id → np.ndarray

# Global singletons for performance — INITIALIZED AT STARTUP
_face_processor = FaceProcessor()
_rppg_extractor = RPPGExtractor(fs=30.0)
_bayesian_engine = BayesianMentalHealthEngine()
_mood_engine = MoodInferenceEngine()
_waveform_validator = WaveformValidator()
_identity_verifier = IdentityVerifier()
_governance_monitor = GovernanceMonitor()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse()


# ---------------------------------------------------------------------------
# POST /detect_gender — High-speed Automated Gender Identification
# ---------------------------------------------------------------------------

@app.post("/detect_gender", response_model=GenderDetectionResponse)
async def detect_gender(file: UploadFile = File(...)):
    """Endpoint for UI-layer automated gender identification (no age)."""
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        import cv2
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image encoding")

        result = _face_processor.estimate_demographics(frame)
        if result:
            return GenderDetectionResponse(
                success=True,
                gender=result["gender"],
                age=result["age"],
                confidence=result["confidence"],
                message="Demographics identified successfully"
            )
        else:
            return GenderDetectionResponse(success=False, message="No face detected or demographics failed")
    except Exception as e:
        trace_log(f"[NeuroVitals] [GENDER_DETECT] Error: {e}")
        return GenderDetectionResponse(success=False, message=str(e))




# ---------------------------------------------------------------------------
# POST /analyze  — Full rPPG Analysis Pipeline
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_video(
    file: UploadFile = File(...),
    age: int = Form(35),
    gender: str = Form("other")
):
    """Start a new analysis scan.

    - Truncate the runtime trace log to keep it fresh.
    - Delete any stray temporary video files left over from previous crashes.
    """
    # ---- Cleanup old runtime artifacts ----
    # Reset the trace log (if it exists)
    try:
        with open(TRACE_LOG, "w", encoding="utf-8") as f:
            f.truncate(0)
    except Exception as e:
        trace_log(f"[NeuroVitals] [CLEANUP] Failed to truncate trace log: {e}")
    # Remove any other .log files in the logs folder to start fresh
    logs_dir = os.path.dirname(TRACE_LOG)
    for lf in os.listdir(logs_dir):
        log_path = os.path.join(logs_dir, lf)
        if lf != os.path.basename(TRACE_LOG) and lf.lower().endswith('.log'):
            try:
                os.remove(log_path)
                trace_log(f"[NeuroVitals] [CLEANUP] Deleted old log file {lf}")
            except Exception as e:
                trace_log(f"[NeuroVitals] [CLEANUP] Failed to delete log file {lf}: {e}")
    # Remove leftover .webm temp files from the system temp directory
    temp_dir = tempfile.gettempdir()
    for fname in os.listdir(temp_dir):
        if fname.lower().endswith('.webm'):
            try:
                os.remove(os.path.join(temp_dir, fname))
            except Exception as e:
                trace_log(f"[NeuroVitals] [CLEANUP] Failed to delete temp file {fname}: {e}")
    # Continue with the normal request handling below

    """Accept a short video, run the full rPPG → Bayesian → SHAP pipeline.
    
    Using 'def' instead of 'async def' to offload this CPU-bound work
    to FastAPI's internal threadpool, preventing event loop blockage.
    """

    video_path = None # Initialize video_path outside the try block

    try:
        # Use simple file.file.read() to avoid async file operations in a sync worker
        contents = file.file.read()

        # Write to temp file for cv2
        trace_log(f"[NeuroVitals] Received request. Payload size: {len(contents)} bytes")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        try:
            tmp.write(contents)
            tmp.close()
            video_path = tmp.name
            trace_log(f"[NeuroVitals] Temp video stored: {video_path}")
        except Exception as e:
            trace_log(f"[NeuroVitals] ERROR writing temp file: {e}")
            raise HTTPException(status_code=500, detail="Failed to store uploaded video")

        # --- 1 & 2. Streaming Frame and ROI extraction ---
        trace_log("[NeuroVitals] [STEP 1 & 2] Starting streaming frame extraction to prevent OOM...")
        
        import cv2
        cap = cv2.VideoCapture(video_path)
        rgb_means_buffer = []
        best_roi = None
        landmarks_buffer = []
        demo_samples = []
        max_frames = MAX_FRAMES  # Use configurable limit
        frame_idx = 0
        
        # Determine total frames if possible to spread out demographic sampling
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = max_frames
        sample_interval = max(1, min(total_frames, max_frames) // 5)

        try:
            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Downscale frame slightly if it's huge (e.g. 1080p -> 720p) to save memory during inference
                # Not strictly necessary since we free it right after, but helps Mediapipe/InsightFace peak memory
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    frame = cv2.resize(frame, (1280, int(h * scale)))

                if frame_idx % 20 == 0:
                    trace_log(f"[NeuroVitals] [STEP 2] ROI Frame {frame_idx}... Buffer size: {len(rgb_means_buffer)}")

                # --- 1.5 Automated Demographic Estimation (Consensus Voting) ---
                if gender.lower() == "auto" and frame_idx % sample_interval == 0 and len(demo_samples) < 5:
                    res = _face_processor.estimate_demographics(frame)
                    if res:
                        demo_samples.append(res)

                # --- 2. Face ROI extraction & Landmark Collection ---
                res_roi, res_lm = None, None
                try:
                    res_roi, res_lm = _face_processor.extract_roi_with_landmarks(frame)
                except Exception as sdk_e:
                    trace_log(f"[NeuroVitals] [STEP 2] SDK Error at frame {frame_idx}: {sdk_e}")
                
                if res_roi is not None:
                    # Immediately convert to RGB mean floats so we don't hold the huge image in RAM!
                    rgb_means_buffer.append(cv2_mean_rgb(res_roi))
                    landmarks_buffer.append(res_lm)
                    # Keep just ONE good quality face crop (around the middle of the video) for Identity Verification
                    if best_roi is None or len(rgb_means_buffer) == max_frames // 2:
                        best_roi = res_roi.copy()
                
                # Manual memory check/hint
                if frame_idx % 50 == 0:
                    import gc
                    gc.collect()
                    
                frame_idx += 1
                
        except Exception as loop_e:
            trace_log(f"[NeuroVitals] [STEP 2] FATAL LOOP ERROR: {loop_e}")
            trace_log(traceback.format_exc())
            cap.release()
            raise HTTPException(status_code=500, detail="ROI pipeline failure")
            
        cap.release()
        import gc
        gc.collect()

        trace_log(f"[NeuroVitals] [STEP 1 & 2] Processing complete. Valid faces: {len(rgb_means_buffer)}")

        # Process demographics consensus if requested
        if gender.lower() == "auto":
            if demo_samples:
                # Majority vote for gender
                genders = [s["gender"] for s in demo_samples]
                gender = max(set(genders), key=genders.count)
                
                # Median for age to reject outliers
                ages = [s["age"] for s in demo_samples]
                age = int(np.median(ages))
                
                trace_log(f"[NeuroVitals] [STEP 1.5] Consensus Results -> Gender: {gender}, Age: {age} (from {len(demo_samples)} samples)")
            else:
                trace_log("[NeuroVitals] [STEP 1.5] Demographic estimation failed, using defaults.")
                gender = "other"

        if len(rgb_means_buffer) < 2:  # Reduced from 10 to 2 for extreme resiliency
            trace_log(f"[NeuroVitals] [STEP 2] WARNING: Only {len(rgb_means_buffer)} faces found. Proceeding with limited data.")
            if len(rgb_means_buffer) == 0:
                raise HTTPException(status_code=400, detail="No face detected in video stream")

        # --- 3. rPPG signal extraction (CHROM) ---
        trace_log("[NeuroVitals] [STEP 3] Running CHROM signal extraction...")
        extraction_res = _rppg_extractor.extract([], rgb_means=np.array(rgb_means_buffer))
        
        if isinstance(extraction_res, dict):
            signal = extraction_res.get("pulse_signal", np.array([]))
            r_ratio = extraction_res.get("r_ratio")
            b_ratio = extraction_res.get("b_ratio")
        else:
            # Fallback for old RPPGExtractor versions
            signal = extraction_res
            r_ratio, b_ratio = None, None

        trace_log(f"[NeuroVitals] [STEP 3] Signal extracted. Samples: {len(signal)}")

        # --- 4. Full feature extraction ---
        trace_log("[NeuroVitals] [STEP 4] Computing clinical features (Morphology + Multi-channel)...")
        features = extract_all_features(
            signal, 
            fs=30.0, 
            r_ratio=r_ratio, 
            b_ratio=b_ratio,
            age=age
        )
        
        # Clean up features: convert NaN to 0.0 for easier consumption if preferred, 
        # but here we log them to see what's actually happening.
        trace_log(f"[NeuroVitals] [STEP 4] RAW Features: {features}")
        
        # Explicitly check for HR
        hr_val = features.get('heart_rate_bpm')
        if hr_val is None or math.isnan(hr_val):
            trace_log("[NeuroVitals] [STEP 4] WARNING: Heart rate is NaN (insufficient peaks)")
        else:
            trace_log(f"[NeuroVitals] [STEP 4] Heart Rate Detected: {hr_val:.2f} BPM")

        # --- 5. Bayesian mental health inference ---
        trace_log(f"[NeuroVitals] [STEP 5] Running Bayesian Inference (Age: {age}, Gender: {gender})...")
        inference = _bayesian_engine.full_inference(features, age=age, gender=gender)
        
        posteriors = inference["posteriors"]
        risk_class = inference["risk_class"]
        trace_log(f"[NeuroVitals] [STEP 5] Inference Result: {risk_class}")

        # --- 6. Identity & liveness ---
        trace_log("[NeuroVitals] [STEP 6] Running Identity Verification...")
        # Liveness now uses the real landmarks sequence
        liveness_score = _identity_verifier.track_liveness(landmarks_buffer)
        trace_log(f"[NeuroVitals] [STEP 6] Blink-based Liveness: {liveness_score:.2f}")

        # Verification (Face Embedding)
        # We pass the best ROI we kept during the stream to the embedder
        embedding_model = EmbeddingModel()
        embedding = embedding_model.embed(best_roi)
        # Placeholder verification against self
        identity_ok = verify_identity(embedding, embedding, threshold=0.85)

        # --- 7. Build explainability detail ---
        rmssd = features.get("rmssd", 0.0) or 0.0
        lf_hf = features.get("lf_hf_ratio", 0.0) or 0.0
        entropy = features.get("spectral_entropy", 0.0) or 0.0
        rsa = features.get("rsa", 0.0) or 0.0
        stress = features.get("stress_reactivity", 0.0) or 0.0

        explainability = ExplainabilityDetail(
            Low_HRV=float(max(0, 1.0 - rmssd / 0.1)),
            High_LFHF=float(min(lf_hf / 5.0, 1.0)),
            High_Entropy=float(min(entropy / 5.0, 1.0)),
            Low_RSA=float(max(0, 1.0 - rsa / 0.005)),
            High_Stress=float(stress),
        )

        # --- 8. Signal Authenticity (LSTM Anomaly) ---
        authenticity = _waveform_validator.validate_signal(signal)
        trace_log(f"[NeuroVitals] [STEP 8] Waveform Authenticity Score: {authenticity:.2f}")

        # --- 9. Clinical Result Mapping & Explainability ---
        inference = _bayesian_engine.full_inference(features, age=age, gender=gender)
        posteriors = inference["posteriors"]
        risk_class = inference["risk_class"]
        explainability_data = inference["explainability"]

        # --- 9.5 Mood Inference ---
        mood_data = _mood_engine.infer_mood(features, age=age, gender=gender)
        trace_log(f"[NeuroVitals] [MOOD] Current State: {mood_data['MoodState']}")

        # --- 10. Confidence based on SQI ---
        sqi = features.get("signal_quality_index", 0.5)
        
        # --- 11. Assemble response ---
        # Ensure all NaN features are converted to 0.0 for JSON compatibility
        processed_features = {
            k: (v if not (isinstance(v, (float, np.float64)) and np.isnan(v)) else 0.0)
            for k, v in features.items()
            if not isinstance(v, dict)
        }
        # Explicitly add hrv_raw back if it exists
        if "hrv_raw" in features:
            processed_features["hrv_raw"] = features["hrv_raw"]

        # Final Liveness is a blend of facial consistency + waveform authenticity
        final_liveness = float(np.clip(0.7 * liveness_score + 0.3 * authenticity, 0.0, 1.0))

        response = AnalysisResponse(
            IdentityVerified=bool(identity_ok),
            LivenessScore=final_liveness,
            SignalQualityIndex=float(sqi),
            DepressionRiskScore=float(posteriors.get("depression", 0.0)),
            AnxietyRiskScore=float(posteriors.get("anxiety", 0.0)),
            AutonomicStabilityIndex=float(1.0 - processed_features.get("stress_reactivity", 0.0)),
            MentalHealthRiskClass=risk_class,
            Explainability=ExplainabilityDetail(**explainability_data),
            Mood=MoodResult(**mood_data),
            BayesianPosteriors=BayesianResult(**posteriors),
            ClinicalFeatures=ClinicalMetricSet(**processed_features),
            ClinicalTrends=ClinicalTrendSet(**features.get("trends", {})),
            ConfidenceScore=float(np.clip(sqi * 0.95, 0.0, 1.0)),
            DominantCondition=max(posteriors, key=posteriors.get) if posteriors else "Unknown",
            DetectedGender=gender,
            DetectedAge=age,
        )

        audit_analysis(logger, {
            "risk_class": response.MentalHealthRiskClass,
            "confidence": response.ConfidenceScore,
            "sqi": response.SignalQualityIndex,
        })

                # Explicit cleanup of large in‑memory objects to free RAM before returning
        del signal
        del rgb_means_buffer
        del landmarks_buffer
        del features
        import gc
        gc.collect()
        return response

    except HTTPException:
        # Re-raise HTTPExceptions as they are intended responses
        raise
    except Exception as e:
        error_msg = f"[NeuroVitals] [FATAL] Critical error during analysis: {str(e)}\n{traceback.format_exc()}"
        trace_log(error_msg)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        # Cleanup
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError as e:
                trace_log(f"[NeuroVitals] WARNING: Failed to delete temp file {video_path}: {e}")


# ---------------------------------------------------------------------------
# POST /enroll  — Enroll Face Embedding
# ---------------------------------------------------------------------------

@app.post("/enroll", response_model=EnrollResponse)
async def enroll_face(
    subject_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Enroll a face image for identity verification."""
    import cv2

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    model = EmbeddingModel()
    embedding = model.embed(image)
    _enrolled_embeddings[subject_id] = embedding

    audit_identity(logger, {
        "action": "enroll",
        "subject_id": subject_id,
    })

    return EnrollResponse(subject_id=subject_id)


# ---------------------------------------------------------------------------
# POST /verify  — Verify Identity
# ---------------------------------------------------------------------------

@app.post("/verify", response_model=VerifyResponse)
async def verify_face(
    subject_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Verify a face image against an enrolled embedding."""
    import cv2

    if subject_id not in _enrolled_embeddings:
        raise HTTPException(status_code=404,
                            detail=f"Subject '{subject_id}' not enrolled")

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    model = EmbeddingModel()
    probe = model.embed(image)
    enrolled = _enrolled_embeddings[subject_id]

    sim = cosine_similarity(probe, enrolled)
    verified = verify_identity(probe, enrolled, threshold=0.85)

    outcome = "success" if verified else "failure"
    audit_identity(logger, {
        "action": "verify",
        "subject_id": subject_id,
        "similarity": round(sim, 4),
        "verified": verified,
    }, outcome=outcome)

    return VerifyResponse(
        subject_id=subject_id,
        identity_verified=verified,
        similarity_score=round(sim, 4),
        liveness_score=0.9,  # placeholder
        message="Identity verified" if verified else "Identity mismatch",
    )


# ---------------------------------------------------------------------------
# GET /governance/report
# ---------------------------------------------------------------------------

@app.get("/governance/report", response_model=GovernanceReport)
async def governance_report():
    """Return the latest governance / drift / bias monitoring report.

    In production this would use stored reference distributions and
    recent prediction logs.  Here we return a placeholder report.
    """
    report = _governance_monitor.generate_report()

    audit_governance(logger, {"action": "report_generated"})

    recommendation = "No action required"
    if report.get("any_drift_detected"):
        recommendation = "Feature drift detected — model recalibration recommended"

    return GovernanceReport(
        status=report.get("status", "generated"),
        any_drift_detected=report.get("any_drift_detected", False),
        recommendation=recommendation,
    )
