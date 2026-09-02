import os
import subprocess
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimize for Render free tier CPU
torch.set_num_threads(1)

app = FastAPI(title="Wav2Lip Service")

@app.on_event("startup")
async def startup_event():
    logger.info("Wav2Lip service starting in CPU-only mode")
    logger.info(f"PyTorch using {torch.get_num_threads()} threads")

@app.get("/health")
def health_check():
    return {"status": "ok"}

import uuid
import threading

JOBS = {}

@app.post("/generate-async")
async def generate_video_async(audio: UploadFile = File(...)):
    # Save the incoming audio file
    input_audio_path = f"/tmp/{audio.filename}"
    with open(input_audio_path, "wb") as f:
        f.write(await audio.read())
        
    output_video_path = f"/tmp/output_{audio.filename}.mp4"
    
    # Check if the teacher image exists
    face_path = "teacher.jpg"
    if not os.path.exists(face_path):
        return {"error": f"Presenter image {face_path} not found."}
        
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "processing", "video_path": None, "error": None}
    
    def run_inference():
        logger.info(f"Running Wav2Lip inference for {audio.filename}...")
        try:
            # Run inference from within the Wav2Lip directory so 'temp/temp.wav' path resolves correctly
            command = [
                "python", "inference.py",
                "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
                "--face", f"../{face_path}",
                "--audio", input_audio_path,
                "--outfile", output_video_path,
                "--box", "93", "293", "414", "614"
            ]
            
            subprocess.run(command, capture_output=True, text=True, check=True, cwd="Wav2Lip")
            logger.info(f"Inference completed successfully.")
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["video_path"] = output_video_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Inference failed with exit code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = e.stderr
        finally:
            # Clean up input audio
            if os.path.exists(input_audio_path):
                os.remove(input_audio_path)

    thread = threading.Thread(target=run_inference)
    thread.start()
    
    return {"job_id": job_id, "status": "processing"}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in JOBS:
        return {"error": "Job not found"}
        
    job = JOBS[job_id]
    if job["status"] == "completed":
        return FileResponse(job["video_path"], media_type="video/mp4", filename=f"avatar_{job_id}.mp4")
    elif job["status"] == "failed":
        return {"status": "failed", "error": job["error"]}
    else:
        return {"status": "processing"}
