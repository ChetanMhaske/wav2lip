import os
import subprocess
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimize for Render free tier CPU
torch.set_num_threads(4)

app = FastAPI(title="Wav2Lip Service")

@app.on_event("startup")
async def startup_event():
    logger.info("Wav2Lip service starting in CPU-only mode")
    logger.info(f"PyTorch using {torch.get_num_threads()} threads")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/generate")
async def generate_video(audio: UploadFile = File(...)):
    # Save the incoming audio file
    input_audio_path = f"/tmp/{audio.filename}"
    with open(input_audio_path, "wb") as f:
        f.write(await audio.read())
        
    output_video_path = f"/tmp/output_{audio.filename}.mp4"
    
    # Check if the teacher image exists
    face_path = "teacher.jpg"
    if not os.path.exists(face_path):
        return {"error": f"Presenter image {face_path} not found."}
        
    logger.info(f"Running Wav2Lip inference for {audio.filename}...")
    
    # Run the Wav2Lip inference script
    try:
        command = [
            "python", "Wav2Lip/inference.py",
            "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth",
            "--face", face_path,
            "--audio", input_audio_path,
            "--outfile", output_video_path
        ]
        
        # Use check=True to raise exception on failure
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info(f"Inference completed successfully.")
        
    except subprocess.CalledProcessError as e:
        from fastapi import HTTPException
        logger.error(f"Inference failed with exit code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise HTTPException(status_code=500, detail={"error": "Video generation failed", "stderr": e.stderr})
    finally:
        # Clean up input audio
        if os.path.exists(input_audio_path):
            os.remove(input_audio_path)
            
    return FileResponse(output_video_path, media_type="video/mp4", filename=f"avatar_{audio.filename}.mp4")
