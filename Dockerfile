FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including ffmpeg and wget
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Wav2Lip checkpoint
RUN mkdir -p /app/Wav2Lip/checkpoints
RUN wget --progress=dot:giga -O /app/Wav2Lip/checkpoints/wav2lip_gan.pth https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth

# Copy application files
COPY teacher.jpg .
COPY Wav2Lip ./Wav2Lip
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
