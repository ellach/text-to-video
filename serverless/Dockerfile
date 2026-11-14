# Build from the repository root:  docker build -f serverless/Dockerfile -t wan-worker .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY serverless/requirements.txt .
RUN pip install --no-cache-dir -U -r requirements.txt
COPY generate_wan.py add_logo.py ./
COPY logoblend ./logoblend
COPY serverless/handler.py .
# Weights are NOT in the image (tens of GB). Mount a network volume with the model, or the first job downloads it.
ENV MODEL_DIR=/runpod-volume/models/Wan2.2-TI2V-5B-Diffusers HF_HOME=/runpod-volume/hf
CMD ["python", "-u", "handler.py"]
