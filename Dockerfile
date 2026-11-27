# RunPod GitHub build: use the default Dockerfile path
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 tesseract-ocr tesseract-ocr-heb tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY serverless/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# The ad editor (Qwen-Image-Edit-2511) needs diffusers >= 0.36; it lives in its own folder so the video worker keeps diffusers 0.35.1.
RUN pip install --no-cache-dir --no-deps --target /opt/edit-libs "diffusers==0.36.0"
COPY generate_wan.py generate_image.py add_logo.py ./
COPY logoblend ./logoblend
COPY serverless/handler.py serverless/template_analyze.py serverless/edit_runner.py ./
# Weights are NOT in the image (tens of GB). Mount a network volume with the model, or the first job downloads it.
ENV MODEL_DIR=/runpod-volume/models/Wan2.2-TI2V-5B-Diffusers HF_HOME=/runpod-volume/hf
CMD ["python", "-u", "handler.py"]
