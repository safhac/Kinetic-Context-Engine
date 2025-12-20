FROM python:3.9-slim

WORKDIR /app

# 1. Install System Dependencies (Required for MediaPipe & OpenCV)
# libgl1-mesa-glx: Required for OpenGL (Computer Vision)
# libglib2.0-0: Required for basic image processing utilities
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the Source Code
# We copy the local 'src' folder into the container's '/app/src' folder
COPY src /app/src

# 4. Runtime Configuration
# Disable buffering to see logs immediately in Docker
ENV PYTHONUNBUFFERED=1

# 5. Launch Command
# Note: Now points to src/main.py instead of just main.py
CMD ["python", "src/main.py"]