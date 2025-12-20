FROM python:3.9-slim

WORKDIR /app

# 1. Install System Dependencies (Updated for Debian Trixie)
# 'libgl1' replaces the deprecated 'libgl1-mesa-glx'
# 'libglib2.0-0' is required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Source (Ensure you have moved main.py to src/ first!)
COPY src /app/src

# 4. Runtime Config
ENV PYTHONUNBUFFERED=1

CMD ["python", "src/main.py"]