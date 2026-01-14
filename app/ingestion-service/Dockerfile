FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src /app/src

# Run with Uvicorn (Production-grade ASGI server)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]