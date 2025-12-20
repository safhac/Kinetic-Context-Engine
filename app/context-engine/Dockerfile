FROM python:3.9-slim

WORKDIR /app

# Install dependencies (Kafka client)
RUN pip install --no-cache-dir kafka-python

# Copy source code
COPY src /app/src

# Unbuffered output for immediate logging in Docker
ENV PYTHONUNBUFFERED=1

CMD ["python", "src/main.py"]