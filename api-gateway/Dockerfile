FROM python:3.9-slim

WORKDIR /app

COPY ./api-gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module
COPY ./shared /app/shared
# Copy src
COPY ./api-gateway/src /app/src
COPY api-gateway/static /app/static

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]