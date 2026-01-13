FROM python:3.9-slim

# 1. Install dependencies
WORKDIR /app
COPY ./api-gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy source code
# This puts your main code in /app/src
COPY ./api-gateway/src /app/src

# 3. Copy shared module
# This puts shared code in /app/shared
COPY ./shared /app/shared

# 4. Copy static files
# This puts index.html and app.js in /app/static
COPY ./api-gateway/static /app/static

# 5. Set Python path
# Ensures Python can find 'shared' and 'src' modules inside /app
ENV PYTHONPATH=/app

# 6. Run the app
# WORKDIR is already /app, so uvicorn can find 'src.main:app'
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]