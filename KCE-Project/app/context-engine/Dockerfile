FROM python:3.9-slim
WORKDIR /app
RUN pip install kafka-python
COPY src /app/src
ENV PYTHONUNBUFFERED=1
CMD ["python", "src/main.py"]