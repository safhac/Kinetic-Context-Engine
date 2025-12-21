# ai-workers/main.py
from celery import Celery
import os
import sys

# Add root to path so we can import shared schemas if needed
sys.path.append(os.getcwd())

# 1. Initialize the Celery App
celery_app = Celery(
    'kce_worker_cluster',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/0'
)

# 2. Register your tasks
# This tells Celery: "Go look in the body-worker folder for a file named tasks.py"
celery_app.conf.imports = [
    'body-worker.tasks', 
    # 'face-worker.tasks' # Uncomment when you are ready to build the face worker
]

# 3. Optional: Configure specific settings
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='Asia/Jerusalem',
    enable_utc=True,
)