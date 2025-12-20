from pydantic import BaseModel
from typing import Optional

class ImageTask(BaseModel):
    task_id: str
    image_url: str
    metadata: Optional[dict] = {}

class TaskResult(BaseModel):
    task_id: str
    service: str
    status: str
    result: dict