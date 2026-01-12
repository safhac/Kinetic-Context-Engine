from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- NEW ARCHITECTURE SCHEMAS ---


class VideoIngestRequest(BaseModel):
    """
    Contract for /internal/ingest/video
    Used when Gateway hands off a full video file.
    """
    task_id: str
    file_path: str
    context: str = "general"
    original_name: str
    # The Ingestion Service uses this list to trigger specific Kafka topics
    pipelines: List[str] = Field(default=["face", "body", "audio"])


class TaskDispatch(BaseModel):
    """
    Contract for /internal/dispatch/{type}
    Used for direct injection (e.g., just trigger Body Worker)
    """
    task_id: str
    metadata: Dict = Field(default_factory=dict)
