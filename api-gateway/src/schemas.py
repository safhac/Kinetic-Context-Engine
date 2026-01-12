from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict
from datetime import datetime
import uuid

# --- GOVERNANCE POLICY ---
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'mp3', 'wav', 'ogg', 'flac'}


class VideoUploadTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    content_type: str
    size_bytes: int
    context_tag: str = "general"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator('size_bytes')
    def validate_size(cls, v):
        if v > MAX_FILE_SIZE:
            raise ValueError(
                f"File exceeds maximum size of {MAX_FILE_SIZE/1024/1024}MB")
        return v

    @field_validator('filename')
    def validate_extension(cls, v):
        ext = v.split('.')[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Extension .{ext} not allowed. Supported: {ALLOWED_EXTENSIONS}")
        return v


class ProcessingEvent(BaseModel):
    """The Event emitted to Kafka"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    event_type: str = "MEDIA_RECEIVED"
    payload: Dict
    source: str = "api-gateway"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --- LEGACY / SPECIFIC ENDPOINT SCHEMAS ---
class ImageTask(BaseModel):
    task_id: str
    image_url: str
    metadata: Optional[dict] = {}


class AudioTask(BaseModel):
    task_id: str
    audio_url: str
    metadata: Optional[dict] = {}
