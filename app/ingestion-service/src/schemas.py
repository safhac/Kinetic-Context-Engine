from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
from datetime import datetime

class VideoMetadata(BaseModel):
    source_id: str = Field(..., description="Unique device or camera ID")
    timestamp: float = Field(..., description="Unix timestamp of recording")
    # Strict regex enforces resolution format (e.g., "1920x1080")
    resolution: str = Field(..., regex="^\\d+x\\d+$", example="1920x1080")
    encoding: str = Field("h264", description="Video encoding format")

class TelemetryPayload(BaseModel):
    session_id: str
    metadata: VideoMetadata
    # Flexible dictionary for raw sensor readings (gyroscope, accelerometer)
    sensor_data: Dict[str, Any] = Field(..., description="Raw sensor readings (gyro, accel)")

    # Custom validator to ensure sensor data isn't empty
    @validator('sensor_data')
    def validate_sensor_data(cls, v):
        if not v:
            raise ValueError("Sensor data cannot be empty")
        return v