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


class VideoProfile(BaseModel):
    task_id: str
    is_vertical: bool
    has_audio: bool
    duration: float
    view_type: str  # 'headshot' | 'full'

    @staticmethod
    def from_file(task_id: str, file_path: str):
        """Runs ffprobe to profile the video."""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", file_path
        ]
        res = json.loads(subprocess.check_output(cmd))
        streams = res.get('streams', [])

        # Get video stream
        video = next((s for s in streams if s['codec_type'] == 'video'), None)
        # Get audio stream
        audio = next((s for s in streams if s['codec_type'] == 'audio'), None)

        if not video:
            raise Exception("No video stream found")

        return VideoProfile(
            task_id=task_id,
            is_vertical=int(video.get('height', 0)) > int(
                video.get('width', 0)),
            has_audio=audio is not None,
            duration=float(res.get('format', {}).get('duration', 0)),
            view_type="headshot" if int(
                video.get('height', 0)) < 720 else "full"
        )
