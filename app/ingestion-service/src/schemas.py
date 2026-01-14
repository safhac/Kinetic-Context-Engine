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
    def profile(file_path):
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", file_path
        ]
        res = json.loads(subprocess.check_output(cmd))

        # Logic to determine view type and constraints
        streams = res.get('streams', [])
        video = next(s for s in streams if s['codec_type'] == 'video')
        audio = next((s for s in streams if s['codec_type'] == 'audio'), None)

        return {
            "is_vertical": int(video['height']) > int(video['width']),
            "has_audio": audio is not None,
            "duration": float(res['format']['duration']),
            "view_type": "headshot" if int(video['height']) < 720 else "full"
        }


class GestureSignal(BaseModel):
    task_id: str
    worker_type: str  # 'face' | 'body' | 'audio'
    timestamp: float
    text: str
    confidence: float

    # Requirements logic:
    def to_vtt_cue(self, profile: VideoProfile) -> Optional[str]:
        # Rule 1: Dismissal Logic
        if self.worker_type == "audio" and not profile.has_audio:
            return None
        if self.worker_type == "body" and profile.view_type == "headshot":
            return None

        # Rule 2: 2s Persistence
        start = self.timestamp
        end = self.timestamp + 2.0

        # Rule 3: TikTok/Vertical Positioning
        pos_map = {
            "face":  "line:10% align:center size:80%",
            "body":  "line:22% align:center size:80%",
            "audio": "line:34% align:center size:80%"
        }
        positioning = pos_map.get(self.worker_type, "line:90%")

        # Rule 4: Color Tagging
        return (
            f"{self.format_ts(start)} --> {self.format_ts(end)} {positioning}\n"
            f"<c.{self.worker_type}>{self.worker_type.upper()}: {self.text}</c.{self.worker_type}>\n"
        )

    @staticmethod
    def format_ts(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02}:{minutes:02}:{secs:06.3f}".replace('.', '.')
