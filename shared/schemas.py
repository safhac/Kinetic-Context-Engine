from pydantic import BaseModel
from typing import Optional


class VideoProfile(BaseModel):
    task_id: str
    is_vertical: bool
    has_audio: bool
    duration: float
    view_type: str  # 'headshot' | 'full'


class GestureSignal(BaseModel):
    task_id: str
    worker_type: str  # 'face', 'body', 'audio'
    timestamp: float
    text: str
    confidence: float

    def to_vtt_cue(self, profile: VideoProfile) -> Optional[str]:
        """Handles dismissal logic and vertical stacking positioning."""
        # 1. Dismissal Logic
        if self.worker_type == "audio" and not profile.has_audio:
            return None
        if self.worker_type == "body" and profile.view_type == "headshot":
            return None

        # 2. Timing (2s persistence)
        start = self.timestamp
        end = self.timestamp + 2.0

        # 3. Positioning (Top Stacking for Vertical Video)
        # Face is highest, then Body, then Audio
        pos_map = {
            "face":  "line:10% align:center size:80%",
            "body":  "line:22% align:center size:80%",
            "audio": "line:34% align:center size:80%"
        }
        pos = pos_map.get(self.worker_type, "line:90%")

        return (
            f"{self.format_ts(start)} --> {self.format_ts(end)} {pos}\n"
            f"<c.{self.worker_type}>{self.worker_type.upper()}: {self.text}</c.{self.worker_type}>\n\n"
        )

    @staticmethod
    def format_ts(seconds: float) -> str:
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        return f"{int(hrs):02}:{int(mins):02}:{secs:06.3f}"
