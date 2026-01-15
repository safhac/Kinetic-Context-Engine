import json
import subprocess
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class VideoProfile(BaseModel):
    task_id: str
    is_vertical: bool
    has_audio: bool
    duration: float
    view_type: str  # 'headshot' | 'full'

    @staticmethod
    def from_file(task_id: str, file_path: str):
        """Runs ffprobe to profile the video."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", file_path
            ]
            # check_output returns bytes, so we decode or let json.loads handle it
            output = subprocess.check_output(cmd)
            res = json.loads(output)

            streams = res.get('streams', [])

            # Get video stream
            video = next(
                (s for s in streams if s['codec_type'] == 'video'), None)
            # Get audio stream
            audio = next(
                (s for s in streams if s['codec_type'] == 'audio'), None)

            if not video:
                # Fallback if ffprobe sees no video stream
                return VideoProfile(
                    task_id=task_id, is_vertical=False, has_audio=False,
                    duration=0.0, view_type="full"
                )

            width = int(video.get('width', 0))
            height = int(video.get('height', 0))
            duration = float(res.get('format', {}).get('duration', 0))

            return VideoProfile(
                task_id=task_id,
                is_vertical=(height > width),
                has_audio=(audio is not None),
                duration=duration,
                view_type="headshot" if height < 720 else "full"
            )
        except Exception as e:
            print(f"Error profiling video: {e}")
            # Return a safe default so the pipeline doesn't crash
            return VideoProfile(
                task_id=task_id, is_vertical=False, has_audio=True,
                duration=0.0, view_type="full"
            )

# --- DEPRECATED BUT KEEPING FOR TYPE SAFETY ---
# These are less critical now that we don't have the Gateway,
# but good to keep if you extend the API later.


class VideoIngestRequest(BaseModel):
    task_id: str
    file_path: str
    context: str = "general"
    original_name: str
    pipelines: List[str] = Field(default=["face", "body", "audio"])


class TaskDispatch(BaseModel):
    task_id: str
    metadata: Dict = Field(default_factory=dict)
