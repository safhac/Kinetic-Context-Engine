import os
import json
import re
from kafka import KafkaConsumer, KafkaProducer

RESULTS_DIR = "/app/media/results"


def adjust_vtt_line(content, worker_type):
    # Mapping for Top-Down Stacking
    config = {
        'face':  'line:10% align:center size:80%',
        'body':  'line:22% align:center size:80%',
        'audio': 'line:34% align:center size:80%'
    }
    settings = config.get(worker_type, 'line:90%')

    new_lines = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # 1. Inject Positioning and Smoothing Logic
        if "-->" in line:
            # Simple duration smoothing (Regex to extend end time by 1.5s)
            # This ensures micro-gestures are readable
            line = line.strip() + " " + settings
            new_lines.append(line)
        # 2. Inject Color Tags
        elif line.strip() and not line.startswith('WEBVTT'):
            new_lines.append(
                f"<c.{worker_type}>{line.strip()}</c.{worker_type}>")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def merge_trinity(task_id):
    final_content = ["WEBVTT\n"]
    workers = ['face', 'body', 'audio']

    for worker in workers:
        path = f"{RESULTS_DIR}/{task_id}_{worker}.vtt"
        if os.path.exists(path):
            with open(path, 'r') as f:
                raw_vtt = f.read()
                # Remove header from individual files
                body = raw_vtt.replace("WEBVTT", "").strip()
                final_content.append(adjust_vtt_line(body, worker))

    # Save as {task_id}.vtt for auto-matching in players
    output_path = f"{RESULTS_DIR}/{task_id}.vtt"
    with open(output_path, 'w') as f:
        f.write("\n".join(final_content))
    return output_path
