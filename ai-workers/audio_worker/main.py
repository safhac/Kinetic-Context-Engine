import os
import json
import logging
import librosa
from kafka import KafkaConsumer, KafkaProducer
from audio_processor import AudioProcessor

# Logging Setup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("audio-worker")

# Configuration
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:29092")
SOURCE_TOPIC = os.environ.get("SOURCE_TOPIC", "audio-tasks")
DEST_TOPIC = os.environ.get("DEST_TOPIC", "processed_signals")


def main():
    logger.info("👂 Audio Worker Starting...")

    # Initialize Processor (No buffer duration needed for full file processing)
    processor = AudioProcessor()

    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id="kce-audio-worker",
        session_timeout_ms=60000,
        heartbeat_interval_ms=10000,
        max_poll_interval_ms=900000
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    logger.info(f"✅ Listening on {SOURCE_TOPIC}")

    for message in consumer:
        try:
            task = message.value
            logger.info(f"Processing Task: {task.get('task_id')}")

            # 1. Get File Path
            file_path = task.get("file_path")
            if not file_path or not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                continue

            # 2. Load Audio from Disk (Resample to 16k for Whisper)
            logger.info(f"Loading audio from {file_path}...")
            # librosa loads as float32 automatically
            audio_array, _ = librosa.load(file_path, sr=16000)

            # 3. Process the Full Audio
            text, pitch = processor.process_array(audio_array)

            if text:
                logger.info(
                    f"🗣️ Transcript: '{text[:50]}...' (Pitch: {pitch:.1f}Hz)")

                output = {
                    "task_id": task.get("task_id"),
                    "signal": "verbal_analysis",
                    "text": text,
                    "pitch": pitch,
                    "source": "audio_analysis"
                }
                producer.send(DEST_TOPIC, output)
                logger.info(f"⚡ Sent Result")

        except Exception as e:
            logger.error(f"Error processing audio task: {e}")


if __name__ == "__main__":
    main()
