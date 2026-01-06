import os
import json
import logging
from kafka import KafkaConsumer, KafkaProducer
from audio_processor import AudioProcessor

# Import your linguistic logic
try:
    from verbal_signals import get_active_verbal_signals
except ImportError:
    # Use standard import if running in Docker with correct PYTHONPATH
    from verbal_signals import get_active_verbal_signals

# Setup Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("audio-worker")

# Kafka Configuration
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:29092")
SOURCE_TOPIC = os.environ.get(
    "SOURCE_TOPIC", "raw_audio_chunk")  # Distinct topic for audio
DEST_TOPIC = os.environ.get("DEST_TOPIC", "processed_signals")


def main():
    logger.info("👂 Audio Worker Starting...")

    # 1. Initialize Processor
    processor = AudioProcessor(buffer_duration=4.0)  # Process every 4 seconds

    # 2. Connect to Kafka
    consumer = KafkaConsumer(
        SOURCE_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset='latest',
        group_id='kce-audio-worker'
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    logger.info(f"✅ Listening on {SOURCE_TOPIC}")

    for message in consumer:
        try:
            # Assume message.value is raw bytes of audio chunk
            # If you wrap it in JSON, you'll need to decode it first
            is_ready = processor.add_audio_chunk(message.value)

            if is_ready:
                logger.info("Processing Audio Buffer...")
                text, pitch = processor.process_buffer()

                if text:
                    logger.info(
                        f"🗣️ Transcript: '{text}' (Pitch: {pitch:.1f}Hz)")

                    # 3. Analyze Signals
                    signals = get_active_verbal_signals(
                        text,
                        current_pitch=pitch,
                        baseline_pitch=processor.baseline_pitch
                    )

                    # 4. Broadcast Signals
                    for sig in signals:
                        output = {
                            "signal": sig,
                            "intensity": 1.0,  # Could scale based on pitch deviation
                            "timestamp": message.timestamp / 1000.0,
                            "source": "audio_analysis",
                            "metadata": {"text_fragment": text}
                        }
                        producer.send(DEST_TOPIC, output)
                        logger.info(f"⚡ Sent Signal: {sig}")

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")


if __name__ == "__main__":
    main()
