import pyaudio
import time
from kafka import KafkaProducer
import numpy as np

# Configuration matches AudioProcessor settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # Send small packets (low latency)
KAFKA_TOPIC = "raw_audio_chunk"


def main():
    # 1. Connect to Kafka (running on localhost:9092 via mapped ports)
    try:
        producer = KafkaProducer(
            bootstrap_servers="localhost:9092",
            # Send raw bytes, no JSON serialization needed for audio
            value_serializer=None
        )
        print(f"✅ Connected to Kafka. Target Topic: {KAFKA_TOPIC}")
    except Exception as e:
        print(f"❌ Kafka Connection Failed: {e}")
        return

    # 2. Setup Microphone (PyAudio)
    p = pyaudio.PyAudio()

    try:
        stream = p.open(
            format=pyaudio.paFloat32,  # Important: Matches np.float32 in AudioProcessor
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        print("🎙️  Microphone Live! Streaming to KCE Brain... (Press Ctrl+C to stop)")
        print("🗣️  Say something: 'I'm innocent I tell ya!'")

        while True:
            # Read raw data from mic
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            # Send raw bytes to Kafka
            producer.send(KAFKA_TOPIC, data)

    except KeyboardInterrupt:
        print("\n🛑 Stopping Stream...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        producer.close()


if __name__ == "__main__":
    main()
