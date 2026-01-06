import pyaudio
import time
from kafka import KafkaProducer
import numpy as np
import sys

# Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
KAFKA_TOPIC = "raw_audio_chunk"


def list_microphones(p):
    print("\n🎤 Available Audio Devices:")
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    found = False
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            name = p.get_device_info_by_host_api_device_index(0, i).get('name')
            print(f"   [{i}] {name}")
            found = True
    if not found:
        print("   ❌ No microphones found!")
    print("")


def main():
    p = pyaudio.PyAudio()

    # 1. Let user pick a device if defaults fail
    list_microphones(p)
    device_index = input(
        "Enter Microphone Device Index (or press Enter for default): ")
    device_index = int(device_index) if device_index.strip() else None

    # 2. Connect to Kafka
    try:
        producer = KafkaProducer(
            bootstrap_servers="localhost:9092", value_serializer=None)
        print(f"✅ Connected to Kafka.")
    except Exception as e:
        print(f"❌ Kafka Error: {e}")
        return

    # 3. Open Stream
    try:
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        print("\n🎙️  Streaming... (Talk to test!)")

        while True:
            # Read raw bytes
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            # --- DEBUG: VOLUME METER ---
            # Convert bytes back to numpy array to measure loudness
            floats = np.frombuffer(data, dtype=np.float32)
            volume = np.linalg.norm(floats) * 10

            # Visual bar: |||||||
            bars = "|" * int(volume)
            if int(volume) > 0:
                sys.stdout.write(f"\r🔊 Level: {bars:<50}")
                sys.stdout.flush()
                # Only send to Kafka if there is sound (optional optimization)
                producer.send(KAFKA_TOPIC, data)
            else:
                sys.stdout.write(
                    f"\r🔇 Silence...                                   ")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        producer.close()


if __name__ == "__main__":
    main()
