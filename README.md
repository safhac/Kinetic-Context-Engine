# Kinetic Context Engine (KCE)


**⚠️ ETHICAL DISCLAIMER & LIMITATIONS**

This software is a **Proof of Concept (PoC)** for multi-modal sensor fusion and signal processing. It is designed for **research and educational purposes only**.

* **Not a Truth Verifier:** This system detects *stress indicators* (physiological and behavioral), which correlate with but do **not** prove deception. Innocent individuals may trigger high stress scores due to anxiety, and practiced liars may trigger low scores.  
* **No Legal/Medical Standing:** The output of this software should never be used for legal, employment, medical, or interpersonal decision-making.  
* **Bias Warning:** The underlying models (MediaPipe, Whisper) and heuristic rules may exhibit bias across different demographics, accents, or neurotypes.



**Digitizing Human Non-Verbal Communication**

## Vision

The **Kinetic Context Engine (KCE)** is a distributed, event-driven system designed to bridge the gap between **Sensation** (raw video/telemetry) and **Perception** (semantic meaning).

Instead of treating video analysis as a black box, KCE decomposes streams into atomic signals (micro-expressions, body language) and reconstructs them into context-aware meanings using a transparent, configurable rules engine.

## Why Open Source?

We believe that digitizing human behavior requires a global, collaborative effort. We have open-sourced the core engine to solve two fundamental challenges identified in our [Strategic Analysis](OPEN_SOURCE_STRATEGY.md):

### 1. The "Dictionary Problem" (Crowdsourcing Perception)

Building a comprehensive library of human non-verbal communication is too complex for a single team.

* **The Goal:** We provide the structure (`rules.json`); we ask **behavioral scientists, psychologists, and researchers** to fill it.

* **The Ask:** Submit Pull Requests defining new gestures (e.g., "Eyebrow Flash", "Defensive Crossing") mapping signals to meaning.

### 2. Universal Sensor Integration (Crowdsourcing Sensation)

Computer Vision evolves rapidly. Today the standard is MediaPipe; tomorrow it might be something else.

* **The Goal:** A modular `SensorInterface` that abstracts the "How" (CV library) from the "What" (Signal).

* **The Ask:** **CV Engineers** can write simple adapters for new libraries (MMPose, OpenFace, YOLO) to make them compatible with the KCE ecosystem.

### 3. Ethical Transparency

In the field of Affective Computing, "Black Boxes" create distrust. By open-sourcing our interpretation logic, we ensure that the system's decisions—how it reads a "smile" or a "frown"—are auditable, verifiable, and free from hidden biases.

## Architecture

The system follows a **Microservices Architecture** powered by **Apache Kafka**:

1. **Ingestion Layer:** API Gateway accepting video/telemetry.

2. **Sensation Layer (AI Workers):** Modular workers (currently **MediaPipe**) that extract atomic signals (e.g., `eyebrow_raise`, `intensity: 0.8`).

3. **Perception Layer (Context Adapter):** The "Brain" that subscribes to signals and applies the `rules.json` logic to infer meaning (e.g., "Friendly Greeting").

4. **Infrastructure:** Kafka (Streaming), Redis (State/Task Queue), and Celery (Orchestration).

*Upcoming:* We are currently migrating the orchestration layer from Docker Compose to **HashiCorp Nomad** for distributed resilience.

## Getting Started

### Prerequisites

* Docker & Docker Compose

* Python 3.9+

### Installation

1. Clone the repository:

   ```bash
   git clone [https://github.com/your-org/kinetic-context-engine.git](https://github.com/your-org/kinetic-context-engine.git)
   cd kinetic-context-engine