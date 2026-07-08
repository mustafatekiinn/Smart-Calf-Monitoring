# Smart-Calf-Monitoring-System - AI Rules & Architectural Constitution

This document contains strict rules that the AI agent MUST obey to prevent the project from turning into spaghetti code; and to ensure the system can be easily configured, connected to live cameras, served via FastAPI, and updated in a modular way in the future.

---

## 1. Core Purpose & Focus
- **Purpose:** To monitor the health status, milk consumption rates, and behaviors of calves in farms using AI-supported systems.
- **Focus Area:** Gaining deep competence in Image Processing, time-series analysis, and establishing a stable, industrial-standard infrastructure.

---

## 2. Core Architectural Principles: Sustainability & Modularity
- **No Spaghetti Code:** No module can perform tasks outside its responsibility. Image processing, AI prediction, data management, FastAPI server, and UI layers MUST be completely decoupled from each other.
- **Configuration-Driven:** No threshold, file path, camera address (RTSP link), or model parameter will be hardcoded. Everything must be read from a centralized configuration file inside the `config/` directory.
- **Extensibility (Open/Closed):** Models (YOLO, LSTM, etc.) must be integrated via abstractions so they can be easily swapped without altering the core logic.

---

## 3. Project Directory Structure
AI, you must position all files you generate or update strictly adhering to this modular structure:
- `config/` -> System settings, camera definitions, API settings, and model thresholds (YAML/JSON).
- `core/` -> Core Business Logic.
  - `core/vision/` -> YOLO integration, video readers, image processing, and tracking.
  - `core/models/` -> LSTM prediction models, consumption/behavior, and health algorithms.
- `data/` -> Datasets, logs, database utilities, and data processing (Pandas) tools.
- `api/` -> **FastAPI** layer serving data to the outside world (Routers, Pydantic Schemas, Dependencies).
- `tests/` -> Unit and Integration tests proving the accuracy of the algorithms.

---

## 4. Coding & Clean Code Standards
- **Single Responsibility:** A function/class must only do one job.
- **Type Hinting & Validation:** Argument and return types (e.g., `def process_frame(frame: np.ndarray) -> dict:`) must be explicitly declared in all functions. Data structures should be validated using `Pydantic`.
- **Error Handling & Logging:** Robust `try-except` blocks must be used for RTSP drops, missing data, or model errors. Structured logging (via `loguru` or standard `logging`) is required. Errors must not be silenced (Fail-fast principle).

---

## 5. RFID and Camera ID Matching (Data Merging) Rules
- **Temporary vs. Absolute ID:** Calves entering the camera view without an RFID scan are assigned a "TEMP_ID_X". The moment an RFID tag is verified (e.g., "EAR_TAG_102"), all historical data tied to `TEMP_ID_X` must be retrospectively merged into `EAR_TAG_102`.
- **Timeout Mechanism:** Temporary IDs that appear briefly, disappear, and never match an RFID tag must be deleted after a specified timeout to prevent data bloat (ID explosion).
- **Performant Merging:** High-performance Pandas methods like `merge()` or `combine_first()` must be preferred for data merging operations.

---

## 6. Live Camera and RTSP Streaming Infrastructure
- **Video Source Abstraction:** The image processing module must not depend directly on the source. Use an abstract `VideoSource` class with subclasses like `LocalVideoReader` and `LiveCameraReader`.
- **Buffer Clearing (Anti-Lag):** Live streams must run on a separate **Thread** that continuously fetches the latest frame (dropping unread old frames). The YOLO model must only fetch the most recent frame from this queue to prevent multi-second lag.
- **Auto-Reconnect:** The system must not crash during network drops. It must smartly attempt to reconnect periodically and report its status via logs or the API.

---

## 7. Vision, Region Tracking, and Behavior Analysis
- **ROI (Region of Interest) Management:** Coordinates for waterers, feeders, and resting areas must be configurable. Calf entry and exit times to these zones must be tracked at second-level precision.
- **Strict 10-Second Rule:** Detections such as standing, lying, inactive, or head-down must NOT be recorded as a permanent behavior log unless they occur continuously for **at least 10 seconds**. This filters out flickering and false instant detections.

---

## 8. Feeding Robot Integration and Health Algorithm
- **1 Hz Frequency Data:** During robot visits, at least 14 parameters (ID, consumption, temperature, diaphragm power, pause counts, etc.) must be processed at a 1-second sampling rate (1 Hz).
- **Two-Stage Analysis:**
  1. **Real-time:** Immediate detection of sudden anomalies during the visit.
  2. **Profile & Reference:** Creating a long-term profile using the last 10 days' data (`dayCount: 10`) and scoring the calf's health by comparing the last 3 days' data against this profile.
- **Standard JSON Output:** The algorithm must produce a fully compliant JSON report containing averages (e.g., diaphragm), health points (`healthPoint`), and percentages.

---

## 9. Data Management & Persistence (Database & State)
- **Time-Series Focus:** For 1 Hz robot data and behavior timelines, time-series optimized databases (e.g., InfluxDB, TimescaleDB, or highly optimized SQL tables) should be preferred.
- **Config Validation:** YAML/JSON files in `config/` must be validated with Pydantic models on startup. The system should "Fail Fast" if misconfigured.

---

## 10. Additional AI Directives
- Always keep asynchronous compatibility (FastAPI async/await logic) and industrial standards (Dockerization) in mind when writing code.
- Optimize your algorithms considering performance costs and the constraints of edge computing (farm environments).
