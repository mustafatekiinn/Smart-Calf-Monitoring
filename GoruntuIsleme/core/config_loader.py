import yaml
import os
from typing import Dict
from pydantic import BaseModel, Field

# --- Pydantic Data Modelleri ---
# Bu modeller config.yaml dosyasından gelen verilerin tiplerini doğrular.

class ModelConfig(BaseModel):
    weights_path: str
    confidence: float = 0.25
    tracker: str = "bytetrack.yaml"
    device: str = "cpu"

class CameraConfig(BaseModel):
    source: str
    type: str = "local_frames"
    frame_pattern: str = "*.jpg"
    zones_profile: str

class ZonesConfig(BaseModel):
    file: str

class TrackingConfig(BaseModel):
    min_duration_seconds: int = 10
    fps: int = 1
    time_method: str = "ocr"  # "ocr", "filename" veya "fps"

class OutputConfig(BaseModel):
    csv_path: str
    log_level: str = "INFO"

class AppConfig(BaseModel):
    model: ModelConfig
    cameras: Dict[str, CameraConfig]
    zones: ZonesConfig
    tracking: TrackingConfig
    output: OutputConfig

# --- Yükleyici Fonksiyon ---

def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """
    Belirtilen yoldaki YAML yapılandırma dosyasını okur ve doğrulanmış AppConfig nesnesi döndürür.
    Fail-Fast kuralı gereği, dosya yoksa veya yapı hatalıysa uygulama anında hata verir.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[!] HATA: Yapılandırma dosyası bulunamadı: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    # Pydantic ile validasyon (Tip hataları varsa Pydantic ValidationError fırlatır)
    config = AppConfig(**data)
    return config
