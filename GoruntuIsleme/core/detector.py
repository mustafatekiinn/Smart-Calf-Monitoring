import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO

from GoruntuIsleme.core.config_loader import ModelConfig
from GoruntuIsleme.utils.logger import get_logger

logger = get_logger("Detector")

@dataclass
class TrackedCalf:
    track_id: int
    bbox: Tuple[int, int, int, int]
    center_point: Tuple[int, int]
    confidence: float

class CalfDetector:
    """YOLO tabanlı buzağı tespit ve takip (tracking) sınıfı."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        logger.info(f"YOLO Modeli yükleniyor: {self.config.weights_path}")
        
        # YOLO modelini oluştur
        try:
            self.model = YOLO(self.config.weights_path)
            logger.info("YOLO Modeli başarıyla yüklendi.")
        except Exception as e:
            logger.error(f"Model yüklenirken hata oluştu: {e}")
            raise

    def track(self, frame: np.ndarray) -> List[TrackedCalf]:
        """
        Verilen frame üzerinde ByteTrack algoritması ile takip yapar.
        Ekranda tespit edilen her bir buzağı için TrackedCalf nesnesi döndürür.
        """
        # verbose=False yaparak konsolu kalabalıktan kurtarıyoruz
        results = self.model.track(
            frame, 
            persist=True, 
            tracker=self.config.tracker, 
            conf=self.config.confidence, 
            verbose=False,
            device=self.config.device
        )

        tracked_calves = []
        if not results:
            return tracked_calves

        result = results[0]
        
        # Eğer ekranda takip edilen bir nesne (ID atanmış) varsa
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.int().cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.float().cpu().tolist()

            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = box
                
                # Tepeden çekim kamerasında buzağının merkezini hesapla
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                tracked_calves.append(TrackedCalf(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    center_point=(center_x, center_y),
                    confidence=conf
                ))

        return tracked_calves
