import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from GoruntuIsleme.core.detector import TrackedCalf
from GoruntuIsleme.utils.logger import get_logger

logger = get_logger("ZoneTimeTracker")

@dataclass
class ZoneTimeRecord:
    track_id: int
    zone_name: str
    enter_time: float
    exit_time: float
    duration_seconds: float

@dataclass
class CalfZoneState:
    track_id: int
    current_zone: Optional[str] = None
    zone_enter_time: float = 0.0
    zone_history: List[ZoneTimeRecord] = field(default_factory=list)

class ZoneTimeTracker:
    """Buzağıların bölgelerde geçirdiği süreyi hesaplayan ana algoritma."""
    
    def __init__(self, zones: Dict[str, List[List[int]]], min_duration: int = 10):
        # JSON'dan gelen x,y listesini OpenCV'nin anladığı formata (numpy array) çevir
        self.zones_poly: Dict[str, np.ndarray] = {
            zone_name: np.array(points, dtype=np.int32) 
            for zone_name, points in zones.items()
        }
        self.min_duration = min_duration
        self.states: Dict[int, CalfZoneState] = {}

    def _determine_zone(self, point: tuple[int, int]) -> Optional[str]:
        """Verilen X, Y koordinatının hangi çokgenin (bölgenin) içinde olduğunu tespit eder."""
        for zone_name, poly in self.zones_poly.items():
            # >= 0 demek noktanın poligonun içinde veya tam kenarında olduğu anlamına gelir
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                return zone_name
        return None

    def update(self, tracked_calves: List[TrackedCalf], current_time: float) -> None:
        """
        Her frame'de çalışan ana güncelleme fonksiyonu.
        Buzağının bölge değiştirip değiştirmediğine bakar ve 10 saniye kuralını işler.
        """
        active_ids = set()

        for calf in tracked_calves:
            track_id = calf.track_id
            active_ids.add(track_id)
            
            detected_zone = self._determine_zone(calf.center_point)
            
            # Buzağı sisteme ilk defa giriyorsa state nesnesini oluştur
            if track_id not in self.states:
                self.states[track_id] = CalfZoneState(track_id=track_id)
            
            state = self.states[track_id]
            
            # Durum Değişikliği (Eski bölgeden çıktıysa veya yeni bölgeye girdiyse)
            if state.current_zone != detected_zone:
                
                # 1. Eski bölgeden çıkış işlemlerini yap
                if state.current_zone is not None:
                    duration = current_time - state.zone_enter_time
                    
                    # ai.md kuralı: 10 saniyeden kısa süre kalındıysa anlık bir hatadır (flickering), sayma!
                    if duration >= self.min_duration:
                        record = ZoneTimeRecord(
                            track_id=track_id,
                            zone_name=state.current_zone,
                            enter_time=state.zone_enter_time,
                            exit_time=current_time,
                            duration_seconds=duration
                        )
                        state.zone_history.append(record)
                        logger.debug(f"ID {track_id} | {state.current_zone} bölgesinden çıktı. (Süre: {duration:.1f} sn)")
                
                # 2. Yeni bölgeye giriş işlemlerini yap
                state.current_zone = detected_zone
                state.zone_enter_time = current_time
                if detected_zone:
                    logger.debug(f"ID {track_id} | {detected_zone} bölgesine girdi.")

        # Eğer bir buzağı ekrandan tamamen çıktıysa ve kameradan kaybolduysa açık olan sayacını durdur
        missing_ids = set(self.states.keys()) - active_ids
        for m_id in missing_ids:
            state = self.states[m_id]
            if state.current_zone is not None:
                duration = current_time - state.zone_enter_time
                if duration >= self.min_duration:
                    record = ZoneTimeRecord(
                        track_id=m_id,
                        zone_name=state.current_zone,
                        enter_time=state.zone_enter_time,
                        exit_time=current_time,
                        duration_seconds=duration
                    )
                    state.zone_history.append(record)
                # Durumu tamamen sıfırla ki ekrana geri geldiğinde yeni kayıt açsın
                state.current_zone = None

    def get_all_records(self) -> List[ZoneTimeRecord]:
        """Tüm tamamlanmış bölge ziyaret kayıtlarını tek bir listede toplar."""
        all_records = []
        for state in self.states.values():
            all_records.extend(state.zone_history)
        return all_records
