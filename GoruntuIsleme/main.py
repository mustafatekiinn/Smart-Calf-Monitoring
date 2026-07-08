import os
import cv2
import json
import glob
import pandas as pd
import sys
from typing import Dict, List

# Proje ana dizinini Python path'ine ekle (ModuleNotFoundError hatasını önler)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from GoruntuIsleme.core.config_loader import load_config, AppConfig
from GoruntuIsleme.core.detector import CalfDetector, TrackedCalf
from GoruntuIsleme.core.tracker import ZoneTimeTracker, ZoneTimeRecord
from GoruntuIsleme.core.time_extractor import TimeExtractor
from GoruntuIsleme.utils.logger import get_logger

logger = get_logger("MainApp")

def load_farm_zones(file_path: str) -> Dict[str, Dict[str, List[List[int]]]]:
    if not os.path.exists(file_path):
        logger.warning(f"Bölge JSON dosyası bulunamadı: {file_path}")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_frame_files(source_dir: str, pattern: str) -> List[str]:
    # Eğer source_dir GoruntuIsleme içerisindeyse ama proje kökünden çalıştırılıyorsa
    if not os.path.exists(source_dir) and os.path.exists(os.path.join("GoruntuIsleme", source_dir)):
        source_dir = os.path.join("GoruntuIsleme", source_dir)
        
    search_path = os.path.join(source_dir, pattern)
    files = sorted(glob.glob(search_path))
    return files

def process_camera(camera_name: str, config: AppConfig, farm_zones: dict, detector: CalfDetector) -> pd.DataFrame:
    cam_config = config.cameras.get(camera_name)
    if not cam_config:
        logger.error(f"{camera_name} config dosyasında tanımsız.")
        return pd.DataFrame()

    zones_profile = cam_config.zones_profile
    camera_zones = farm_zones.get(zones_profile, {})
    if not camera_zones:
        logger.warning(f"{camera_name} için {zones_profile} adlı poligon haritası farm_zones.json'da yok!")

    logger.info(f"[{camera_name}] İşlem başlıyor. Kaynak: {cam_config.source}")

    tracker = ZoneTimeTracker(zones=camera_zones, min_duration=config.tracking.min_duration_seconds)
    
    frame_files = get_frame_files(cam_config.source, cam_config.frame_pattern)
    if not frame_files:
        logger.error(f"[{camera_name}] Hata: Görüntü dosyası bulunamadı! Yol: {cam_config.source}")
        return pd.DataFrame()
        
    logger.info(f"[{camera_name}] Toplam {len(frame_files)} kare işlenecek.")

    # Zaman Çıkarıcıyı Başlat (OCR / Filename / FPS)
    time_extractor = TimeExtractor(method=config.tracking.time_method, fps=config.tracking.fps)

    cv2.namedWindow(f"Canli Yayin - {camera_name}", cv2.WINDOW_NORMAL)

    for idx, img_path in enumerate(frame_files):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Yeni Sistem: Görüntüden doğrudan saati okuma (veya dosya adından)
        filename = os.path.basename(img_path)
        current_time_seconds = time_extractor.extract_time(frame, filename, idx)

        # 1. Adım: Buzağıları Bul
        tracked_calves = detector.track(frame)
        
        # 2. Adım: Bölgelerde geçirdikleri süreyi hesapla
        tracker.update(tracked_calves, current_time_seconds)

        # 3. Adım: Ekranda Görselleştirme
        for calf in tracked_calves:
            x1, y1, x2, y2 = calf.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, calf.center_point, 5, (0, 0, 255), -1)
            
            # Ekrandaki anlık bölge ismini öğren
            state = tracker.states.get(calf.track_id)
            zone_text = state.current_zone if state and state.current_zone else "Diger_Alan"
            
            label = f"ID: {calf.track_id} | {zone_text}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Poligonları Hayalet Çizgi Olarak Çiz
        for zone_name, poly in tracker.zones_poly.items():
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(frame, zone_name, (poly[0][0], poly[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow(f"Canli Yayin - {camera_name}", frame)
        
        # 'q' tuşu ile izlemeyi iptal et
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info(f"[{camera_name}] Kullanıcı tarafından manuel olarak durduruldu.")
            break

    cv2.destroyAllWindows()

    # Kameranın işi bitince raporları topla
    records = tracker.get_all_records()
    df = pd.DataFrame([r.__dict__ for r in records])
    
    if not df.empty:
        df['camera'] = camera_name
    return df

def main():
    logger.info("=== Smart Calf Monitoring System Başlatılıyor ===")
    
    # 1. Config Yükle
    config_path = "GoruntuIsleme/config/config.yaml"
    if not os.path.exists(config_path) and os.path.exists("config/config.yaml"):
        config_path = "config/config.yaml"

    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Config yüklenemedi: {e}")
        return

    # 2. Bölgeleri Yükle
    zones_file = config.zones.file
    if not os.path.exists(zones_file) and os.path.exists(os.path.join("GoruntuIsleme", zones_file)):
        zones_file = os.path.join("GoruntuIsleme", zones_file)
        
    farm_zones = load_farm_zones(zones_file)

    # 3. Model Dedektörünü Başlat
    detector = CalfDetector(config.model)

    # 4. Tüm Kameraları Sırayla İşle
    all_dfs = []
    for camera_name in config.cameras.keys():
        df_cam = process_camera(camera_name, config, farm_zones, detector)
        if not df_cam.empty:
            all_dfs.append(df_cam)

    # 5. CSV Raporlama Sistemi
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Çıktı klasörünü bul/oluştur
        out_path = config.output.csv_path
        if not os.path.isabs(out_path):
            # Her zaman GoruntuIsleme klasörü içine kaydet
            out_path = os.path.join(current_dir, out_path)
            
        output_dir = os.path.dirname(out_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # A) Ham Veri Raporu (Excel için ayırıcıyı noktalı virgül yapıyoruz)
        final_df.to_csv(out_path, index=False, sep=';')
        logger.info(f"Ham analiz raporu kaydedildi: {out_path}")
        
        # B) Gelişmiş Özet Rapor (ID ve Bölgelere Göre Toplam Dakika)
        final_df['duration_minutes'] = final_df['duration_seconds'] / 60.0
        summary_df = final_df.groupby(['track_id', 'camera', 'zone_name'])['duration_minutes'].sum().unstack(fill_value=0).reset_index()
        
        # Sütun isimlerini okunabilir yap (Örn: Yemlik -> Yemlik_dakika)
        new_cols = {}
        for col in summary_df.columns:
            if col not in ['track_id', 'camera']:
                new_cols[col] = f"{col}_dakika"
        summary_df.rename(columns=new_cols, inplace=True)
        
        # Toplam süreyi hesapla
        dakika_sutunlari = [col for col in summary_df.columns if col.endswith('_dakika')]
        summary_df['Toplam_dakika'] = summary_df[dakika_sutunlari].sum(axis=1)
        
        summary_path = out_path.replace(".csv", "_ozet.csv")
        summary_df.to_csv(summary_path, index=False, sep=';')
        logger.info(f"🚀 Özet analiz raporu kaydedildi: {summary_path}")

    else:
        logger.warning("Kaydedilecek hiçbir veri bulunamadı (10 sn'den uzun ziyaret yok veya klasörler boş).")

if __name__ == "__main__":
    main()
