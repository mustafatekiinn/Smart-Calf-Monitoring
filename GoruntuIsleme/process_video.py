import os
import cv2
import sys
import argparse
import pandas as pd
from datetime import datetime

# Proje ana dizinini Python path'ine ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from GoruntuIsleme.core.config_loader import load_config
from GoruntuIsleme.core.detector import CalfDetector
from GoruntuIsleme.core.tracker import ZoneTimeTracker
from GoruntuIsleme.core.time_extractor import TimeExtractor
from GoruntuIsleme.utils.logger import get_logger
from GoruntuIsleme.main import load_farm_zones

logger = get_logger("VideoProcessor")

def generate_reports(tracker: ZoneTimeTracker, camera_name: str, video_path: str, output_dir: str):
    records = tracker.get_all_records()
    if not records:
        logger.warning("Kaydedilecek hiçbir veri bulunamadı.")
        return

    df = pd.DataFrame([r.__dict__ for r in records])
    df['camera'] = camera_name

    # Videonun kendi ismini al (Örn: "gece.mp4" -> "gece")
    video_basename = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_basename)[0]

    os.makedirs(output_dir, exist_ok=True)
    
    # A) Ham Rapor
    raw_csv_path = os.path.join(output_dir, f"{video_name_no_ext}.csv")
    df.to_csv(raw_csv_path, index=False, sep=';')
    logger.info(f"Ham analiz raporu kaydedildi: {raw_csv_path}")

    # B) Özet Rapor (Pandas Pivot İşlemleri - main.py ile birebir aynı)
    df['duration_minutes'] = df['duration_seconds'] / 60.0
    
    agg_df = df.groupby(['track_id', 'camera', 'zone_name']).agg(
        toplam_sure=('duration_minutes', 'sum'),
        ziyaret_sayisi=('duration_minutes', 'count')
    ).reset_index()
    
    pivot_sure = agg_df.pivot(index=['track_id', 'camera'], columns='zone_name', values='toplam_sure').fillna(0)
    pivot_sayi = agg_df.pivot(index=['track_id', 'camera'], columns='zone_name', values='ziyaret_sayisi').fillna(0)
    
    pivot_sure.columns = [f"{col}_dakika" for col in pivot_sure.columns]
    pivot_sayi.columns = [f"{col}_ziyaret" for col in pivot_sayi.columns]
    
    summary_df = pd.concat([pivot_sure, pivot_sayi], axis=1).reset_index()
    
    dakika_sutunlari = [col for col in summary_df.columns if col.endswith('_dakika')]
    summary_df['Toplam_dakika'] = summary_df[dakika_sutunlari].sum(axis=1)
    
    ziyaret_sutunlari = [col for col in summary_df.columns if col.endswith('_ziyaret')]
    summary_df['Toplam_ziyaret'] = summary_df[ziyaret_sutunlari].sum(axis=1)
    
    summary_path = os.path.join(output_dir, f"{video_name_no_ext}_ozet.csv")
    summary_df.to_csv(summary_path, index=False, sep=';')
    logger.info(f"Özet analiz raporu kaydedildi: {summary_path}")


def process_video(video_path: str, camera_name: str, skip_frames: int):
    # 1. Config ve Bölgeleri Yükle
    config_path = os.path.join(current_dir, "config", "config.yaml")
    config = load_config(config_path)
    
    zones_file = os.path.join(current_dir, config.zones.file)
    farm_zones = load_farm_zones(zones_file)
    camera_zones = farm_zones.get(camera_name, {})
    
    if not camera_zones:
        logger.error(f"HATA: '{camera_name}' için data/farm_zones.json dosyasında poligon çizimi bulunamadı!")
        logger.error(f"Önce 'python GoruntuIsleme/utils/roi_selector.py' ile {camera_name} alanlarını çizmelisin.")
        return

    # 2. Modülleri Başlat (Kullanıcının isteği üzerine OCR kullanılıyor)
    detector = CalfDetector(config.model)
    tracker = ZoneTimeTracker(zones=camera_zones, min_duration=config.tracking.min_duration_seconds)
    time_extractor = TimeExtractor(method="ocr", fps=config.tracking.fps)

    # 3. Videoyu Aç
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Video açılamadı veya bulunamadı: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video Yüklendi -> Toplam Kare: {total_frames} | Video FPS'i: {fps:.1f}")
    logger.info(f"Hızlandırma Aktif -> Her {skip_frames} karede 1 analiz yapılacak.")

    frame_count = 0
    processed_count = 0
    
    # Dosya adını pencere isminde göster
    window_name = f"Video Analiz - {os.path.basename(video_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        # Hızlandırma için sadece grab() yap, görüntüyü decode etme (Zaman kazandırır)
        ret = cap.grab()
        if not ret:
            break
            
        frame_count += 1
        
        # Sadece belirlenen aralıkta bir frame'i decode edip işle
        if frame_count % skip_frames != 0:
            continue
            
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue
            
        processed_count += 1
        
        # A. Zamanı OCR ile kameradan doğrudan oku
        # (Yapay zeka sadece okuduğu kareyi analiz ettiği için performans kaybı minimumdur)
        current_time_seconds = time_extractor.extract_time(frame, f"frame_{frame_count}.jpg", processed_count)
        
        # B. Buzağıları Bul ve Duruş Analizi Yap (Yatıyor/Ayakta)
        tracked_calves = detector.track(frame)
        
        # C. Bölge sürelerini güncelle
        tracker.update(tracked_calves, current_time_seconds)
        
        # D. Görselleştirme
        for calf in tracked_calves:
            x1, y1, x2, y2 = calf.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, calf.center_point, 5, (0, 0, 255), -1)
            
            state = tracker.states.get(calf.track_id)
            zone_text = state.current_zone if state and state.current_zone else "Diger_Alan"
            
            label = f"ID: {calf.track_id} | {zone_text}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Çizilen poligonları (bölgeleri) ekranda göster
        for zone_name, poly in tracker.zones_poly.items():
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(frame, zone_name, (poly[0][0], poly[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Ekrana bilgi yazdır
        progress = f"Islenen Kare: {processed_count} (Video Suresi: %{int(frame_count/total_frames*100)})"
        cv2.putText(frame, progress, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow(window_name, frame)
        
        # 'q' ile iptal edebilme
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Kullanıcı işlemi manuel olarak durdurdu.")
            break

    # Videoyu kapat
    cap.release()
    cv2.destroyAllWindows()

    # Çıktıları üret
    output_dir = os.path.join(current_dir, "output")
    generate_reports(tracker, camera_name, video_path, output_dir)
    logger.info("🎉 Video analizi başarıyla tamamlandı!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Çevrimdışı (Offline) Video Analiz ve Raporlama Aracı")
    parser.add_argument("--video", type=str, required=True, help="Analiz edilecek videonun tam veya bağıl yolu (Örn: data/gece.mp4)")
    parser.add_argument("--camera", type=str, required=True, help="Hangi kameranın çizimlerinin (bölgelerinin) kullanılacağı (Örn: kamera_3)")
    parser.add_argument("--skip-frames", type=int, default=30, help="Kaç karede bir işlem yapılacağı. Hızlandırmak için (Örn: 30 kare = saniyede 1 kare)")
    
    args = parser.parse_args()
    process_video(args.video, args.camera, args.skip_frames)
