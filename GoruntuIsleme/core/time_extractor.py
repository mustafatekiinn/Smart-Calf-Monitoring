import cv2
import re
from GoruntuIsleme.utils.logger import get_logger

logger = get_logger("TimeExtractor")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr kütüphanesi bulunamadı! OCR (Metin Okuma) çalışmayacak.")
    logger.warning("Kurmak için: pip install easyocr")

class TimeExtractor:
    """
    Kare (frame) içerisinden veya dosya isminden gerçek zamanı saniye cinsinden hesaplar.
    """
    def __init__(self, method: str = "ocr", fps: int = 1):
        self.method = method
        self.fps = fps
        self.reader = None
        self.last_valid_time = 0.0
        
        if self.method == "ocr":
            if EASYOCR_AVAILABLE:
                logger.info("EasyOCR yapay zeka okuyucusu yükleniyor... (İlk açılışta biraz sürebilir)")
                # Sadece İngilizce (rakamlar için yeterli) ve GPU olmadan başlatıyoruz
                self.reader = easyocr.Reader(['en'], gpu=False) 
            else:
                logger.error("EasyOCR kurulu olmadığı için zaman çıkarma metodu 'fps' olarak değiştirildi.")
                self.method = "fps"

    def extract_time(self, frame, filename: str, frame_index: int) -> float:
        # --- YÖNTEM 1: DOSYA İSMİNDEN OKUMA ---
        # Örn: kamera3_1781981418359.jpg (milisaniye cinsinden UNIX zaman damgası)
        if self.method == "filename":
            match = re.search(r'_(\d{13})', filename)
            if match:
                self.last_valid_time = int(match.group(1)) / 1000.0
                return self.last_valid_time
                
        # --- YÖNTEM 2: OCR İLE GÖRÜNTÜDEN OKUMA ---
        elif self.method == "ocr" and self.reader:
            # Görüntünün sadece sol üst köşesini kırp (Hız ve performans için)
            h, w = frame.shape[:2]
            roi = frame[0:int(h * 0.15), 0:int(w * 0.35)]
            
            # Metnin daha iyi okunabilmesi için resmi gri tonlamaya çevir
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # OCR işlemi (detail=0 sadece metni döndürür)
            results = self.reader.readtext(gray, detail=0)
            text = " ".join(results)
            
            # Metnin içinde saat formatı ara (Örn: 14:32:10 veya 14.32.10)
            time_match = re.search(r'(\d{2})[:\.](\d{2})[:\.](\d{2})', text)
            if time_match:
                hours, mins, secs = map(int, time_match.groups())
                # Günün toplam saniyesini hesapla
                current_time = hours * 3600 + mins * 60 + secs
                self.last_valid_time = current_time
                return current_time
            else:
                # Eğer o karede saat net okunamazsa (bulanıklık vb.) tahmini süre ekle
                self.last_valid_time += (1.0 / max(1, self.fps))
                return self.last_valid_time

        # --- YÖNTEM 3: FPS BAZLI (VARSAYILAN) ---
        self.last_valid_time = frame_index * (1.0 / max(1, self.fps))
        return self.last_valid_time
