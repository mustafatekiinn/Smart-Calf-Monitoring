import cv2
import json
import os
import sys
import numpy as np
import glob

# Proje ana dizinini Python path'ine ekle (ModuleNotFoundError hatasını önler)
current_dir = os.path.dirname(os.path.abspath(__file__))
goruntu_isleme_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(goruntu_isleme_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from GoruntuIsleme.core.config_loader import load_config

# 1. Config'i Yükle (Kamera klasörlerini otomatik bulmak için)
try:
    config_path = os.path.join(goruntu_isleme_dir, "config", "config.yaml")
    config = load_config(config_path)
    
    JSON_YOLU = config.zones.file
    if not os.path.isabs(JSON_YOLU):
        JSON_YOLU = os.path.join(goruntu_isleme_dir, JSON_YOLU)
except Exception as e:
    print(f"Uyarı: Config yüklenemedi. ({e})")
    JSON_YOLU = os.path.join(goruntu_isleme_dir, "data", "farm_zones.json")
    config = None

noktalar = []
ekran_olcegi = 1.0

def tiklama_olayi(olay, x, y, flags, param):
    global noktalar, ekran_olcegi
    if olay == cv2.EVENT_LBUTTONDOWN:
        gercek_x = int(x / ekran_olcegi)
        gercek_y = int(y / ekran_olcegi)
        noktalar.append([gercek_x, gercek_y])
        print(f"📍 Nokta eklendi: ({gercek_x}, {gercek_y})")

def json_kaydet(kamera_adi, bolge_adi, koordinatlar):
    if os.path.exists(JSON_YOLU):
        with open(JSON_YOLU, "r", encoding="utf-8") as f:
            try:
                veri = json.load(f)
            except json.JSONDecodeError:
                veri = {}
    else:
        veri = {}
        os.makedirs(os.path.dirname(JSON_YOLU), exist_ok=True)

    # Kamera profilini oluştur
    if kamera_adi not in veri:
        veri[kamera_adi] = {}

    veri[kamera_adi][bolge_adi] = koordinatlar

    with open(JSON_YOLU, "w", encoding="utf-8") as f:
        json.dump(veri, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ BAŞARILI: '{bolge_adi}' bölgesi kaydedildi!")

def main():
    global noktalar, ekran_olcegi
    print("--- 🗺️ AKILLI ÇİFTLİK BÖLGE SEÇİCİ ---")
    
    if config:
        print("\nKonfigürasyondaki Kameralar:")
        for cam_name in config.cameras.keys():
            print(f" - {cam_name}")
            
    kamera_adi = input("\nKamera profil adını girin (Örn: kamera_3): ")
    bolge_adi = input("Çizeceğiniz bölgenin adını girin (Örn: Yemlik): ")

    # Kameranın ilk fotoğrafını OTOMATİK bul!
    test_foto_yolu = None
    if config and kamera_adi in config.cameras:
        cam_config = config.cameras[kamera_adi]
        
        # Absolute path kullanarak klasörü bul
        search_path = os.path.join(goruntu_isleme_dir, cam_config.source, cam_config.frame_pattern)
             
        files = sorted(glob.glob(search_path))
        if files:
            test_foto_yolu = files[0]
            print(f"[*] İlk fotoğraf otomatik bulundu: {test_foto_yolu}")
            
    if not test_foto_yolu:
        test_foto_yolu = input("Fotoğraf bulunamadı. Lütfen test fotoğrafı yolunu manuel girin: ")

    img = cv2.imread(test_foto_yolu)
    if img is None:
        print(f"\n❌ Hata: Fotoğraf yüklenemedi! ({test_foto_yolu})")
        return

    MAX_EKRAN_GENISLIK = 1280
    orj_h, orj_w = img.shape[:2]

    # Ekran sığdırması
    if orj_w > MAX_EKRAN_GENISLIK:
        ekran_olcegi = MAX_EKRAN_GENISLIK / orj_w
        yeni_w = int(orj_w * ekran_olcegi)
        yeni_h = int(orj_h * ekran_olcegi)
        gosterim_resmi = cv2.resize(img, (yeni_w, yeni_h))
    else:
        ekran_olcegi = 1.0
        gosterim_resmi = img.copy()

    # --- ÖNCEKİ ÇİZİMLERİ HAYALET OLARAK YÜKLE ---
    mevcut_bolgeler = {}
    if os.path.exists(JSON_YOLU):
        with open(JSON_YOLU, "r", encoding="utf-8") as f:
            try:
                veri = json.load(f)
                if kamera_adi in veri:
                    mevcut_bolgeler = veri[kamera_adi]
            except:
                pass

    pencere_ismi = f"Bolge Secici - [{kamera_adi} > {bolge_adi}] | S:Kaydet | C:Temizle | Q:Cikis"
    cv2.namedWindow(pencere_ismi, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(pencere_ismi, tiklama_olayi)

    print("\n[!] Fotoğraf açıldı. Alanı belirlemek için köşelere tıklayın.")
    print("İşiniz bitince fotoğraf ekranı aktifken klavyeden 'S' tuşuna basarak kaydedin.")
    
    while True:
        kopya_img = gosterim_resmi.copy()
        
        # Eski çizimleri turuncu hayalet çizgilerle göster
        for m_bolge, m_koordinatlar in mevcut_bolgeler.items():
            if m_bolge == bolge_adi:
                continue 
                
            pts = np.array([[int(x * ekran_olcegi), int(y * ekran_olcegi)] for x, y in m_koordinatlar], np.int32)
            cv2.polylines(kopya_img, [pts], True, (0, 165, 255), 2) 
            cv2.putText(kopya_img, m_bolge, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Güncel olarak tıklanan noktaları kırmızı yuvarlak ile göster
        for i, nokta in enumerate(noktalar):
            ekran_noktasi = (int(nokta[0] * ekran_olcegi), int(nokta[1] * ekran_olcegi))
            cv2.circle(kopya_img, ekran_noktasi, 5, (0, 0, 255), -1) 
            if i > 0:
                prev_ekran = (int(noktalar[i-1][0] * ekran_olcegi), int(noktalar[i-1][1] * ekran_olcegi))
                cv2.line(kopya_img, prev_ekran, ekran_noktasi, (0, 255, 0), 2)
        
        # Çokgeni tamamlayan yeşil çizgiyi oluştur
        if len(noktalar) > 2:
            ilknokta = (int(noktalar[0][0] * ekran_olcegi), int(noktalar[0][1] * ekran_olcegi))
            sonnokta = (int(noktalar[-1][0] * ekran_olcegi), int(noktalar[-1][1] * ekran_olcegi))
            cv2.line(kopya_img, sonnokta, ilknokta, (0, 255, 0), 2)

        cv2.imshow(pencere_ismi, kopya_img)
        tus = cv2.waitKey(1) & 0xFF

        if tus == ord('s') or tus == ord('S'):
            if len(noktalar) >= 3:
                json_kaydet(kamera_adi, bolge_adi, noktalar)
                break
            else:
                print("\n⚠️ Uyarı: En az 3 nokta seçmelisiniz!")
        elif tus == ord('c') or tus == ord('C'):
            noktalar = []
            print("\n🧹 Noktalar temizlendi.")
        elif tus == ord('q') or tus == ord('Q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()