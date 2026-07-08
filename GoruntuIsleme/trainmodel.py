from ultralytics import YOLO
import os

def main():
    print("🚀 Model Eğitimi (Fine-Tuning) Başlıyor...")
    
    # Bulunduğumuz GoruntuIsleme dizinini al
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Mevcut en iyi modelin yolunu belirtiyoruz (Transfer Learning)
    model_path = os.path.join(current_dir, "runs", "detect", "buzagi_modelimiz", "weights", "best.pt")
    
    if not os.path.exists(model_path):
        print(f"⚠️ Hata: {model_path} bulunamadı!")
        print("Lütfen ilk eğitimi temel model (yolo11n.pt) ile yapın veya yolu kontrol edin.")
        return

    print(f"✅ Mevcut ağırlıklar yüklendi: {model_path}")
    
    # Modeli mevcut ağırlıklarla yüklüyoruz
    model = YOLO(model_path) 

    # Veri seti yolu ve çıktı (runs) yolu
    data_path = os.path.join(current_dir, "buzagi_veriseti", "data.yaml")
    project_dir = os.path.join(current_dir, "runs", "detect")

    if not os.path.exists(data_path):
        print(f"⚠️ Hata: Veri seti bulunamadı -> {data_path}")
        print("Lütfen Roboflow'dan indirdiğiniz klasörü 'GoruntuIsleme' içine 'buzagi_veriseti' adıyla kopyalayın.")
        return

    # Modeli kendi veri setimizle eğitiyoruz
    model.train(
        data=data_path, 
        epochs=30,           # Fine-tuning için 30 epoch
        imgsz=640, 
        batch=8,
        lr0=0.001,           # Düşük öğrenme oranı = eski bilgileri koruma
        project=project_dir,
        name="buzagi_modelimiz_finetuned",
        device=0             # <--- GPU Kullanımını Zorunlu Kılar
    )

    print("🎉 Eğitim tamamlandı!")
    print("Yeni model ağırlıkları 'GoruntuIsleme/runs/detect/buzagi_modelimiz_finetuned/weights/best.pt' konumuna kaydedildi.")
    print("Kullanmaya başlamak için config.yaml dosyasındaki weights_path ayarını güncelleyebilirsiniz.")

if __name__ == "__main__":
    main()