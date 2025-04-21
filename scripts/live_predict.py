import tensorflow as tf
import cv2
import numpy as np
import os

# Modeli yükle
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'plant_model_advanced.keras')
model = tf.keras.models.load_model(model_path)

# Sınıf isimlerini alalım (otomatik):
data_dir = os.path.join(base_dir, '..', 'data', 'processed', 'train')
class_names = sorted(os.listdir(data_dir))

img_height, img_width = 224, 224

# Kamera aç
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera açılamadı!")
    exit()

print("✅ Kamera açıldı, çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Kameradan görüntü alınamadı!")
        break

    # Görüntüyü hazırla
    img = cv2.resize(frame, (img_width, img_height))
    img_array = np.expand_dims(img / 255.0, axis=0)

    # Tahmin et
    predictions = model.predict(img_array, verbose=0)
    pred_index = np.argmax(predictions)
    class_label = class_names[pred_index]

    # Ekrana yazdır
    label_text = f"Tespit: {class_label}"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Eğer hastalıklıysa (diseased içeriyorsa) bildirim göster
    if 'diseased' in class_label:
        warning_text = "UYARI: Bitki Hastalikli!"
        cv2.putText(frame, warning_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    # Görüntüyü göster
    cv2.imshow('Canli Bitki Takibi', frame)

    # Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
