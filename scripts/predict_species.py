import os
import tensorflow as tf
import numpy as np
import cv2

# Ayarlar
img_size = 128
model_path = r'C:\Users\aeren\plant_monitor\models\plant_species_classifier.keras'
input_dir = r'C:\Users\aeren\plant_monitor\camera_inputs'

# Modeli yükle
model = tf.keras.models.load_model(model_path)

# Sınıf etiketleri (eğitim sırasında alfabetik sıralanır)
class_names = sorted(os.listdir(r'C:\Users\aeren\plant_monitor\data\raw'))

# Görselleri sırayla işle
for filename in sorted(os.listdir(input_dir)):
    if not filename.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(input_dir, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Görsel okunamadı: {filename}")
        continue

    img_resized = cv2.resize(img, (img_size, img_size)) / 255.0
    prediction = model.predict(np.expand_dims(img_resized, axis=0))[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    print(f"📷 {filename} → Tahmin: {predicted_class} ({confidence:.2f}%)")
