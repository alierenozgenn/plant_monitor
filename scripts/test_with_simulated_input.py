import os
import cv2
from detect_disease_region import predict_disease  # aşağıda fonksiyon olarak ayıracağız

# Simülasyon: Bu klasördeki her görsel için test yap
simulated_camera_dir = r"C:\Users\aeren\plant_monitor\camera_inputs"

for filename in os.listdir(simulated_camera_dir):
    if not filename.endswith('.jpg'):
        continue

    image_path = os.path.join(simulated_camera_dir, filename)
    print(f"\n🎥 Görüntü işleniyor: {filename}")
    predict_disease(image_path)
