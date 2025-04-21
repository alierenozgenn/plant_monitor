import tensorflow as tf
import numpy as np
import cv2
import os

img_size = 256
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'unet_leaf_segmentation.keras')
model = tf.keras.models.load_model(model_path)

def predict_disease(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Görsel okunamadı:", image_path)
        return

    img_resized = cv2.resize(img, (img_size, img_size)) / 255.0
    pred_mask = model.predict(np.expand_dims(img_resized, axis=0))[0, :, :, 0]

    _, pred_mask_bin = cv2.threshold(pred_mask, 0.3, 1, cv2.THRESH_BINARY)
    pred_mask_bin = (pred_mask_bin * 255).astype(np.uint8)

    disease_pixels = np.sum(pred_mask_bin > 0)
    print("🧪 Min:", pred_mask.min(), "Max:", pred_mask.max(), "Mean:", pred_mask.mean())
    print("🩺 Hastalık piksel sayısı:", disease_pixels)

    if disease_pixels > 100:
        print("⚠️ HASTALIK TESPİT EDİLDİ!")
    else:
        print("✅ Yaprak sağlıklı.")

    # Görselin üzerine maskeyi uygula
    overlay = (img_resized * 255).astype(np.uint8)
    overlay[pred_mask_bin > 128] = [0, 0, 255]

    save_path = image_path.replace("camera_inputs", "processed_outputs")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)
    print(f"💾 Kaydedildi: {save_path}")
