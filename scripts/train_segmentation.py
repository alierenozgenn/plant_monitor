import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model

# Ayarlar
img_size = 256
epochs = 30

img_dir = r'C:\Users\aeren\plant_monitor\segmentation_data\images'
mask_dir = r'C:\Users\aeren\plant_monitor\segmentation_data\masks'
model_save_path = r'C:\Users\aeren\plant_monitor\models\unet_leaf_segmentation.keras'

# Veri setini yükle
X, Y = [], []
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

for filename in img_files:
    base_id = os.path.splitext(filename)[0]
    img_path = os.path.join(img_dir, filename)
    mask_path = os.path.join(mask_dir, base_id + ".png")

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if img is None or mask is None:
        print(f"❌ Hatalı dosya: {filename}")
        continue

    # Maske renkli ise griye çevir
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Normalize ve yeniden boyutlandır
    img = cv2.resize(img, (img_size, img_size)) / 255.0
    mask = cv2.resize(mask, (img_size, img_size))

    # Binary hale getir (0-1)
    _, mask = cv2.threshold(mask, 30, 1, cv2.THRESH_BINARY)

    X.append(img)
    Y.append(np.expand_dims(mask, axis=-1))

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

# Eğitim ve doğrulama ayrımı
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# Basit U-Net Modeli
def build_unet(input_size=(img_size, img_size, 3)):
    inputs = Input(input_size)

    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)

    u2 = UpSampling2D()(c3)
    merge2 = Concatenate()([u2, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(merge2)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)

    u1 = UpSampling2D()(c4)
    merge1 = Concatenate()([u1, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(merge1)
    c5 = Conv2D(16, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Eğitimi başlat
model = build_unet()
model.summary()
model.fit(X_train, Y_train, epochs=epochs, batch_size=8, validation_data=(X_val, Y_val))
model.save(model_save_path)

print(f"\n✅ Model başarıyla kaydedildi: {model_save_path}")
