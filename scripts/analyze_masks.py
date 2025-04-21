import os
import cv2
import numpy as np

# 🔍 Ayarlar
mask_dir = r'C:\Users\aeren\plant_monitor\segmentation_data\masks'
threshold_value = 30  # Hastalık olarak kabul edilecek minimum gri değeri

total = 0
fully_black = 0
usable = 0

# 📊 Piksel oranları için istatistik listesi
percent_nonzero = []

for filename in os.listdir(mask_dir):
    if not filename.endswith('.png'):
        continue

    path = os.path.join(mask_dir, filename)
    mask = cv2.imread(path, 0)  # Gri tonlama olarak oku

    if mask is None:
        print("❌ Okunamadı:", filename)
        continue

    total += 1

    # Maskede kaç piksel hastalık içeriyor?
    _, binary = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)
    nonzero = np.count_nonzero(binary)
    ratio = nonzero / (mask.shape[0] * mask.shape[1])
    percent_nonzero.append(ratio)

    if nonzero == 0:
        fully_black += 1
    else:
        usable += 1

# 🧾 Sonuçlar
print("\n📊 MASKELER ANALİZ SONUCU")
print(f"Toplam maske       : {total}")
print(f"Tamamen siyah      : {fully_black}")
print(f"Hastalık içeren    : {usable}")
print(f"Bozuk dosya        : {total - fully_black - usable}")

if usable > 0:
    print(f"\n📈 Ortalama hastalık oranı (non-zero pixels): {np.mean(percent_nonzero) * 100:.2f}%")
    print(f"🔺 Maksimum oran  : {np.max(percent_nonzero) * 100:.2f}%")
    print(f"🔻 Minimum oran  : {np.min(percent_nonzero) * 100:.2f}%")
