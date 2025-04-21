import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(base_dir, '..', 'data', 'raw')
processed_data_dir = os.path.join(base_dir, '..', 'data', 'processed')

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

species_list = ['Orkide', 'Monstera', 'Aloe_vera']
statuses = ['healthy', 'diseased']

for split in ['train', 'validation', 'test']:
    os.makedirs(os.path.join(processed_data_dir, split), exist_ok=True)

for species in species_list:
    for status in statuses:
        source_dir = os.path.join(raw_data_dir, species, status)
        
        if not os.path.exists(source_dir):
            print(f"❌ Klasör bulunamadı: {source_dir}")
            continue

        images = os.listdir(source_dir)
        if len(images) == 0:
            print(f"⚠️ {source_dir} klasörü boş!")
            continue

        train_imgs, temp_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

        def copy_files(files, split_name):
            dst_folder = os.path.join(processed_data_dir, split_name, f"{species}_{status}")
            os.makedirs(dst_folder, exist_ok=True)
            for img in files:
                src = os.path.join(source_dir, img)
                dst = os.path.join(dst_folder, img)
                shutil.copy(src, dst)

        copy_files(train_imgs, 'train')
        copy_files(val_imgs, 'validation')
        copy_files(test_imgs, 'test')

print("✅ Klasörler tekrar oluşturuldu!")
