import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data', 'processed')

img_height, img_width = 224, 224
batch_size = 16

# Gelişmiş Veri Artırma Teknikleri
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7,1.3]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = val_test_datagen.flow_from_directory(
    os.path.join(data_dir, 'validation'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# EfficientNetB3 ile model oluştur
base_model = EfficientNetB3(weights='imagenet', include_top=False,
                            input_shape=(img_height, img_width, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# İlk başta tüm katmanları dondur (transfer learning)
base_model.trainable = False

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# İlk aşama eğitimi
print("İlk eğitim (transfer learning)...")
model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator)

# Fine-tuning için üst katmanları aç
base_model.trainable = True

# Son katmanlar hariç tümünü dondur (fine-tuning için)
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Fine-tuning için çok küçük öğrenme oranıyla tekrar derle
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Fine-tuning eğitimi
print("Fine-tuning eğitimi...")
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator)

# Modeli test et ve kaydet
loss, accuracy = model.evaluate(test_generator)
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

model.save(os.path.join(base_dir, '..', 'models', 'plant_model_advanced.keras'))
