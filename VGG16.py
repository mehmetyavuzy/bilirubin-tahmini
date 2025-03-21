import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

# Veri setinin bulunduğu doğru dizini belirtelim.
dataset_path = r"C:/Users/yavuz/Desktop/veri seti/jaundiced_normal/"

# Görüntü boyutları.
IMG_SIZE = 128


# 1. Görüntüleri ve etiketleri yükleyen bir fonksiyon tanımlayalım.
def load_images_and_labels(dataset_path, img_size):
    """Veri setini yükler, görüntüleri ölçeklendirir ve etiketlerle birleştirir."""
    images = []
    labels = []
    classes = os.listdir(dataset_path)  # Sınıf isimlerini al.

    for idx, cls in enumerate(classes):  # Her sınıfı işle.
        class_path = os.path.join(dataset_path, cls)
        for img_name in os.listdir(class_path):  # Sınıftaki tüm görüntüleri işle.
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:  # Geçerli bir görüntü varsa işle.
                img = cv2.resize(img, (img_size, img_size))  # Görüntüyü boyutlandır.
                images.append(img)
                labels.append(idx)  # Sınıf etiketini ekle.
    return np.array(images), np.array(labels), classes


# Görüntüleri ve etiketleri yükleyelim.
images, labels, classes = load_images_and_labels(dataset_path, IMG_SIZE)

# 2. Veriyi eğitim ve test setlerine ayıralım.
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

# 3. Görüntüleri [0, 1] aralığına normalleştirelim.
X_train = X_train / 255.0
X_test = X_test / 255.0

# Etiketleri one-hot encode yapalım.
y_train_onehot = to_categorical(y_train, num_classes=len(classes))
y_test_onehot = to_categorical(y_test, num_classes=len(classes))

# 4. Veri artırma (augmentation) işlemi
datagen = ImageDataGenerator(
    rotation_range=30,  # Daha fazla döndürme
    width_shift_range=0.3,  # Daha fazla kaydırma
    height_shift_range=0.3,  # Daha fazla kaydırma
    shear_range=0.3,  # Daha fazla kırpma
    zoom_range=0.3,  # Daha fazla yakınlaştırma
    horizontal_flip=True,  # Yatay çevirme
    fill_mode='nearest'  # Doldurma modu
)

datagen.fit(X_train)

# 5. Transfer öğrenme ile önceden eğitilmiş model kullanarak daha derin bir model oluşturalım
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Yeni katmanlar ekleyerek modelimizi tamamlıyoruz.
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)  # Yeni bir dense katmanı ekledik
x = Dropout(0.5)(x)
x = Dense(len(classes), activation='softmax')(x)

# Yeni modelimizi oluşturuyoruz.
model = Model(inputs=base_model.input, outputs=x)

# Transfer öğrenme katmanlarını serbest bırakma (son katmanları eğitebiliriz)
for layer in base_model.layers:
    layer.trainable = False  # Sadece son katmanlar eğitilecek, ilk katmanlar donmuş olacak

# Modeli derliyoruz.
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Erken Durdurma (Early Stopping) ve Öğrenme Oranı Azaltma (Learning Rate Scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# 7. Modeli eğitelim
history = model.fit(
    datagen.flow(X_train, y_train_onehot, batch_size=32),
    validation_data=(X_test, y_test_onehot),
    epochs=50,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler]  # Erken durdurma ve lr scheduler ekledik
)

# 8. Test seti doğruluk oranını kontrol edelim.
test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
print(f"Test Doğruluğu: {test_accuracy * 100:.2f}%")

# 9. Sonuçları görselleştirelim
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.title('Eğitim ve Doğrulama Doğruluğu')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kayıpları')
plt.plot(history.history['val_loss'], label='Doğrulama Kayıpları')
plt.legend()
plt.title('Eğitim ve Doğrulama Kayıpları')

plt.show()