import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np

# Шлях до основної папки з даними
data_path = "flowers"

# Використання ImageDataGenerator для автоматичного завантаження та нормалізації
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

# Завантаження даних без використання навчання або валідації
data = datagen.flow_from_directory(
    f"{data_path}/train",
    target_size=(128, 128),  # Розмір зображень для масштабування
    batch_size=32,
    class_mode="categorical",  # Для багатокласової класифікації
    shuffle=False  # Не змішувати дані, щоб отримати послідовність зображень по класах
)

# Розвідковий аналіз даних (EDA)

# 1. Перевірка категорій та їх кількості
class_names = list(data.class_indices.keys())
print(f"Категорії класів: {class_names}")
print(f"Кількість класів: {len(class_names)}")
print(f"Кількість зображень в кожній категорії:")
for class_name in class_names:
    print(f"{class_name}: {len(os.listdir(os.path.join(f"{data_path}/train", class_name)))} зображень")

# 2. Перевірка балансу класів
class_counts = [len(os.listdir(os.path.join(f"{data_path}/train", class_name))) for class_name in class_names]
plt.figure(figsize=(10, 6))
plt.bar(class_names, class_counts)
plt.title('Розподіл класів у наборі даних')
plt.xlabel('Категорії')
plt.ylabel('Кількість зображень')
plt.xticks(rotation=45)
plt.show()

# 3. Візуалізація декількох зображень для кожного класу
plt.figure(figsize=(10, 10))
for i, class_name in enumerate(class_names):
    class_images, _ = data.__next__()  # Змінено тут

    for j in range(2):
        plt.subplot(len(class_names), 2, i * 2 + j + 1)
        plt.imshow(class_images[j])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(class_name)
plt.tight_layout()
plt.show()

# 4. Перевірка розмірів зображень
image_shape = data.image_shape
print(f"Розміри зображень після масштабування: {image_shape}")
