import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.regularizers import l2

# Шлях до основної папки з даними
data_path = "flowers"

# Генератор для тренувальних даних з розширенням
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Нормалізація зображень (масштабування пікселів в діапазон [0,1])
    rotation_range=40,  # Випадкові обертання зображень на 40 градусів
    width_shift_range=0.2,  # Випадковий зсув зображень по горизонталі
    height_shift_range=0.2,  # Випадковий зсув зображень по вертикалі
    shear_range=0.2,  # Випадкові зміщення (shear transformations)
    zoom_range=0.2,  # Випадкове масштабування
    horizontal_flip=True,  # Випадковий горизонтальний віддзеркалення
    fill_mode='nearest',  # Заповнення пікселів після трансформацій
)

# Генератор для тестових даних (тільки нормалізація)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Завантаження тренувальних даних з розширенням
train_data = train_datagen.flow_from_directory(
    f"{data_path}/train",  # Папка з тренувальними даними
    target_size=(128, 128),  # Розмір зображень для масштабування
    batch_size=32,
    class_mode="categorical",  # Для багатокласової класифікації
)

# Завантаження тестових даних (без розширення, тільки нормалізація)
test_data = test_datagen.flow_from_directory(
    f"{data_path}/test",  # Папка з тестовими даними
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
)

# Побудова моделі з регуляризацією
model = Sequential()
model.add(Conv2D(128, (3, 3), input_shape=(128, 128, 3), kernel_regularizer=l2(0.01)))  # L2 регуляризація
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), kernel_regularizer=l2(0.01)))  # L2 регуляризація
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.01)))  # L2 регуляризація
model.add(Dropout(0.5))  # Dropout для регуляризації

# Останній шар: кількість нейронів = кількість класів (5 для 'daisy', 'dandelion', 'rose', 'sunflower', 'tulip')
model.add(Dense(5, activation="softmax"))  # Softmax для багатокласової класифікації

# Компіляція моделі
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks для моніторингу тренування
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
]

# Навчання моделі
model.fit(
    train_data,
    epochs=10,
    validation_data=test_data,  # Використовуємо тестові дані для валідації
    callbacks=callbacks
)

train_loss, train_acc = model.evaluate(train_data)
print(f"Train Accuracy: {train_acc*100:.2f}%, "
      f"Loss = {train_loss:.2f}%")

# Оцінка точності на тестових даних
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc*100:.2f}%, "
      f"Loss = {test_loss:.2f}%")