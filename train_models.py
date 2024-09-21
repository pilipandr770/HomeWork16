# train_models.py
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications import VGG16
from keras.utils import to_categorical
from keras.optimizers import Adam
import pickle

# Завантаження набору даних CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Створення CNN моделі
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Створення VGG16 моделі
def create_vgg16_model(input_shape=(224, 224, 3)):
    vgg16 = VGG16(weights=None, include_top=False, input_shape=input_shape)
    model = Sequential()
    model.add(vgg16)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Тренування моделей
cnn_model = create_cnn_model((32, 32, 3))
vgg16_model = create_vgg16_model((32, 32, 3))

# Навчання CNN моделі
print("Тренування CNN моделі...")
cnn_history = cnn_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64)

# Збереження CNN моделі
cnn_model.save('models/cnn_model.h5')

# Навчання VGG16 моделі
print("Тренування VGG16 моделі...")
vgg16_history = vgg16_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64)

# Збереження VGG16 моделі
vgg16_model.save('models/vgg16_model.h5')

# Збереження історії навчання
histories = {
    'CNN': cnn_history.history,
    'VGG16': vgg16_history.history
}

with open('history.pkl', 'wb') as f:
    pickle.dump(histories, f)
