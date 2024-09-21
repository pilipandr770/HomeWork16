# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import os

# Завантаження моделей
@st.cache_resource
def load_models():
    cnn_model = load_model(os.path.join("models", "cnn_model.h5"))
    vgg16_model = load_model(os.path.join("models", "vgg16_model.h5"))
    return cnn_model, vgg16_model

cnn_model, vgg16_model = load_models()

# Функція передобробки зображень
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Нормалізація
    image = np.expand_dims(image, axis=0)
    return image

# Інтерфейс Streamlit
st.title("Image Classification Web App")

# Вибір моделі
model_choice = st.selectbox("Оберіть модель:", ["CNN", "VGG16"])
model = cnn_model if model_choice == "CNN" else vgg16_model
target_size = (64, 64) if model_choice == "CNN" else (224, 224)

# Завантаження зображення
uploaded_image = st.file_uploader("Завантажте зображення для класифікації:", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Завантажене зображення", use_column_width=True)

    # Передобробка та класифікація
    processed_image = preprocess_image(image, target_size)
    prediction = model.predict(processed_image)
    
    # Вивід ймовірностей
    st.write("Ймовірності для кожного класу:")
    probabilities = prediction[0]
    for idx, prob in enumerate(probabilities):
        st.write(f"Клас {idx}: {prob:.4f}")
    
    # Визначення найбільш ймовірного класу
    predicted_class = np.argmax(prediction)
    st.write(f"Передбачений клас: {predicted_class}")

# Додаткова функція для відображення графіків функції втрат та точності
if st.button('Показати графіки функції втрат та точності'):
    with open('history.pkl', 'rb') as f:
        histories = pickle.load(f)
    history = histories[model_choice]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['loss'], label='Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Функція втрат')
    ax1.legend()

    ax2.plot(history['accuracy'], label='Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Точність')
    ax2.legend()

    st.pyplot(fig)
