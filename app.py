```python
import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

st.write("Versión de Python:", platform.python_version())

model = load_model("keras_model.h5")

st.title("Reconocimiento de Personas")

st.write("Toma una fotografía para comprobar si hay una persona en la imagen.")

with st.sidebar:
    st.subheader("Reconocimiento")
    st.write("Esta aplicación utiliza un modelo entrenado en Teachable Machine para identificar si hay una persona en la imagen.")

img_file_buffer = st.camera_input("Toma una foto")

if img_file_buffer is not None:

    img = Image.open(img_file_buffer).convert("RGB")

    st.image(img, caption="Imagen capturada", use_container_width=True)

    img = img.resize((224, 224))

    img_array = np.array(img)

    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    data[0] = normalized_image_array

    prediction = model.predict(data, verbose=0)

    if prediction[0][0] > prediction[0][1]:
        st.success("👤 En la imagen hay una persona")
    else:
        st.info("🚫 En la imagen no hay nadie")
```



