import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
import cv2
from PIL import Image
import numpy as np

loaded_model = load_model("digitsnew.h5")

st.title(" ?????? ")
sketch = st_canvas(stroke_width=15, height=300, stroke_color="#FFFFFF",background_color="#000000", key="full_app")
bt = st.button("RESULT")
if bt:
    if sketch.image_data is not None:
        if sketch.image_data is not None:
            img = cv2.resize(sketch.image_data.astype(np.uint8), (28, 28))
            img_rescalling = cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_NEAREST)
            st.image(img_rescalling)
            x_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            y_pred = loaded_model.predict(x_img.reshape(1, 28, 28))
            st.write(f"result: {np.argmax(y_pred[0])}")
            st.bar_chart(y_pred[0])
