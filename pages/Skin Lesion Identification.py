import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import numpy as np

from model import ClassifierNet

# Load model
model = ClassifierNet()

img_input = layers.Input(shape=(256, 256, 3))
meta_input = layers.Input(shape=(3,))

model((img_input, meta_input))
model.load_weights('weights/weights_part_3.h5')

# Load images
images = os.listdir('Images/')
cols = st.columns(len(images))

for i in range(len(images)):
    image_file = 'Images/' + images[i]

    with cols[i]:
        st.image(image_file)
        button = st.button('Choose', key=str(i))

        if button:
            image = Image.open(image_file)
            image = image.resize((256, 256))
            image = np.array(image).reshape((1, 256, 256, 3)) / 255.0

            outputs = model((image, np.array([[25., 1., 0.]])))

            st.text('Nevus: ' + str(outputs[0][0].numpy()))
            st.text('Melanoma: ' + str(outputs[0][1].numpy()))
            st.text('Seborrheic Keratosis: ' + str(outputs[0][2].numpy()))

