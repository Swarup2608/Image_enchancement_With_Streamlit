import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image # Image processing
from tensorflow.keras.preprocessing import image


# Loading the dog breed model
model = load_model("model.h5")

# Creating the app 
st.title("Image Enhancement Using Python")
st.markdown("Upload the image : ")

# Uploading the dog image
dog = st.file_uploader("Choose an image : ",type=["png","jpeg","jpg"])
submit = st.button("Increase Quality for the image")


if submit:
    if dog is not None:
        
        dog = Image.open(dog).convert('RGB')

        low_light_img = dog.resize((256,256),Image.NEAREST)
        image = image.img_to_array(low_light_img)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis = 0)
        Y_pred = model.predict(image)
        output_image = Y_pred[0] * 255.0
        output_image = output_image.clip(0,255)
        output_image = output_image.reshape((np.shape(output_image)[0],np.shape(output_image)[1],3))
        output_image = np.uint32(output_image)
        output = Image.fromarray(output_image.astype('uint8'),'RGB')
        st.image(output)
