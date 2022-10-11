import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np

import torch
import tensorflow as tf  

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)


# A workaround to center the images on fullscreen.
st.markdown("""
    <style>
        button[title^=Exit]+div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        # [data-testid="stHorizontalBlock"] {
        #     align-items: center;
        # }
    </style>
    """, unsafe_allow_html=True)

st.title("PoC Demo : Shipping Labels Classification Pipeline")
st.write("""
    This project aims to use deep learning algorithms to detect whether a product is counterfeit by analyzing images of the box and its labels.
    
    This repository is PoC Demo using web scraped images, realized during sprints, of how the pipeline actually works. Here is an overview of the pipeline:

    Github repository: [https://github.com/CD2L/shipping-label-classification-pipeline](https://github.com/CD2L/shipping-label-classification-pipeline).
    """)

# filename = st.file_uploader("Select an image to upload.", type=["png", "jpg", "jpeg"])
url = st.text_input("Enter a HTTP URL to an image", "https://a2btracking.com/wp-content/uploads/2017/07/MSL_RFID-1-e1499715334881.jpg")
im = np.array(Image.open(BytesIO(requests.get(url).content)))

cols = st.columns(2)
cols[0].image(im, caption="Uploaded image", use_column_width=None, width=500)

# Shipping label detection using YOLOv5
model_label_detection = torch.hub.load("ultralytics/yolov5", "custom", path="./models/00_label_detection.pt", source="local")
im_with_bboxes = model_label_detection(im)

cols[1].image(im, caption="Predicted image", use_column_width=None, width=500)



# model_clf = tf.models.load_model("./models/00_clf.h5")
