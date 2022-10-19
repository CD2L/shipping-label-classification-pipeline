import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import re

import torch

import cv2

from pytesseract import pytesseract as pt
from pytesseract import Output

import tensorflow as tf  
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from transformers import TFAutoModelForSequenceClassification 

import string
import pickle
import shutil
import os
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

CLASSES = ['sender', 'receiver', 'unknown']

model_clf = load_model("./models/03_addr_clf.h5")
model_ident = TFAutoModelForSequenceClassification.from_pretrained("./models/addr-ident-model")

@st.cache
def OCR(im: np.ndarray, scale: int = 0):
    h, w = im.shape[:2]

    if isinstance(scale, float):
        scale = int(scale)
    w += w*scale
    h += h*scale

    bbox = im.copy()
    bbox = cv2.resize(
        bbox, (w, h), interpolation=cv2.INTER_AREA)  # rescaling
    bbox = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)

    config = r"-l eng --oem 3 --psm 3"
    text = pt.image_to_data(bbox, output_type=Output.DICT, config=config)

    bbox = cv2.cvtColor(bbox, cv2.COLOR_GRAY2BGR)

    n_boxes = len(text['level'])
    for i in range(n_boxes):
        if (text['text'][i] != ""):
            (x, y, w, h) = (text['left'][i], text['top']
                            [i], text['width'][i], text['height'][i])
            cv2.rectangle(bbox, (x, y), (x + w, y + h), (255, 0, 0), 4)

    return text, bbox




def preprocessing(X,y = None, max_words_c=200):
    with open('./models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        
    X_ = []
    for i, sentence in enumerate(X):
        tmp_sentence = sentence.lower()
        tmp_sentence = tmp_sentence.replace('\n', '')
        tmp_sentence = tmp_sentence.translate(translator)
        X_.append(tmp_sentence)
    X = X_.copy()
    
    X_encoded =  X.copy()
    X_encoded = map(lambda x: re.sub(r" +", " " , x), X_encoded)

    X_c = tokenizer.texts_to_sequences(X_encoded)
    X_c = sequence.pad_sequences(X_c, maxlen=max_words_c, padding='post')
    return X_c


# filename = st.file_uploader("Select an image to upload.", type=["png", "jpg", "jpeg"])
url = st.text_input("Enter a HTTP URL to an image", "https://a2btracking.com/wp-content/uploads/2017/07/MSL_RFID-1-e1499715334881.jpg")
im = np.array(Image.open(BytesIO(requests.get(url).content)).convert("L"))

cols = st.columns(2)
cols[0].image(im, caption="Uploaded image", use_column_width=None, width=500)

# Shipping label detection using YOLOv5
model_label_detection = torch.hub.load("yolov5/", "custom", path="./models/00_label_detection.pt", source="local")
im_with_bboxes = model_label_detection(im.copy())

cols[1].image(im_with_bboxes.render(),
              caption="Predicted image", use_column_width=None, width=500)

if os.path.exists("runs/"):
    shutil.rmtree("runs/")

# Get bounding boxes
bboxes = model_label_detection(im.copy()).crop(save=False)
bboxes = [np.array(bboxes[i]["im"]) for i in range(len(bboxes))]

rows = st.columns(len(bboxes))

all_texts, all_boxes = [], []
for id, b in enumerate(bboxes):
    text, box = OCR(b, scale=1)

    text = " ".join(" ".join(text["text"]).splitlines()).lower()

    all_texts.append(text)
    all_boxes.append(box)   

with open("./models/tokenizer_addr_identification.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

tokenized_data = []
for i in all_texts:
    tokenized_data.append(tokenizer.encode(i,truncation=True,padding=True,return_tensors="np"))

out = []
for i in tokenized_data:
    out.append(np.array(model_ident.predict(i)[0]))
out = np.array(out)
out = out.reshape(out.shape[0],-1)

addr_idx = [ idx for idx, _ in sorted(enumerate(out[:,1]), key=lambda x: x[1])[-2:]]

all_texts_new, all_boxes_new, all_decoded_addr, all_bboxes_new = [], [], [], []
for i in addr_idx:
    all_texts_new.append(tokenized_data[i])
    all_boxes_new.append(all_boxes[i])
    all_bboxes_new.append(bboxes[i])
    all_decoded_addr.append(all_texts[i])

# st.write(out[:, 0] > .5)

# st.write(out)
# st.write(all_texts_new)
# st.write(all_texts)

tokenized_data = preprocessing(all_decoded_addr)
st.write(tokenized_data)
addr_a = tokenized_data[0].reshape(1, -1)
addr_b = tokenized_data[1].reshape(1, -1)

# st.write(addr_a)
out = model_clf.predict({'input_addr_1': addr_a, 'input_addr_2': addr_b})
# st.write(out)

for id, b in enumerate(all_bboxes_new):
    cols = st.columns(4)
    st.text("")
    st.text("")
    if id == 0:
        cols[0].write("### Detected labels")
        cols[1].write("### Text detection")
        cols[2].write("### Text recognition")
        cols[3].write("### Classification")

    cols[0].image(b)
    cols[1].image(all_boxes_new[id])
    cols[2].write(all_decoded_addr[id])

    X = preprocessing(all_texts[id])[0]
    X = np.array(X[0]).reshape(1,-1)

    # st.write(X)
    # st.write(type(X))

    # pred = model_clf.predict(X)
    title = np.array(['Sender', 'Receiver'])
    cols[3].write(
        np.vstack((title, out[id][:])).T
    )
