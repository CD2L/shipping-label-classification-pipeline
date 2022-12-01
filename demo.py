import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import re

import shutil

import torch

import cv2

from pytesseract import pytesseract as pt
from pytesseract import Output
from collections import Counter

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from transformers import TFAutoModelForSequenceClassification

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

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
st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

st.title("PoC Demo")
# st.write(
#     """
#     This project aims to use deep learning algorithms to detect whether a product is counterfeit by analyzing images of the box and its labels.

#     This repository is PoC Demo using web scraped images, realized during sprints, of how the pipeline actually works. Here is an overview of the pipeline:

#     Github repository: [https://github.com/CD2L/shipping-label-classification-pipeline](https://github.com/CD2L/shipping-label-classification-pipeline).
#     """
# )

# tab2 = st.tabs([ "Address Classifier"])

# filename = st.file_uploader("Select an image to upload.", type=["png", "jpg", "jpeg"])
url = st.text_input(
    "Enter a HTTP URL to an image",
    # "https://a2btracking.com/wp-content/uploads/2017/07/MSL_RFID-1-e1499715334881.jpg",
)

# im = np.array(Image.open(filename))
im = np.array(Image.open(BytesIO(requests.get(url).content)))

CLASSES = ["sender", "receiver", "unknown"]

model_clf = load_model("./models/03_addr_clf.h5")
# model_ident = load_model("./models/glove_embedding_identification.h5")
model_ident_tokenization = load_model("./models/15_addr_identification.h5")


@st.cache
def OCR(im: np.ndarray, scale: int = 0):
    h, w = im.shape[:2]

    if isinstance(scale, float):
        scale = int(scale)
    w += w * scale
    h += h * scale

    bbox = im.copy()
    bbox = cv2.resize(bbox, (w, h), interpolation=cv2.INTER_AREA)  # rescaling
    bbox = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)

    config = r"-l eng --oem 3 --psm 3"
    text = pt.image_to_data(bbox, output_type=Output.DICT, config=config)

    bbox = cv2.cvtColor(bbox, cv2.COLOR_GRAY2BGR)

    n_boxes = len(text["level"])
    for i in range(n_boxes):
        if text["text"][i] != "":
            (x, y, w, h) = (
                text["left"][i],
                text["top"][i],
                text["width"][i],
                text["height"][i],
            )
            cv2.rectangle(bbox, (x, y), (x + w, y + h), (255, 0, 0), 4)

    return text, bbox


def preprocessing(X, y=None, max_words_c=200):
    with open("./models/02_tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))

    X_ = []
    for i, sentence in enumerate(X):
        tmp_sentence = sentence.lower()
        tmp_sentence = tmp_sentence.replace("\n", "")
        tmp_sentence = tmp_sentence.translate(translator)
        X_.append(tmp_sentence)
    X = X_.copy()

    X_encoded = X.copy()
    X_encoded = map(lambda x: re.sub(r" +", " ", x), X_encoded)

    X_c = tokenizer.texts_to_sequences(X_encoded)
    X_c = sequence.pad_sequences(X_c, maxlen=max_words_c, padding="post")
    return X_c


from gensim.models import KeyedVectors


def vectorization(df):
    glove_model = KeyedVectors.load("./models/model.glove")
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))

    X = np.zeros((len(df), 200, 300))
    n = 0

    for sentence in df:
        tmp_sentence = sentence.lower().translate(translator)
        sentence = tmp_sentence.replace("\n", "")
        tokens = sentence.split()
        vecs = np.zeros((200, 300))
        m = 0
        for word in tokens:
            try:
                vec = glove_model.get_vector(word)
                vecs[m] = vec
                m += 1
            except KeyError:
                pass
        if len(vecs) > 0:
            X[n] = vecs
        n += 1

    return X


def preprocessing_ident_tokenization(X, y=None, max_words_c=200, max_words_w=100):
    with open("./models/tokenizer_ident.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

    str_to_replace = ',;:!?./§&~"#([-|`_\\^@)]=}²<>%$£¤*+'

    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    translator2 = str.maketrans(str_to_replace, " " * len(str_to_replace))

    X_ = []
    for i, sentence in enumerate(X):
        tmp_sentence = sentence.lower()
        tmp_sentence = tmp_sentence.replace("\n", "")
        tmp_sentence = tmp_sentence.translate(translator)
        tmp_sentence = tmp_sentence.translate(translator2)

        tmp_sentence = re.sub(r"\d{6,}", "$", tmp_sentence)
        tmp_sentence = re.sub(r"\d{4,6}", "####", tmp_sentence)
        tmp_sentence = re.sub(r"\d{3,4}", "###", tmp_sentence)
        tmp_sentence = re.sub(r"\d{2,3}", "##", tmp_sentence)
        tmp_sentence = re.sub(r"\d", "#", tmp_sentence)
        tmp_sentence = re.sub(r"(\b\S+\b)", r"@\1", tmp_sentence)
        tmp_sentence = re.sub(r" ", "", tmp_sentence)
        X_.append(tmp_sentence)
    X = X_.copy()

    X_clvl = tokenizer.texts_to_sequences(X)
    X_clvl = sequence.pad_sequences(X_clvl, maxlen=max_words_c, padding="post")

    return X_clvl


# with tab1:

#     # with st.sidebar:
#     #     st.text("adr clf params")

#     model_label_dpd = torch.hub.load(
#         # "yolov5/", "custom", path="./models/00_label_detection.pt", source="local"
#         "yolov5/", "custom", path="./models/best_dpd_5.pt", source="local"
#     ).to(
#         torch.device("cuda")
#     )

#     model_label_dpd.conf = .5
#     model_label_dpd.iou = .5

#         # model_label_dpd.conf = st.slider(
#         #     label="Confidence Threshold:",
#         #     min_value=0.0,
#         #     max_value=1.0,
#         #     value=0.5,
#         #     step=0.05,
#         # )
#         # model_label_dpd.iou = st.slider(
#         #     label="Overlap Threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.05
#         # )

#     cols = st.columns(2)
#     cols[0].image(im, caption="Uploaded image",
#                   use_column_width=None, width=500)

#     im_with_bboxes = model_label_dpd(im.copy())

#     cols[1].image(
#         im_with_bboxes.render(), caption="Predicted image", use_column_width=None, width=500
#     )

#     if os.path.exists("runs/"):
#         shutil.rmtree("runs/")

#     # Get bounding boxes
#     pred = model_label_dpd(im.copy()).crop(save=False)

#     # st.write(pred)

#     boxes = [np.array(pred[i]["im"]) for i in range(len(pred))]
#     labels = [pred[i]["label"] for i in range(len(pred))]

#     st.write("### Count occurrences of ADR labels.")

#     x = [f.split(" ")[0] for f in labels]
#     cc = Counter(x)
#     for i, j in cc.items():
#         y  = "times" if j > 1 else "time"
#         st.write(f"Detected **{i}** {j} {y}.")


#     # rows = st.columns(len(bboxes))

#     for id, b in enumerate(labels):
#         cols = st.columns(2)
#         st.text("")
#         st.text("")
#         if id == 0:
#             cols[0].write("### Detected ADRs")
#             cols[1].write("### Label")

#         cols[0].image(boxes[id])
#         cols[1].write(b)
#         # cols[1].image(all_boxes_new[id])
#         # cols[2].write(all_decoded_addr[id])

#         # st.write(X)
#         # st.write(type(X))

#         # pred = model_clf.predict(X)
#         # title = np.array(["Sender", "Receiver"])
#         # cols[3].write(np.vstack((title, out[id][:])).T)

# from sys import exit; exit()

# with tab2:
# # filename = st.file_uploader("Select an image to upload.", type=["png", "jpg", "jpeg"])
# url = st.text_input(
#     "Enter a HTTP URL to an image",
#     "https://a2btracking.com/wp-content/uploads/2017/07/MSL_RFID-1-e1499715334881.jpg",
# )

# # im = np.array(Image.open(filename))
# im = np.array(Image.open(BytesIO(requests.get(url).content)).convert("L"))

model_label_detection = torch.hub.load(
    # "yolov5/", "custom", path="./models/00_label_detection.pt", source="local"
    "yolov5/",
    "custom",
    path="./models/bestbest.pt",
    source="local",
    force_reload=True,
).to(torch.device("cuda"))


with st.sidebar:
    model_label_detection.conf = st.slider(
        label="Confidence Threshold:",
        min_value=0.5,
        max_value=1.0,
        value=0.1,
        step=0.05,
    )
    model_label_detection.iou = st.slider(
        label="Overlap Threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )


# model_label_detection.conf = .5
# model_label_detection.iou = .5

cols = st.columns(2)
cols[0].image(im, caption="Uploaded image", use_column_width=None, width=500)


im_with_bboxes = model_label_detection(im.copy())

cols[1].image(
    im_with_bboxes.render(), caption="Predicted image", use_column_width=None, width=500
)

if os.path.exists("runs/"):
    shutil.rmtree("runs/")

# Get bounding boxes
bboxes = model_label_detection(im.copy()).crop(save=False)
bboxes = [np.array(bboxes[i]["im"]) for i in range(len(bboxes))]

# rows = st.columns(len(bboxes))

all_texts, all_boxes = [], []
for id, b in enumerate(bboxes):
    text, box = OCR(b, scale=1)

    text = " ".join(" ".join(text["text"]).splitlines()).lower()
    # st.write(text)

    all_texts.append(text)
    all_boxes.append(box)

with open("./models/tokenizer_addr_identification.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# vectorized_data = vectorization(all_texts)
tokenized_data = preprocessing_ident_tokenization(all_texts)
st.write(tokenized_data)
# out1 = model_ident.predict(vectorized_data)
out2 = model_ident_tokenization.predict(tokenized_data)

out = out2

out = np.array(out)

addr_idx = [idx for idx, _ in sorted(enumerate(out[:, 0]), key=lambda x: x[1])[-2:]]
# st.write(out)
# st.write(addr_idx)


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
# st.write(tokenized_data)
addr_a = tokenized_data[0].reshape(1, -1)
addr_b = tokenized_data[1].reshape(1, -1)

# st.write(addr_a)
out = model_clf.predict({"input_addr_1": addr_a, "input_addr_2": addr_b})
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

    # st.write(X)
    # st.write(type(X))

    # pred = model_clf.predict(X)
    title = np.array(["Sender", "Receiver"])
    cols[3].write(np.vstack((title, out[id][:])).T)
