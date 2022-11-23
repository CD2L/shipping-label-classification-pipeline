## IMPORT ################
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import re

import torch

import cv2

from pytesseract import pytesseract as pt
from pytesseract import Output

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf  
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from detectron2.engine import DefaultPredictor

from gensim.models import KeyedVectors

import string
import pickle
import shutil
import os
import warnings
import json
from pathlib import Path

import timeit

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

from nltk.translate.bleu_score import sentence_bleu

from utils.functions import *

## CONST #################
CLASSES = ['sender', 'receiver']

image_url = ''  

## MODELS ###############
def plot_confusion_matrix(y_pred, y_true, classes):
    cm = confusion_matrix(y_pred, y_true,normalize='true')
    fig, ax = plt.subplots(figsize=(10,10))
    
    sns.heatmap(cm, annot=True, fmt='g')
    
    plt.xticks(rotation=90)
    ax.set_xlabel('Predicted')
    ax.xaxis.set_ticklabels(classes)
    plt.yticks(rotation=0)
    ax.set_ylabel('True')
    ax.yaxis.set_ticklabels(classes)
    
    plt.title('Confusion matrix')
    plt.show()

def label_localization(img_path, cfg_save_path = "./models/OD_cfg.pickle"):
    with open(cfg_save_path, 'rb') as f:
        cfg = pickle.load(f)
    cfg.MODEL.DEVICE = 'cuda:0'
    cfg.MODEL.WEIGHTS = os.path.join("./models/", "label_localization.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.MASK_ON = False
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    
    predictor = DefaultPredictor(cfg)

    return on_image(img_path, predictor)

def yolo(url):
    im = np.array(Image.open(BytesIO(requests.get(url).content)).convert("L"))
    
    model_label_detection = torch.hub.load("yolov5/", "custom", path="./best-yolov5x6.pt", source="local", device="cuda:1", verbose=False)
    model_label_detection.conf = .5
    model_label_detection.iou = .5
    model_label_detection.augment = .5
    
    im_with_bboxes = model_label_detection(im.copy())
    return im_with_bboxes

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
            (x, y, w, h) = (
                text['left'][i], 
                text['top'][i], 
                text['width'][i], 
                text['height'][i]
            )
            cv2.rectangle(bbox, (x, y), (x + w, y + h), (255, 0, 0), 4)

    return text, bbox

def addr_identification(texts, mode='mean'):
    if not len(texts) > 0:
        return -1 
    
    def preprocessing(X,y = None,max_words_c = 200):
        with open('./models/tokenizer_ident.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        str_to_replace = ',;:!?./§&~"#([-|`_\\^@)]=}²<>%$£¤*+'

        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        translator2 = str.maketrans(str_to_replace, ' '*len(str_to_replace)) 

        X_ = []
        for i, sentence in enumerate(X):
            tmp_sentence = sentence.lower()
            tmp_sentence = tmp_sentence.replace('\n', '')
            tmp_sentence = tmp_sentence.translate(translator)
            tmp_sentence = tmp_sentence.translate(translator2)
            
            tmp_sentence = re.sub(r"\d{6,}", "$" , tmp_sentence)
            tmp_sentence = re.sub(r"\d{4,6}", "####" , tmp_sentence)
            tmp_sentence = re.sub(r"\d{3,4}", "###" , tmp_sentence)
            tmp_sentence = re.sub(r"\d{2,3}", "##" , tmp_sentence)
            tmp_sentence = re.sub(r"\d", "#" , tmp_sentence)
            tmp_sentence = re.sub(r"(\b\S+\b)", r"@\1" , tmp_sentence)
            tmp_sentence = re.sub(r' ', '', tmp_sentence)
            X_.append(tmp_sentence)
        X = X_.copy()

        X_clvl = tokenizer.texts_to_sequences(X)
        X_clvl = sequence.pad_sequences(X_clvl, maxlen=max_words_c, padding='post')

        return X_clvl

        return X
    def vectorization(df):
        glove_model = KeyedVectors.load('./models/model.glove',)
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

        X = np.zeros((len(df), 200, 300))
        n = 0

        for sentence in df:
            tmp_sentence = sentence.lower().translate(translator)
            sentence = tmp_sentence.replace('\n', '')
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

    if mode == 'mean':
        model_tokenization = load_model("./models/00_addr_identification_token.h5")
        model_vectorization = load_model("./models/glove_embedding_identification.h5")
        
        tokenized_data = preprocessing(texts)
        vectorized_data = vectorization(texts)
        
        out_tokenization = model_tokenization.predict(tokenized_data, verbose=False)
        out_vectorization = model_vectorization.predict(vectorized_data, verbose=False)
        
        return (out_tokenization+out_vectorization)/2

    elif mode == 'vector':
        model_vectorization = load_model("./models/glove_embedding_identification.h5")
        
        vectorized_data = vectorization(texts)
        
        out_vectorization = model_vectorization.predict(vectorized_data, verbose=False)

        return out_vectorization
    else:
        model_tokenization = load_model("./models/00_addr_identification_token.h5")
        
        tokenized_data = preprocessing(texts)
        
        out_tokenization = model_tokenization.predict(tokenized_data, verbose=False)

        return out_tokenization

def addr_classification(texts):
    if not len(texts) > 0:
        return -1 

    model_clf = load_model("./models/00_addr_clf.h5")

    def preprocessing(X,y = None, max_words_c = 200):
        with open('./models/tokenizer_clf.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
            
        X_ = []
        for i, sentence in enumerate(X):
            tmp_sentence = sentence.lower()
            tmp_sentence = tmp_sentence.replace('\n', '')
            tmp_sentence = tmp_sentence.translate(translator)
            
            tmp_sentence = re.sub(r' +', ' ', tmp_sentence)
            X_.append(tmp_sentence)
        X = X_.copy()
        
        X_encoded =  X.copy()
        X_encoded = map(lambda x: re.sub(r" +", " " , x), X_encoded)

        X_c = tokenizer.texts_to_sequences(X_encoded)
        X_c = sequence.pad_sequences(X_c, maxlen=max_words_c, padding='post')
        return X_c

    tokenized_data = preprocessing(texts)
    
    addr_a = tokenized_data[0].reshape(1, -1)
    addr_b = tokenized_data[1].reshape(1, -1)
    
    out = model_clf.predict({'input_addr_1': addr_a, 'input_addr_2': addr_b})
    return out

def all(url):
    im_with_bboxes = yolo(url)

    bboxes = im_with_bboxes.crop(save=False, )
    bboxes = [np.array(bboxes[i]["im"]) for i in range(len(bboxes))]
    
    all_texts, all_boxes = [], []
    for id, b in enumerate(bboxes):
        text, box = OCR(b, scale=1)

        text = " ".join(" ".join(text["text"]).splitlines()).lower()

        all_texts.append(text)
        all_boxes.append(box) 
        
    out = addr_identification(all_texts, mode='token')
    out = np.array(out)

    addr_idx = [ idx for idx, _ in sorted(enumerate(out[:,0]), key=lambda x: x[1])[-2:]]

    all_texts_new, all_boxes_new, all_decoded_addr, all_bboxes_new = [], [], [], []
    for i in addr_idx:
        all_boxes_new.append(all_boxes[i])
        all_bboxes_new.append(bboxes[i])
        all_decoded_addr.append(all_texts[i])

    out = addr_classification(all_decoded_addr)

## TEST ###################
def addr_clf_test(test_file = './utils/generated_clf.csv', sep = ';', verbose=False):
    df = pd.read_csv(test_file, sep, encoding='utf-8')
    
    X = np.array(df.drop(['width', 'height'], 1))
    X = X.reshape(-1,2,2)
    
    acc_a = 0
    acc_b = 0
    
    for idx, item in enumerate(X):
        label_a, addr_a = item[0]
        label_b, addr_b = item[1]
        out = addr_classification([addr_a,addr_b])
                
        if label_a == 'sender_details':
            acc_a += (out[0][0][0] > 0.5)/X.shape[0]
        else:
            acc_a += (out[0][0][1] > 0.5)/X.shape[0]
        
        if label_b == 'sender_details':
            acc_b += (out[1][0][0] > 0.5)/X.shape[0]
        else:
            acc_b += (out[1][0][1] > 0.5)/X.shape[0]
            
        if verbose:
            print(idx,':', acc_a, '/', idx/X.shape[0])
    
    return acc_a, acc_b

def addr_ident_test(test_file = './utils/generated.csv', sep = ';', mode='mean', confusion_matrix=True, verbose=True, iter=None):
    df = pd.read_csv(test_file, sep, encoding='utf-8')
    df.label = df.label.map({'address':0,'code':1,'contact':2,'other':3})

    if iter is None:
        iter = len(df.label)

    if iter > len(df.label):
        raise Exception(f"Iter maximum for this file is {len(df.label)} ({iter} provided)")
    
    if verbose:
        print(f'Testing with {iter} lines...\n')
        
    X, y = np.array(df.text), np.array(df.label)
    X, y = X[:iter], y[:iter]
    
    out = addr_identification(X, mode=mode)
    
    out = (out == np.amax(out,axis=1).reshape(-1,1)).argmax(axis=1)
    y = to_categorical(y, dtype='int').argmax(axis=1)
    
    acc = accuracy_score(y, out)
    
    if confusion_matrix:
        plot_confusion_matrix(out, y, ['address','code','contact','other'])

    if verbose:
        print(acc)
        
    return acc

def ocr_test(test_dir = './utils/data-examples/ocr/', json_file = 'references.json', verbose=True):
    with open(os.path.join(test_dir, json_file), 'rb') as fp:
        dict = json.loads(fp.read())
        files = os.scandir(test_dir)
        
        bleu_scores = []
        for idx, file in enumerate(files):
            if file.name.endswith((".png", ".jpeg", ".jpg")):
                references_ = dict[file.name]
                references = []
                for ref in references_:
                    ref = re.sub(rf'[{string.punctuation}]', ' ', ref)
                    ref = re.sub(r' +', ' ', ref)
                    references.append(ref.split())
        
                img = np.array(Image.open(file.path))
                res = OCR(img)[0]
                
                sentence = ' '.join(res['text'])
                sentence = sentence.lower().strip()
                sentence = re.sub(rf'[{string.punctuation}]', ' ', sentence)
                sentence = re.sub(r' +', ' ', sentence)
                                
                bleu_score = sentence_bleu(references, sentence.split(), weights=(0.5, 0.3, 0.2, 0))
                bleu_scores.append(bleu_score)
                
                if verbose:
                    print(idx, ':', bleu_score)
                    print(' '.join(references[0]))
                    print(sentence)
                    print("--------------------")
        bleu_scores = np.array(bleu_scores)
        if verbose:
            print(np.mean(bleu_scores))
            
        return np.mean(bleu_scores)

## MAIN ###################
def main():
    res = label_localization("./utils/data-examples/label_localization/1.jpg")
    bboxes = get_bboxes_from(res, [0])
    for bbox in bboxes:
        res = crop(bbox, "./utils/data-examples/label_localization/1.jpg")
  
        plt.figure(figsize=(14,10))
        plt.imshow(res)
        plt.show()

    return 0

if __name__ == '__main__':
    main()