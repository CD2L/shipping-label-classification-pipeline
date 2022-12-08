## IMPORT ################
import re
import string
import pickle
import os
import json
from typing import Any
import time
import numpy as np

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.engine import DefaultPredictor

import torch

import cv2

from pytesseract import pytesseract as pt
from pytesseract import Output

from silence_tensorflow import silence_tensorflow

from gensim.models import KeyedVectors

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from nltk.translate.bleu_score import sentence_bleu
from PIL import Image

from .utils_.functions import crop, rotate_image, get_bboxes_from, plot_confusion_matrix

import easyocr

silence_tensorflow()
## CONST #################
CLASSES = ['sender', 'receiver']

## MODELS ###############
class AbstractModel():
    '''AbstractModel'''
    def __init__(self) -> None:
        pass

    def _img2array(self,img):
        im = img
        if isinstance(img, str):
            im = cv2.imread(img)
        return im

    def test(self):
        '''test abstract method for testing the performance of the model'''
        raise NotImplementedError("Subclass must implement abstract method")
    def predict(self):
        '''predict abstract method for predicting a value'''
        raise NotImplementedError("Subclass must implement abstract method")

class LabelLocalizationModel(AbstractModel):
    '''Label Localization Model'''
    cfg = None
    model = None

    def __init__(self,model_weight_path,cfg_path, mask=False) -> None:
        with open(cfg_path, 'rb') as f:
            self.cfg = pickle.load(f)
        self.cfg.MODEL.DEVICE = 'cuda'
        self.cfg.MODEL.WEIGHTS = model_weight_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.MASK_ON = mask

        self.model = DefaultPredictor(self.cfg)

        super().__init__()

    def _save_result(self, img, name):
        img = self._img2array(img)

        out = self.model(img)

        v = Visualizer(img[:,:,::-1], metadata={}, scale=.5, instance_mode=ColorMode.SEGMENTATION)
        v = v.draw_instance_predictions(out["instances"].to("cpu"))

        plt.figure(figsize=(14,10))
        plt.imsave(name,v.get_image())

        return out

    def predict(self, img: Any, save_result: bool = False, file_name = 'test_ll.png'):
        img = self._img2array(img)
        if save_result:
            return self._save_result(img, file_name)
        return self.model(img)

class ObjectDetectionModel(AbstractModel):
    '''Object detection model'''
    model = None
    def __init__(self, model_path, yolo_path = 'yolov5/', conf=0.5, iou=0.5, augment=0.5) -> None:
        self.model = torch.hub.load(yolo_path, "custom", path=model_path, source="local", device="cuda:1", verbose=False)
        self.model.conf = conf
        self.model.iou = iou
        self.model.augment = augment

    def predict(self, img):
        return self.model(self._img2array(img.copy()))

class AddrIdentificationModel(AbstractModel):
    '''Address identification model'''
    model_tokenization = None
    model_vectorization = None
    tokenizer = None
    vectorizer = None

    def __init__(self, token_model_path = "", vector_model_path = "", tokenizer_path = None, vectorizer_path = None) -> None:
        if len(token_model_path) > 0:
            self.model_tokenization = load_model(token_model_path)
        if len(vector_model_path) > 0:
            self.model_vectorization = load_model(vector_model_path)

        if tokenizer_path is not None:
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f, encoding='utf-8')

        if vectorizer_path is not None:
            self.vectorizer = KeyedVectors.load(vectorizer_path,)

    def test(self, test_file_path, sep=';', n_iter=None, mode='mean', confusion_matrix=True, verbose=False):
        df = pd.read_csv(test_file_path, sep, encoding='utf-8')
        df.label = df.label.map({'address':0,'code':1,'contact':2,'other':3})

        if n_iter is None:
            n_iter = len(df.label)

        if n_iter > len(df.label):
            raise Exception(f"n_iter maximum for this file is {len(df.label)} ({n_iter} provided)")

        if verbose:
            print(f'Testing with {n_iter} lines...\n')

        X, y = np.array(df.text), np.array(df.label)
        X, y = X[:n_iter], y[:n_iter]

        start = time.time()
        if mode == 'vector':
            out = self.predict_vector(X)
        elif mode == 'token':
            out = self.predict_token(X)
        else:
            out = self.predict(X)
        end = time.time()

        out = (out == np.amax(out,axis=1).reshape(-1,1)).argmax(axis=1)

        y = to_categorical(y, dtype='int').argmax(axis=1)

        acc = accuracy_score(y, out)


        if confusion_matrix:
            plot_confusion_matrix(out, y, ['address','code','contact','other'])

        if verbose:
            print(acc)

        return acc, (end-start)*1000, iter


    def tokenization(self, X, max_words_c=200):
        if self.tokenizer is None:
            raise Exception('self.tokenizer is not defined')
        
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

        X_clvl = self.tokenizer.texts_to_sequences(X)
        X_clvl = sequence.pad_sequences(X_clvl, maxlen=max_words_c, padding='post')

        return X_clvl

    def vectorization(self, df):
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
                    vec = self.vectorizer.get_vector(word)
                    vecs[m] = vec
                    m += 1
                except KeyError:
                    pass
            if len(vecs) > 0:
                X[n] = vecs
            n += 1

        return X

    def predict_vector(self, df):
        if self.model_vectorization is None:
            raise Exception("self.vector_model_path is not defined")

        vectorized_data = self.vectorization(df)
        
        out_vectorization = self.model_vectorization.predict(vectorized_data, verbose=False)
        
        return out_vectorization

    def predict_token(self, df):
        if self.model_tokenization is None:
            raise Exception("self.token_model_path is not defined")

        tokenized_data = self.tokenization(df)

        out_tokenization = self.model_tokenization.predict(tokenized_data, verbose=False)

        return out_tokenization

    def predict(self ,df):
        out_vect = self.predict_vector(df)
        out_token = self.predict_token(df)
        
        return (out_vect+out_token)/2

class AddrClassificationModel(AbstractModel):
    '''Address classification model'''
    model = None
    tokenizer = None

    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        self.model = load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def tokenization(self, X):
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

        X_ = []
        for i, sentence in enumerate(X):
            tmp_sentence = sentence.lower()
            tmp_sentence = tmp_sentence.replace('\n', '')
            tmp_sentence = tmp_sentence.translate(translator)

            tmp_sentence = re.sub(r' +', ' ', tmp_sentence)
            X_.append(tmp_sentence)
        X = X_.copy()

        x_encoded =  X.copy()
        x_encoded = map(lambda x: re.sub(r" +", " " , x), x_encoded)

        x_c = self.tokenizer.texts_to_sequences(x_encoded)
        x_c = sequence.pad_sequences(x_c, maxlen=200, padding='post')
        return x_c

    def test(self, test_file_path, sep=';', verbose=False):
        df = pd.read_csv(test_file_path, sep, encoding='utf-8')

        X = np.array(df.drop(['width', 'height'], 1))
        X = X.reshape(-1,2,2)

        acc_a = 0
        acc_b = 0

        start = time.time()

        for idx, item in enumerate(X):
            label_a, addr_a = item[0]
            label_b, addr_b = item[1]
            out = self.predict([addr_a],[addr_b])

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

        end = time.time()
        return acc_a, acc_b, (end-start)*1000, int(len(X)/2)


    def predict(self, addr_a: np.ndarray, addr_b: np.ndarray) -> list:
        tokenized_addr_a = self.tokenization(addr_a)
        tokenized_addr_b = self.tokenization(addr_b)
        return self.model.predict({'input_addr_1': tokenized_addr_a, 'input_addr_2': tokenized_addr_b})

class OCR(AbstractModel):
    '''OCR model (tesseract)'''
    def __init__(self) -> None:
        super().__init__()

    def test(self, json_file, verbose=False):
        with open(json_file, 'rb') as fp:
            dict = json.loads(fp.read())
            files = os.scandir(dict['images_path'])

            bleu_scores = []
            start = time.time()
            for idx, file in enumerate(files):
                if file.name.endswith((".png", ".jpeg", ".jpg")):
                    references_raw = dict['images'][file.name]
                    references = []
                    for ref in references_raw:
                        ref = re.sub(rf'[{string.punctuation}]', ' ', ref)
                        ref = re.sub(r' +', ' ', ref)
                        references.append(ref.split())

                    img = np.array(Image.open(file.path))
                    res = self.predict(img)

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
            end = time.time()
            bleu_scores = np.array(bleu_scores)
            if verbose:
                print(np.mean(bleu_scores))

            return np.mean(bleu_scores), (end-start)*1000, len(dict['images'])


    def preprocessing(self, im):
        bbox = im.copy()
        bbox = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)
        return bbox

    def predict(self, image, config="-l eng --oem 3 --psm 3"):
        return pt.image_to_data(image, output_type=Output.DICT, config=config)

class easyOCR(AbstractModel):
    def __init__(self, lang_list=['en'], gpu=True) -> None:
        self.model = easyocr.Reader(lang_list, gpu)
        
    def test(self, json_file, verbose=False):
        with open(json_file, 'rb') as fp:
            dict = json.loads(fp.read())
            files = os.scandir(dict['images_path'])

            bleu_scores = []
            start = time.time()
            for idx, file in enumerate(files):
                if file.name.endswith((".png", ".jpeg", ".jpg")):
                    references_raw = dict['images'][file.name]
                    references = []
                    for ref in references_raw:
                        ref = re.sub(rf'[{string.punctuation}]', ' ', ref)
                        ref = re.sub(r' +', ' ', ref)
                        references.append(ref.split())

                    img = np.array(Image.open(file.path))
                    res = self.predict(
                        image=img, 
                        low_text=0.5,
                        threshold=0.5,
                        min_size=5,
                        mag_ratio=3,
                        paragraph=True,
                        detail=1,
                        bbox_min_size=1,
                        contrast_ths=0.3,
                        adjust_contrast=0.5,
                        rotation_info=[180]
                    )

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
            end = time.time()
            bleu_scores = np.array(bleu_scores)
            if verbose:
                print(np.mean(bleu_scores))

            return np.mean(bleu_scores), (end-start)*1000, len(dict['images'])

    def predict(self, **kwargs):
        recognized_txts = self.model.readtext(**kwargs)
        txts = []
        for i in range(len(recognized_txts)):
            txts.append(recognized_txts[i][1])
            
        out = " ".join(np.asarray(txts))
        return out
        
    
def all(img_path):
    ll_model = LabelLocalizationModel("./models/label_localization.pth","./models/OD_cfg.pickle")
    yolo = ObjectDetectionModel("./models/best-yolov5x6.pt",0.2)
    ocr_model = OCR()
    idt_model = AddrIdentificationModel(
        token_model_path="./models/00_addr_identification_token.h5",
        tokenizer_path="./models/tokenizer_ident.pickle",
        vector_model_path="./models/glove_embedding_identification.h5",
        vectorizer_path="./models/model.glove"
        )
    clf_model = AddrClassificationModel("./models/00_addr_clf.h5","./models/tokenizer_clf.pickle")

    all_txt = []

    #REDRESSEMENT HERE !!!!!

    res = ll_model.predict(img_path)
    bboxes = get_bboxes_from(res, [0])
    cropped_bboxes = [crop(bbox, img_path) for bbox in bboxes]

    image = rotate_image(cropped_bboxes[0],-70)

    yolo_bboxes = yolo.predict(image)

    for od_bbox in yolo_bboxes.crop(save=False, ):
        im = od_bbox['im']

        text = ocr_model.predict(im)
        text = " ".join(" ".join(text["text"]).splitlines()).lower()
        text = re.sub(r' +',' ', text)

        if len(text) > 5:
            all_txt.append(text)

    out = idt_model.predict_vector(all_txt)
    out = np.array(out)

    addr_idx = [ idx for idx, _ in sorted(enumerate(out[:,0]), key=lambda x: x[1])[-2:]]

    addr_lst = []
    for i in addr_idx:
        addr_lst.append(all_txt[i])
    out_lst = clf_model.predict([addr_lst[0]], [addr_lst[1]])

    return addr_lst, out_lst