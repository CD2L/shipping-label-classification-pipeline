'''Pipeline's demo'''
import os
import easyocr

import cv2
import argparse
import shutil
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from models_pipeline import *
from utils_.functions import on_image
from utils_.scanner import Extractor

app = Flask(__name__, template_folder='demo/views/', static_url_path='/demo/static')

app.config['UPLOAD_FOLDER'] = 'demo/upload/'
upload_folder = 'demo/upload/'

ll_model = LabelLocalizationModel("./weights/label_localization.pth","./weights/OD_cfg.pickle")
yolo = ObjectDetectionModel("./weights/best-yolov5x6.pt",0.2)
ocr_model = OCR()
idt_model = AddrIdentificationModel(
    token_model_path="./weights/00_addr_identification_token.h5",
    tokenizer_path="./weights/tokenizer_ident.pickle",
    vector_model_path="./weights/glove_embedding_identification.h5",
    vectorizer_path="./weights/model.glove"
    )
clf_model = AddrClassificationModel("./weights/00_addr_clf.h5","./weights/tokenizer_clf.pickle")
ext = Extractor()

@app.route('/')
def home():
    
    return render_template('link-form.html')

@app.route('/submit-form', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],'uploaded_file.jpg'))
        return redirect('/demo-processing')

@app.route('/demo-processing')
def demo_processing():
    
    if os.path.isdir('static'):
        shutil.rmtree('static')
    os.mkdir('static')
    
    img = cv2.imread('./demo/upload/uploaded_file.jpg')

    all_txt = []

    out_ll = ll_model.predict(img, file_name='./static/ll_result.jpg', save_result=True)

    bboxes = get_bboxes_from(out_ll, [0])
    cropped_bboxes = [crop(bbox, './demo/upload/uploaded_file.jpg', margin=50) for bbox in bboxes]
    
    img_lst = []
    os.mkdir('./static/ll')
    os.mkdir('./static/redress')
    for idx, img in enumerate(cropped_bboxes):
        cv2.imwrite(f'./static/ll/cropped{idx}.jpg', img)
        out = ext(img)[-1]
        print(out.shape)
        if out.shape[0] < out.shape[1]:
            out = rotate_image(out, -90)
        cv2.imwrite(f'./static/redress/{idx}.jpg',out)

        img_lst.append(out)

    yolo_bboxes = yolo.predict(img_lst)

    # yolo_bboxes.save(save_dir='static/yolo')
    
    ocr = easyocr.Reader(lang_list=["en","nl","fr"], gpu=True, recognizer=True)

    if not os.path.exists('./static/ocr'):
        os.mkdir('./static/ocr')
    

    # bboxes = yolo_bboxes.crop(save=False, )
    # bboxes = [np.array(bboxes[i]["im"]) for i in range(len(bboxes))]

    for idx, od_bbox in enumerate(yolo_bboxes.crop(save=True, save_dir='./static/yolo/')):
        im = od_bbox['im']
        if im.shape[0] > im.shape[1]:
            im = rotate_image(im, 90)
        
        # im = rotate_image(im, 90)
        recognized_txts = ocr.readtext(im, paragraph=False)

        texts = []
        for i in range(len(recognized_txts)):
            txt = recognized_txts[i][1]
            
            texts.append(txt)

        out = " ".join(np.asarray(texts))
        all_txt.append(out)
        
        cv2.imwrite(f'./static/ocr/{idx}.jpg',im)


    if len(all_txt) < 1:
        return "Aucune addresse n'a été détécté"

    print(all_txt)
    out = idt_model.predict(all_txt)
    out = np.array(out)

    addr_idx = [ idx for idx, _ in sorted(enumerate(out[:,0]), key=lambda x: x[1])[-2:]]

    addr_lst = []
    for i in addr_idx:
        addr_lst.append(all_txt[i])
    if len(addr_lst) > 1:
        out_lst = clf_model.predict([addr_lst[0]], [addr_lst[1]])
    else:
        out_lst = clf_model.predict([addr_lst[0]], [''])
        addr_lst.append('')
    
    out = list(zip(addr_lst, out_lst))
    yolo_dir = os.listdir('./static/ocr')
    ocr_dir = os.listdir('./static/ocr')  
    ll_dir = os.listdir('./static/ll')
    redress_dir = os.listdir('./static/redress')
    print(out)
    return render_template('demo.html', yolo_dir=yolo_dir, all_txt=all_txt,addr_lst=addr_lst, out=out,ocr_dir=ocr_dir, ll_dir=ll_dir, redress_dir=redress_dir)    
