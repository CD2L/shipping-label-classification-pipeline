from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.config import get_cfg
from detectron2 import model_zoo

import numpy as np
import random
import cv2
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    
    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img, metadata=dataset_custom_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()
        
def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()
    
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    
    cfg.DATALOADER.NUM_WORKERS = 3
    
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 50000
    cfg.SOLVER.STEPS = []
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    
    return cfg
    
def on_image(image_path, predictor, show=False):
    im=cv2.imread(image_path)
        
    lim = 255 - 50
    im[im > lim] = 255
    im[im <= lim] += 50
    
    outputs = predictor(im)
    
    if show:
        v = Visualizer(im[:,:,::-1], metadata={}, scale=.5, instance_mode=ColorMode.SEGMENTATION)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        plt.figure(figsize=(14,10))
        plt.imshow(v.get_image())
        plt.show()
    
    return outputs


def get_bboxes_from(outputs, classes = []):
    """ returns list of bboxes """

    pred_classes = outputs['instances'].pred_classes

    out = torch.isin(pred_classes, torch.tensor(classes, device='cuda:0'))

    idx_lst = [] 
    for idx, i in enumerate(out):
        if i:
            idx_lst.append(idx)

    return outputs["instances"].__dict__['_fields']['pred_boxes'].__dict__['tensor'][idx_lst]


def crop(bbox, in_img: str):
    img = np.array(cv2.imread(in_img, cv2.COLOR_RGB2GRAY))
    """ bbox is a list with xmin, ymin, xmax, ymax """
    xmin, ymin, xmax, ymax = bbox
    
    cropped_im = img[int(ymin):int(ymax),int(xmin):int(xmax)]
    return cropped_im

def plot_confusion_matrix(y_pred, y_true, classes):
    '''Plotting a confusion matrix'''
    cm = confusion_matrix(y_pred, y_true,normalize='true')
    _, ax = plt.subplots(figsize=(10,10))
    
    sns.heatmap(cm, annot=True, fmt='g')
    
    plt.xticks(rotation=90)
    ax.set_xlabel('Predicted')
    ax.xaxis.set_ticklabels(classes)
    plt.yticks(rotation=0)
    ax.set_ylabel('True')
    ax.yaxis.set_ticklabels(classes)
    
    plt.title('Confusion matrix')
    plt.show()

import cv2

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat