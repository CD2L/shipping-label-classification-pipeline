# Shipping Label Classification Pipeline

This project aims to use deep learning algorithms to detect whether a product is counterfeit by analyzing images of boxes and their labels. This repository is a PoC Demo of how the pipeline actually works using web scraped images. Here is an overview of the pipeline:

![Pipeline](./files/pipeline.svg)

You can download all pre-trained models via the [link](https://drive.google.com/file/d/1cJOny4WFzIUuGLXxJlAnG0BSijZW-NMn/view).

## Dependencies

The pipeline requires a number of packages to run. You can install them in a virtual environment on your machine via the command :

```shell
cd ./shipping-label-classification-pipeline
git clone https://github.com/ultralytics/yolov5
pip install -r ./yolov5/requirements.txt
pip install -r ./requirements.txt
```

Please refer to this this [link](https://tesseract-ocr.github.io/tessdoc/Downloads.html) to download the latest version of Tesseract.

## Getting started

After cloning this repository, you can now run this command:

```shell
streamlit run ./demo.py
```

Once the service is up,  you can interact with its UI: [http://localhost:8501](http://localhost:8501).

## Demo

![Demo](./files/demo.png)
