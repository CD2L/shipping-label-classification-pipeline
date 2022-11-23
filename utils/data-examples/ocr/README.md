# data-examples OCR
This folder exists to test OCR performances. 

## How is this works?
Put your text blocks images here, they can be in png, jpeg or jpg format.
In *references.json*, create a json with the name of the images as keys and their true sentence as values in arrays.

```json
{
    "block1.png": ["sentence that is written in the image"],
    "block2.jpg": ["another sentence"],
    "blockname.jpeg": [
        "you can put multiple sentence",
        "if you think that is useful"
    ],
}
```
Those sentences are called references and will be compared to the result of the OCR using [BLEU](https://en.wikipedia.org/wiki/BLEU). The test function is present in */pipeline.py*. 

