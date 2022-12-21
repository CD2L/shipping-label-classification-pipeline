from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)
import matplotlib.pyplot as plt
from tqdm import tqdm

OUTPUT_DIR_PATH = "blurred_img_clf/dataset/"
NB_IMG_FOR_EACH_CLASS = 1000

shipper_words = [
    "sender",
    "send",
    "send from",
    "shipper",
    "ship from",
    "sender account",
    "from"
]

receiver_words = [
    "ship to",
    "receiver",
    "delivery",
    "delivery address",
    "client",
    "send to"
]

random_generator = GeneratorFromDict(
    count=NB_IMG_FOR_EACH_CLASS,
    blur=2,
    skewing_angle=12,
    random_blur=True,
    random_skew=True
)

shipper_generator = GeneratorFromStrings(
    shipper_words,
    count=NB_IMG_FOR_EACH_CLASS,
    blur=2,
    skewing_angle=12,
    random_blur=True,
    random_skew=True
)

receiver_generator = GeneratorFromStrings(
    receiver_words,
    count=NB_IMG_FOR_EACH_CLASS,
    blur=2,
    skewing_angle=12,
    random_blur=True,
    random_skew=True
)


for i in tqdm(range(NB_IMG_FOR_EACH_CLASS), desc="Generating images..."):
    rand_img, _ = next(random_generator)
    shipper_img, _ = next(shipper_generator)
    receiver_img, _ = next(receiver_generator)

    rand_img.save(OUTPUT_DIR_PATH + f'other/other_{i}.jpg')
    shipper_img.save(OUTPUT_DIR_PATH + f'shipper/shipper_{i}.jpg')
    receiver_img.save(OUTPUT_DIR_PATH + f'receiver/receiver_{i}.jpg')


