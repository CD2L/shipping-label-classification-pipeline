from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)
import matplotlib.pyplot as plt
from tqdm import tqdm

OUTPUT_DIR_PATH = "./dataset/"
NB_IMG_FOR_EACH_CLASS = 2000

shipper_words = [
    "sender",
    "send",
    "send from",
    "shipper",
    "ship from",
    "sender account",
    "sender account 56412",
    "sender account 45998",
    "sender account 13251",
    "sender account 04542",
    "sender account 73324",
    "sender account 84249",
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
    size=36,
    skewing_angle=7,
    random_blur=True,
    random_skew=True,
    allow_variable=True
)

shipper_generator = GeneratorFromStrings(
    shipper_words,
    count=NB_IMG_FOR_EACH_CLASS,
    blur=2,
    size=36,
    skewing_angle=7,
    random_blur=True,
    random_skew=True
)

receiver_generator = GeneratorFromStrings(
    receiver_words,
    count=NB_IMG_FOR_EACH_CLASS,
    blur=2,
    skewing_angle=7,
    random_blur=True,
    random_skew=True,
    size=36
)


for i in tqdm(range(NB_IMG_FOR_EACH_CLASS), desc="Generating images..."):
    rand_img, _ = next(random_generator)
    shipper_img, _ = next(shipper_generator)
    receiver_img, _ = next(receiver_generator)

    rand_img.save(OUTPUT_DIR_PATH + f'other/other_d_{i}.jpg')
    shipper_img.save(OUTPUT_DIR_PATH + f'shipper/shipper_d_{i}.jpg')
    receiver_img.save(OUTPUT_DIR_PATH + f'receiver/receiver_d_{i}.jpg')


