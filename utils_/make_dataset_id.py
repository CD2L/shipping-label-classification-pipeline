from ast import keyword
from datetime import datetime
import random
from faker import Faker

faker = Faker(["en_US", "en_GB"])

bruit = '@/!:|\&")°ç^-([]#~%?,<>abcdefghijklmnopqrstuvwxyz'


def to_integer(dt_time):
    return 10000 * dt_time.year + 100 * dt_time.month + dt_time.day


def fake_contact():
    keywords = ["contact", "tel", "phone", "mobile", "info"]
    random.shuffle(keywords)
    rand_bruit = random.random()

    infos = [faker.name(), faker.phone_number(), faker.text(max_nb_chars=20)]

    lst = []
    for i in keywords[:2]:
        lst.append(i)

    for i in infos:
        k = random.randint(0, len(lst) - 1)
        lst.insert(k, i)
    if rand_bruit < 0.05:
        return add_bruit(" ".join(lst))
    else:
        return " ".join(lst)


def fake_code():
    abc = "abcdefghijklmopqrstuvwxyz"

    rand = random.random()

    numbers = faker.ean(length=8)

    abc_arr = [*abc]
    random.shuffle(abc_arr)

    if rand <= 0.25:
        a, b = numbers[:3], numbers[-3:]
        code_letters = "".join(abc_arr[0:3])
        fakecode = "".join([a, code_letters, b])
    elif rand <= 0.50:
        fakecode = "".join([numbers[:2], numbers])
    elif rand <= 0.75:
        fakecode = numbers
    else:
        fakecode = faker.ean(length=13)

    return fakecode


def fake_other():
    length = random.randint(1, 10)
    rand = random.random()
    rand_bruit = random.random()

    if rand <= 0.2:
        sentence = faker.sentence(nb_words=length + 5)
    elif rand <= 0.4:
        sentence = faker.text(max_nb_chars=length + 5)
    elif rand <= 0.6:
        units = ["kg", "lb", "g", "l", "m"]
        is_unit = random.random() > 0.8
        rand_unit = random.randint(0, len(units) - 1)

        sentence = faker.word() + " " + str(random.randint(1, 9999))

        if is_unit:
            sentence = sentence + " " + units[rand_unit]

    elif rand <= 0.8:
        nb_word = random.randint(1, 3)
        date_format = [
            "%Y-%m-%d",
            "%y-%m-%d",
            "%y-%b-%d",
            "%d/%m/%Y",
            "%d/%m/%y",
            "%d %b %y",
            "%d %b %Y",
            "%d %B %y",
            "%d %B %Y",
        ]
        rand_format = random.randint(0, len(date_format) - 1)
        sentence = (
            " ".join(faker.words(nb_word))
            + " "
            + faker.date_this_century().strftime(date_format[rand_format])
        )
    else:
        sentence = " ".join(faker.words())

    if rand_bruit < 0.05:
        sentence = add_bruit(sentence)

    return sentence


def add_bruit(sentence):
    sentence_c = [*sentence]
    nb_bruit = random.randint(1, 3)
    for _ in range(nb_bruit):
        rand_idx = random.randint(0, len(bruit) - 1)
        rand_pos = random.randint(0, len(sentence_c) - 1)
        sentence_c[rand_pos] = bruit[rand_idx]
    sentence_c = "".join(sentence_c)

    return sentence_c


with open("./street_name.txt", "r", encoding="UTF-8") as f:
    output_f = open("./generated.csv", "w", encoding="UTF-8")
    output_f.write(f"label;text\n")
    for i in iter(f.readlines()):
        addr = i.strip()
        rand_bruit = random.random()
        if rand_bruit < 0.05:
            addr = add_bruit(addr)

        output_f.write(f"address;{addr}\n")
        output_f.write(f"contact;{fake_contact()}\n")
        output_f.write(f"code;{fake_code()}\n")
        output_f.write(f"other;{fake_other()}\n")
