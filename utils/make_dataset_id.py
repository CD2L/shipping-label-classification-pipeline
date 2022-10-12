import random
from faker import Faker

faker = Faker(['cz_CZ', 'fr_FR','en_US', 'en_GB','it_IT','ja_JP','zh_CN', 'pt_PT', 'uk', 'no', 'pl'])

def fake_contact_name():
    with open('name.txt', 'r') as f:
        names = f.readlines()
        k = random.randint(0,len(names)-1)
        return names[k].strip()
    
def fake_contact():
    keywords = ['contact', 'tel', 'phone', 'mobile', 'info']
    random.shuffle(keywords)
    
    infos = [fake_contact_name(),faker.phone_number(),faker.text(max_nb_chars=20)]
    
    arr = []
    for i in keywords[:2]:
        arr.append(i)
    
    for i in infos:
        k = random.randint(0,len(arr)-1)
        arr.insert(k,i)
    
    return " ".join(arr) 

def fake_code():
    numbers = faker.ean(length=8)
    a,b = numbers[:3], numbers[-3:]
    abc = 'abcdefghijklmopqrstuvwxyz'
    abc_arr = [*abc]
    random.shuffle(abc_arr)
    abc = "".join(abc_arr[0:3])
    return "-".join([a,abc,b])

with open('street_name.txt', 'r') as f:
    output_f = open('generated.csv', 'w')
    output_f.write(f"label;text\n")
    for i in iter(f.readlines()):
        output_f.write(f"address;{i.strip()}\n")
        output_f.write(f"contact;{fake_contact()}\n")
        output_f.write(f"other;{fake_code()}\n")