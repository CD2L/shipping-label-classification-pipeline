from random import randint, randrange, choice, shuffle, random
import os

output_folder = './'


label_text = {
    "sender_details" : [
        "sender", 
        "from",
        "exp",
        "exp.",
        "Shipper",
        "Sent by",
        "Posted by", 
        "return", 
        "retourner", 
        "отправитель", 
        "发件人", 
        "nadawca",
        "forwarder",
        "Absender",
        "Afsender",
        "Siuntėjas",
        "Αποστολέας",
        "Подател",
        "Remitente",
        "Emisor",
        "Saatja",
        "Lähettäjä",
        "Expéditeur",
        "Expediteur",
        "Envoyé par",
        "Feladó",
        "Felado",
        "Pengirim",
        "Mittente",
        "送信者",
        "Sūtītājs",
        "Siuntėjas",
        "Afzender",
        "Remetente",
        "Expeditor",
        "Odosielateľ",
        "Pošiljatelj",
        "Avsändare",
        "Odesílatel",
        "Gönderen",
        "Відправник",
        ],
    "receiver_details" : [
        "ship",
        "to" ,
        "ship to",
        "destinataire",
        "delivery",
        "delivery address",
        "dest",
        "dest.",
        "доставить",
        "Доставлено в",
        "Enviado",
        "出荷先",
        "Teslim edilen",
        "Alıcı",
        "Destinatario",
        "Příjemce",
        "Ontvanger",
        "Receiver",
        "Приемник",
        "Empfänger",
        "Vevő",
        "Címzett",
        "Δέκτης",
        "Παραλήπτης",
        "Приймач",
        "Sprejemnik",
        "Prijímač",
        ],
    "unknown": [
        "",
    ]
}

def generate_blocs_size(): 
    size_sender_bloc = (randint(100,200),randint(10,60))
    size_receiver_bloc = (size_sender_bloc[0] + randint(10,50), size_sender_bloc[1] + randint(0,50))
    return size_sender_bloc, size_receiver_bloc

def random_line(fname):
    lines = open(fname,"r",encoding="utf-8").read().splitlines()
    return choice(lines)

def add_bruit(word, chance=.1):
    if len(word) < 1 : return word
    if random() > chance : return word
    
    bruit = 'abcdefghijklmnopqrstuvwxyz!:,?./§ù*%µ^$¨£=+)°à0ç_è-(\'"é&œ´~#{[|`\^@]}'
    rand_idx = randint(0,len(bruit)-1)
    char = bruit[rand_idx]
    
    word_lst = [*word]
    rand_pos = randint(0, len(word)-1)
    word_lst[rand_pos] = char
    
    return ''.join(word_lst)

def generate_text(type, keyword_presence = True):
    k = randint(0,len(label_text[type])-1)
    line = random_line("./utils/street_name.init")
    if keyword_presence:
        txt_arr = line.split()
        txt_arr.insert(randint(0,len(txt_arr)-1), add_bruit(label_text[type][k],0.12))
        shuffle(txt_arr)
        return " ".join(txt_arr)
    return line

    
if os.path.exists(output_folder+"generated_clf.csv"):
  os.remove(output_folder+"generated_clf.csv")

with open(output_folder+"generated_clf.csv", "x", encoding="utf-8") as f:
    f.write(f"label;text;width;height\n")
    for i in range(5000):
        size_s, size_r = generate_blocs_size()

        r = random()
        if r < 0.33: 
            txt_s = generate_text("sender_details",False)
            txt_r = generate_text("receiver_details")
        elif r < 0.66:
            txt_s = generate_text("sender_details")
            txt_r = generate_text("receiver_details", False)
        else:
            txt_s = generate_text("sender_details")
            txt_r = generate_text("receiver_details")
            
            
        lst = [
            f"sender_details;{txt_s};{size_s[0]};{size_s[1]}\n",
            f"receiver_details;{txt_r};{size_r[0]};{size_r[1]}\n",
        ]

        shuffle(lst)
        
        for i, str in enumerate(lst):
            f.write(str)
            
    f.close()