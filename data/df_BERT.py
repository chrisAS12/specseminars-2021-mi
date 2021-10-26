# Data formatting for the required BERT specifics

from df_GEN import get_text, save_text
import re

def create_vocabulary(file):
    text = get_text(file)
    #text_lower_case = [s.lower() for s in text]
    unwanted_stuff = "1 2 3 4 5 6 7 8 9 0 , ; : . ! ?".strip(" ")
    sentences = []
    i = 0
    for t in text:
        sentence = (re.sub("[.,!?\\-]():1234567890", '', t.lower()))
        if any(unwanted in sentence for unwanted in unwanted_stuff):
            print("failed algorithm " , str(i))
            i += 1
        else:
            sentences.append(sentence)
    return list(set("".join(sentences).split()))

files = ["data\LVK2013_FORMATTED.txt" , "data\TRAIN_SET_FORMATTED.txt"]

word_list = []
for i in files:
    word_list.append(create_vocabulary(i))

save_text(word_list, "word_list_v2.txt")