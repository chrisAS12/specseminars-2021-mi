# BERT (Bidirectional Encoder Representation with Transformers)
# Created 26/10/2021
# Author: Krišjānis Mārtiņš Alliks ( chrisAS12 )

# [CLS], [MASK], [SEP]  - TOKENS 

FILE_LOCATION = 'data\word_list_v6.txt'

def get_text(FILE_LOCATION):
    f = open(FILE_LOCATION, encoding="utf-8")
    text = f.read().split(" ")
    return text

word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
word_list = get_text(FILE_LOCATION)

for i,w in enumerate(word_list):
    word_dict[w] = i+4
    #number_dict = {i: w for i, w in enumerate(word_dict)}
    #vocab_size = len(word_dict)

print(word_dict)