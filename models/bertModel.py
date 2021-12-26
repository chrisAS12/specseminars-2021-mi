from datasets import *
from transformers import *
from tokenizers import *
import os
import torch
import json
import tqdm
import os

sentences_path = "data/sentences_seperated_by_lines_format.txt"
word_list_path = "data\words_list.txt"
pretrained_path = 'C:\\Users\\chris\\Desktop\\specseminars-2021-mi\\tokenizer_1'
dataset_path = 'dataset_new_1.pt'

class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return self.encodings['input_ids'].shape[0]