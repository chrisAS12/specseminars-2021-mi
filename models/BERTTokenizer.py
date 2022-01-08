from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
import os
import json

sentences_path = "data\sentences_seperated_by_lines_format.txt"
word_list_path = "data\words_list.txt"

with open(word_list_path) as file:
   vocabulary = file.read().split()
print("Vocab size: ", len(vocabulary))


model_path = "tokenizer_30522_default_vocab_truncation_1024"
max_length = 1024

def generate_tokenizer_BertWordPieceTokenizer():
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False
    )
    tokenizer.train(files=sentences_path, vocab_size=30522, min_frequency=2, wordpieces_prefix='##',
                    limit_alphabet=1000, special_tokens=['[PAD', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
    tokenizer.enable_truncation(max_length=max_length)
    return tokenizer


def generate_tokenizer_ByteLevelBPETokenizer():
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=sentences_path, vocab_size=30522, add_special_tokens=True, truncation=True,
                min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
    ])
    return tokenizer


def save_tokenizer(tokenizer):
    if not os.path.isdir(model_path):
      os.mkdir(model_path)
    tokenizer.save_model(model_path)

    with open(os.path.join(model_path, "config.json"), "w") as f:
        tokenizer_cfg = {
            "do_lower_case": True,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "model_max_length": max_length,
            "max_len": max_length,
        }
        json.dump(tokenizer_cfg, f)


def generate_new_tokenizer():
    tokenizer = generate_tokenizer_BertWordPieceTokenizer()
    save_tokenizer(tokenizer)
#generate_new_tokenizer()