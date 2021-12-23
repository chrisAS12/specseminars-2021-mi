from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer

import os

sentences_path = "data\sentences_seperated_by_lines_format.txt"
word_list_path = "data\words_list.txt"

with open(word_list_path) as file:
    vocabulary = file.read().split()
print("Vocab size: ", len(vocabulary))


def generate_tokenizer_BertWordPieceTokenizer():
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False
    )
    tokenizer.train(files=sentences_path, vocab_size=len(vocabulary), min_frequency=2, wordpieces_prefix='##',
                    limit_alphabet=1000, special_tokens=['[PAD', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
    return tokenizer


def generate_tokenizer_ByteLevelBPETokenizer():
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=sentences_path, vocab_size=len(vocabulary), min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    #'[PAD', '[UNK]', '[CLS]', '[SEP]', '[MASK]'
    #
    # "<s>", "<pad>", "</s>", "<unk>", "<mask>",
    return tokenizer


def main():
    tokenizer = generate_tokenizer_ByteLevelBPETokenizer()
    os.mkdir('./bert_byte_0')
    tokenizer.save_model('./bert_byte_0', 'bert_tokens')

#main()