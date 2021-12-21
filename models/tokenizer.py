from tokenizers import BertWordPieceTokenizer
import os

sentences_path = "data\sentences_seperated_by_lines_format.txt"
word_list_path = "data\words_list.txt"

with open(word_list_path) as file:
    vocabulary = file.read().split();
print("Vocab size: ", len(vocabulary))


def generate_tokenizer():
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False
    )
    tokenizer.train(files=sentences_path, vocab_size=len(vocabulary), min_frequency=2, wordpieces_prefix='##',
                    limit_alphabet=1000, special_tokens=['[PAD', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
    os.mkdir('./bert_tokens_1')
    tokenizer.save_model('./bert_tokens_1', 'bert_tokens')

generate_tokenizer()