from tokenizers import BertWordPieceTokenizer

sentences_path = "data\sentences_seperated_by_lines_format.txt"
word_list_path = "data\words_list.txt"

with open(word_list_path) as file:
    text = file.read();
print(text[:30])

print(len(text))

tokenizer = BertWordPieceTokenizer(
    clean_text=False,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)
tokenizer.train(files=sentences_path, vocab_size=30_000, min_frequency=2,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=['[PAD', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])

