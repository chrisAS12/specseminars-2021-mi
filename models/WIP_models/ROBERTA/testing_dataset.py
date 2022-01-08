from transformers import TextDataset

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/sentences_seperated_by_lines_format.txt",
    block_size=1024,
)