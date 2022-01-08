import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from tokenizers import Tokenizer
import torch
from transformers import BertTokenizerFast, BertConfig, BertForPreTraining
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import json


model_path = "tokenizer_30522_default_vocab_truncation_1024"

tokenizer_default = BertTokenizerFast.from_pretrained(
    model_path, do_lower_case=True)

config = BertConfig()
model = BertForPreTraining(config)

#dataset = load_dataset("cc_news", split="train")

dataset = LineByLineTextDataset(
    tokenizer=tokenizer_default,
    file_path="data/sentences_seperated_by_lines_format.txt",
    block_size=1024,
)



#train_dataset = dataset["train"].map(encode_with_truncation, batched=True)
#train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

print(dataset)

print("data_colator")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer_default,
    pad_to_multiple_of=2,
    mlm=True,
    mlm_probability=0.2
)

print("training_args")
training_args = TrainingArguments(
    output_dir="/bert_output/",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    # save_total_limit=2,
    learning_rate=2e-5,
    evaluation_strategy="steps",
    logging_steps=500,
    save_steps=500,
    # prediction_loss_only=True,
    logging_dir='logs',
)

print("trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    # gpus=8,
    # strategy="ddp"
)

print("training")
trainer.train()
trainer.save_model("bert_1")
