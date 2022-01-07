import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from tokenizers import Tokenizer
import torch
from transformers import BertTokenizerFast, BertConfig, BertForPreTraining
from transformers import TextDatasetForNextSentencePrediction, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import json


model_path = "tokenizer_30522_default_vocab_truncation_1024"

tokenizer_default = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True)

config = BertConfig()
model = BertForPreTraining(config)

#dataset = load_dataset("cc_news", split="train")

def encode_with_truncation(examples):
  return tokenizer_default(examples["text"], truncation=True, padding="max_length", max_length=1024, return_special_tokens_mask=True)


#train_dataset = dataset["train"].map(encode_with_truncation, batched=True)  
#train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])


print("data_colator")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer_default,
    mlm=True,
    mlm_probability=0.2
)

print("training_args")
training_args = TrainingArguments(
    output_dir= "/bert_output/",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size =4,
    per_device_eval_batch_size=16,   
    #save_total_limit=2,
    learning_rate=2e-5,
    evaluation_strategy="steps", 
    logging_steps=500,            
    save_steps=500,
    #prediction_loss_only=True,
    logging_dir='logs',
)

print("trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    #gpus=8,
    #strategy="ddp"
)

print("training")
trainer.train()
trainer.save_model("bert_1")