import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from tokenizers import Tokenizer
import torch
from transformers import BertTokenizerFast, BertConfig, BertForPreTraining
from transformers import TextDatasetForNextSentencePrediction, DataCollatorForLanguageModeling, Trainer, TrainingArguments

model_path = "tokenizer_30522_default_vocab_truncation_1024"

tokenizer_default = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True,max_len=1024)
print(tokenizer_default("hello!", ))
config = BertConfig()
model = BertForPreTraining(config)

device = torch.device("cpu")


print("dataset")
dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer_default,
    file_path="data/sentences_seperated_by_lines_format.txt",
    block_size=1024
)

print("data_colator")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer_default,
    mlm=True,
    mlm_probability=0.10
)

print("training_args")
training_args = TrainingArguments(
    output_dir= "/bert_output/",
    overwrite_output_dir=True,
    num_train_epochs=3,
    #save_steps=5000,
    per_device_train_batch_size =4,
    per_device_eval_batch_size=16,   
    #save_total_limit=2,
    learning_rate=2e-5,
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