import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
import json
import os

# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Загрузка датасета
dataset = load_dataset('json', data_files={'train': 'dataset.json'}, cache_dir='/Users/sesh/Documents/neyro/model/dataset')

# Токенизация
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", cache_dir='/Users/sesh/Documents/neyro/model/hug')

def preprocess_function(examples):

  inputs = examples['input_text']
  targets = examples['json_output']
  model_inputs = tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt")
  model_targets = tokenizer(targets, padding="max_length", truncation=True, return_tensors="pt")
  return {"input_ids": model_inputs["input_ids"],
          "attention_mask": model_inputs["attention_mask"],
          "labels": model_targets["input_ids"]}

dataset = dataset.map(preprocess_function, batched=True)

# Загрузка модели
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large", cache_dir='/Users/sesh/Documents/neyro/model/hug')

# Настройка обучения
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    learning_rate=2e-5,
)

# Обучение модели
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
)

trainer.train()

# Сохранение модели
trainer.save_model("./trained_model")