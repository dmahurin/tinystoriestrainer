from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader
import torch
import pynvml
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import settings
import configurations
import logging

import os

logging.basicConfig(level=logging.DEBUG)

if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DVICE = 'cpu'

if torch.cuda.is_available(): pynvml.nvmlInit()

CHECKPOINT = None # 'checkpoint-50000' # 'checkpoint-70000'
OUTPUT_MODEL = settings.OUTPUT_DIR
results_directory = settings.OUTPUT_DIR

reuse_tokenizer = os.path.exists(settings.OUTPUT_DIR + '/' + 'tokenizer.model')
tokenizer_path = settings.OUTPUT_DIR if reuse_tokenizer else settings.tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
configuration = configurations.get_configuration(settings, tokenizer)
tokenizer.pad_token = tokenizer.eos_token
dataset_name = 'prepared_wikipedia_en'
train_dataset_name = 'train'
validation_dataset_name = 'test'
START_LEARNING_RATE = 5e-5
TRAINING_EPOCHS = 4


def print_gpu_utilization():
    if not torch.cuda.is_available(): return
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    if not torch.cuda.is_available(): return
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache before training


def dataset_metrics(dataset):
	print(f"Size of training dataset: {dataset[train_dataset_name].num_rows} examples")
	if validation_dataset_name in dataset:
		print(f"Size of validation dataset: {dataset[validation_dataset_name].num_rows} examples")

# Initializing a model from the Llama configuration, or from pretrained if CHECKPOINT.
checkpoint_directory = f'{results_directory}/{CHECKPOINT}' if CHECKPOINT is not None else None
resume_from_checkpoint = checkpoint_directory if checkpoint_directory is not None else True if os.path.exists(results_directory) and any(s.startswith("checkpoint") for s in os.listdir(results_directory)) else False
if CHECKPOINT:
	print('loading checkpoint')
	model = LlamaForCausalLM.from_pretrained(checkpoint_directory)
else:
	model = LlamaForCausalLM(LlamaConfig(**configuration))
    
print(f"Number of parameters in model {model.num_parameters()}")
print_gpu_utilization()

model.to(DEVICE)
print_gpu_utilization()

dataset = load_from_disk(dataset_name)
dataset_metrics(dataset)

training_args = TrainingArguments(
    report_to='none',
    output_dir=results_directory,
    overwrite_output_dir=True,
    num_train_epochs=TRAINING_EPOCHS,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_steps=100000,
    weight_decay=0.01,
    max_grad_norm=1.0,
    evaluation_strategy='no',
    save_steps=1000,
    warmup_steps=500,
    save_total_limit=2,
    learning_rate=START_LEARNING_RATE,
    fp16=True
)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset[train_dataset_name],
    eval_dataset=dataset[validation_dataset_name],
    data_collator=data_collator,
    optimizers=(optimizer, None)
)

results = None
try:
	results = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
except:
	raise
finally:
	model.save_pretrained(OUTPUT_MODEL)
	if not reuse_tokenizer: tokenizer.save_pretrained(OUTPUT_MODEL)
	if results:
		with open(f'{OUTPUT_MODEL}/training_results.json', 'w') as result_file:
		    json.dump(results, result_file)
