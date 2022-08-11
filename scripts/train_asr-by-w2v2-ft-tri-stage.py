import json
import math
import os
import torch

from argparse import ArgumentParser
from datasets import load_metric
from helpers.asr import (
    configure_lm,
    configure_w2v2_for_training,
    DataCollatorCTCWithPadding,
    dataset_from_dict,
    get_metrics_computer,
    preprocess_text,
    process_data
)
from transformers import (
    EarlyStoppingCallback,
    logging,
    Trainer,
    TrainingArguments
)

parser = ArgumentParser(
    prog='train_asr-by-w2v2-ft',
    description='Train an ASR model by fine-tuning a pre-trained wav2vec 2.0 model',
)

parser.add_argument('repo_path_or_name', help = "Pre-trained wav2vec 2.0 model, local path or HuggingFace repo name")
parser.add_argument('output_dir', help = "The output directory where the model predictions and checkpoints will be written")

parser.add_argument('train_tsv', help = "Training data. Two-column tab-separated file with 'path' (path to wav file) and 'sentence' (transcription)")
parser.add_argument('eval_tsv', help = "Evaluation data. Two-column tab-separated file with 'path' (path to wav file) and 'sentence' (transcription)")

parser.add_argument('lr', default=1e-4, help='Learning rate for AdamW optimizer')

parser.add_argument('--use_target_vocab', default=True, help='Use a vocabulary created from target transcriptions (training and evaluation)')

parser.add_argument('--lm_arpa', default=None, help='Path to language model .arpa file (optional)')

parser.add_argument('--hft_logging', default=40, help='HuggingFace Transformers verbosity level (40 = errors, 30 = warnings, 20 = info, 10 = debug)')

args = parser.parse_args()
print(args)

# Turns out bool('False') evaluates to True in Python (only bool('') is False)
args.use_target_vocab = False if args.use_target_vocab == 'False' else True

logging.set_verbosity(args.hft_logging)

# For debugging
# args.repo_path_or_name = "facebook/wav2vec2-large-robust-ft-swbd-300h"
# args.train_tsv = 'data/train-asr/train.tsv'
# args.eval_tsv  = 'data/train-asr/test.tsv'
# args.output_dir = 'data/asr-temp'
# args.use_target_vocab = False

os.makedirs(args.output_dir, exist_ok=True)

dataset = dataset_from_dict({
    'train': args.train_tsv,
    'eval' : args.eval_tsv
})

w2v2_config = {
    "feature_extractor" : {
        "return_attention_mask" : True
    },
    "model_kwargs" : {
        "mask_time_prob" : 0,
        "gradient_checkpointing" : True,
        "ctc_loss_reduction" : "mean"
    },
    "bottleneck_adapters_kwargs" : {
        "use_bottleneck_adapter": True,
        "bottleneck_adapter_dim" : 256,
        "bottleneck_adapter_act" : "gelu",
        "unfreeze_layernorm" : True,
        "unfreeze_encoder": True
    }
}

dataset, vocab_dict = preprocess_text(dataset)

model, processor = configure_w2v2_for_training(dataset, args, vocab_dict, w2v2_config)

# Number of trainable parameters
print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')
print(f'Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
print(f'Trainable adapter parameters: {sum(p.numel() for n,p in model.named_parameters() if "bottleneck_adapter" in n)}')

if args.lm_arpa is not None:
    processor = configure_lm(processor, args.lm_arpa, args.output_dir)

dataset = process_data(dataset, processor)

# Set logging to 'INFO' or else progress bar gets hidden
logging.set_verbosity(20)

n_epochs   = 50
batch_size = 32

# How many epochs between evals?
eps_b_eval = 5 
# Save/Eval/Logging steps
sel_steps  = int(math.ceil(len(dataset['train']) / batch_size) * eps_b_eval)

# Learning rate
lr = float(args.lr)
print(f'Learning rate set to {lr}')

# set-up tri-stage learning rate scheduler

def get_flat_linear_schedule_with_warmup(optimizer, num_warmup_steps:int, num_training_steps:int, last_epoch:int =-1):

    print(f"num training steps: {num_training_steps}")
    
    def lr_lambda(current_step):
        constant_steps = int(num_training_steps * 0.4)
        warmup_steps = int(num_training_steps * 0.1)
        
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps+constant_steps:
            return 1
        else:
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (warmup_steps+constant_steps)))
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_flat_cheduler(name = None, optimizer = None, num_warmup_steps = None, num_training_steps = None):
    return get_flat_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

class ReplicationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def create_flat_scheduler(self, num_training_steps: int):
        self.lr_scheduler = get_flat_cheduler(optimizer = self.optimizer,
                                              num_training_steps=num_training_steps)
    def create_optimizer_and_scheduler(self, num_training_steps):
        self.create_optimizer()
        self.create_flat_scheduler(num_training_steps)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    group_by_length=True,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    num_train_epochs=n_epochs,
    fp16=False,
    seed=4892,
    save_steps=sel_steps,
    eval_steps=sel_steps,
    logging_steps=sel_steps,
    learning_rate=lr,
    # Warm up: 100 steps or 10% of total optimisation steps
    # warmup_steps=min(100, int(0.1 * sel_steps * n_epochs)),
    # warmup_steps=500,
    # 2022-03-09: manually set optmizier to PyTorch implementation torch.optim.AdamW
    # 'adamw_torch' to get rid of deprecation warning for default optimizer 'adamw_hf'
    optim="adamw_torch",
    metric_for_best_model="wer",
    save_total_limit=2,
    load_best_model_at_end = True,
    # Lower WER is better
    greater_is_better=False,
    dataloader_num_workers=4,
    report_to = 'wandb',
    run_name = args.repo_path_or_name.split('/')[-2] + '-ft-' + str(lr)
    # run_name = args.repo_path_or_name + '-' + str(lr) + '-pre-train-frisian'
)

trainer = ReplicationTrainer(
    model=model,
    data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
    args=training_args,
    compute_metrics=get_metrics_computer(processor=processor),
    train_dataset=dataset['train'],
    eval_dataset=dataset['eval'],
    tokenizer=processor.feature_extractor,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Training model ...")
trainer.train()
