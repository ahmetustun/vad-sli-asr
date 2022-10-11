import os
import numpy as np
import pandas as pd
import torchaudio
import re
import torch
import librosa

from datasets import Dataset
from helpers.asr import configure_w2v2_for_inference
from jiwer import wer, cer

from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC
)

from datasets import (
    Dataset
)

from glob import glob

EVAL_MODELS_DATASETS = [
    # Frisian data: baselines DEV
    ("/checkpoints/baselines/GroNLP/wav2vec2-dutch-large-5e-5-baseline", "data/frisian-ft/dev.tsv"),
    ("/checkpoints/baselines/facebook/wav2vec2-large-5e-5-baseline", "data/frisian-ft/dev.tsv")
]

def speech_file_to_array_fn(batch):
    speech_array, _ = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    
    return batch

def remove_special_characters(batch):
    chars_to_ignore_regex = '[\-\,\.\!\;\:\"\“\%\”\�136]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"])

    # for ft on Frisian subset
    batch["sentence"] = re.sub('[á]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[à]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[ä]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[å]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[ç]', 'c', batch["sentence"])
    batch["sentence"] = re.sub('[č]', 'c', batch["sentence"])
    batch["sentence"] = re.sub('[è]', 'e', batch["sentence"])
    batch["sentence"] = re.sub('[ë]', 'e', batch["sentence"])
    batch["sentence"] = re.sub('[ï]', 'i', batch["sentence"])
    batch["sentence"] = re.sub('[ö]', 'o', batch["sentence"])
    batch["sentence"] = re.sub('[ü]', 'u', batch["sentence"])

    batch["transcription"] = re.sub(chars_to_ignore_regex, '', batch["transcription"])

    # for ft on Frisian subset
    batch["transcription"] = re.sub('[á]', 'a', batch["transcription"])
    batch["transcription"] = re.sub('[à]', 'a', batch["transcription"])
    batch["transcription"] = re.sub('[ä]', 'a', batch["transcription"])
    batch["transcription"] = re.sub('[å]', 'a', batch["transcription"])
    batch["transcription"] = re.sub('[ç]', 'c', batch["transcription"])
    batch["transcription"] = re.sub('[č]', 'c', batch["transcription"])
    batch["transcription"] = re.sub('[è]', 'e', batch["transcription"])
    batch["transcription"] = re.sub('[ë]', 'e', batch["transcription"])
    batch["transcription"] = re.sub('[ï]', 'i', batch["transcription"])
    batch["transcription"] = re.sub('[ö]', 'o', batch["transcription"])
    batch["transcription"] = re.sub('[ü]', 'u', batch["transcription"])
    
    return batch


def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["transcription"] = processor.batch_decode(pred_ids)
    
    return batch

EVAL_RESULTS = []
for model_path, testset_path in EVAL_MODELS_DATASETS:

    print(f"Reading in data from {testset_path} ...")
    test_ds = Dataset.from_pandas(pd.read_csv(testset_path, sep = '\t'))
    test_ds = test_ds.map(speech_file_to_array_fn)

    cp_path = glob(os.path.join(model_path, 'checkpoint-*'))[0]
    model = Wav2Vec2ForCTC.from_pretrained(cp_path)
    model = model.cuda()
    processor = AutoProcessor.from_pretrained(model_path)

    test_ds = test_ds.map(evaluate, batched=True, batch_size=8)
    test_ds = test_ds.map(remove_special_characters)

    EVAL_RESULTS.append({
        "model" : os.path.basename(model_path),
        "model_lm" : type(processor).__name__ == 'Wav2Vec2ProcessorWithLM',
        "testset" : os.path.basename(testset_path),
        "wer" : round(wer(test_ds['sentence'], test_ds['transcription']), 3),
        "cer" : round(cer(test_ds['sentence'], test_ds['transcription']), 3)
    })

results_df = pd.DataFrame(EVAL_RESULTS)
print(results_df)

results_df.to_csv("data/exps-eval/asr/asr_wer-csr.csv", index=False)
print("Results written to data/exps-eval/asr/asr_wer-csr.csv")
