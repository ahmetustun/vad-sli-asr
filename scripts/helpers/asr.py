import json
import numpy as np
import glob
import os
import pandas as pd
import re
import torch

from dataclasses import dataclass
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    disable_progress_bar,
    enable_progress_bar,
    load_metric
)
from typing import Dict, List, Union
from pyctcdecode import build_ctcdecoder
from transformers import (
    AutoConfig,
    AutoProcessor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM
)

from .custom_wav2vec2 import Wav2Vec2ForCTC
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor


def dataset_from_dict(dataset_dict):

    dataset = DatasetDict()

    for k in dataset_dict.keys():
        dataset[k] = Dataset.from_pandas(pd.read_csv(dataset_dict[k], sep='\t'))

    return dataset

def remove_special_characters(batch):
    chars_to_ignore_regex = '[\-\,\.\!\;\:\"\“\%\”\�136]'
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"])

    # for ft on CGN subset
    # batch["sentence"] = re.sub('[Á]', 'A', batch["sentence"])
    # batch["sentence"] = re.sub('[Ä]', 'A', batch["sentence"])
    # batch["sentence"] = re.sub('[Å]', 'A', batch["sentence"])
    # batch["sentence"] = re.sub('[Ç]', 'C', batch["sentence"])
    # batch["sentence"] = re.sub('[Ê]', 'E', batch["sentence"])
    # batch["sentence"] = re.sub('[Ï]', 'I', batch["sentence"])
    # batch["sentence"] = re.sub('[Ô]', 'O', batch["sentence"])
    # batch["sentence"] = re.sub('[Ú]', 'U', batch["sentence"])
    # batch["sentence"] = re.sub('[Ü]', 'U', batch["sentence"])

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
    
    return batch

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    
    return {"vocab": [vocab], "all_text": [all_text]}

def create_vocab(dataset_dict, word_delimiter_token = "|", special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]):

    vocab_list = []

    for ds_name, ds_data in dataset_dict.items():
        vocab = ds_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=ds_data.column_names)

        vocab_list.extend(vocab["vocab"][0])

    vocab_list = list(set(vocab_list))
    vocab_dict = { v: k for k, v in enumerate(sorted(vocab_list)) }

    vocab_dict[word_delimiter_token] = vocab_dict[" "]
    del vocab_dict[" "]

    for t in special_tokens:
        vocab_dict[t] = len(vocab_dict)

    return vocab_dict

def preprocess_text(dataset_dict):

    disable_progress_bar()

    print("Pre-processing transcriptions ...")
    dataset_dict = dataset_dict.map(remove_special_characters)

    vocab_path = create_vocab(dataset_dict)

    enable_progress_bar()
    
    return dataset_dict, vocab_path

def process_data(dataset_dict, processor):

    print("Processing data ...")

    def _helper(batch, processor=processor):
        audio = batch["path"]

        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        
        # 2022-03-09:
        # Comment out input_length, not sure what actually requires this column
        # But including it results in a warning from Wav2Vec2ForCTC.forward
        # batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
    
        return batch

    dataset_dict = dataset_dict.cast_column("path", Audio(sampling_rate=16_000))

    for ds_name, ds_data in dataset_dict.items():
        dataset_dict[ds_name] = ds_data.map(_helper, remove_columns=ds_data.column_names)

    return dataset_dict

@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def get_metrics_computer(processor):

    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    def compute_metrics(pred):

        pred_logits = pred.predictions

        if type(processor).__name__ == "Wav2Vec2ProcessorWithLM":
            pred_str    = processor.batch_decode(pred_logits).text
        else:
            pred_ids = np.argmax(pred_logits, axis=-1)
            pred_str = processor.batch_decode(pred_ids)

        # Replace data collator padding with tokenizer's padding
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        # Retrieve labels as characters, e.g. 'hello', from label_ids, e.g. [5, 3, 10, 10, 2] (where 5 = 'h')
        label_str = processor.tokenizer.batch_decode(pred.label_ids, group_tokens=False)
        
        print(pd.DataFrame({
            "pred_str"  : pred_str,
            "label_str" : label_str
        }))

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    return compute_metrics

def configure_w2v2_for_training(dataset, args, vocab_dict, w2v2_config={}):

    feature_extractor_kwargs = w2v2_config["feature_extractor"] if "feature_extractor" in w2v2_config.keys() else {}
    model_kwargs = w2v2_config["model_kwargs"] if "model_kwargs" in w2v2_config.keys() else {}

    bottleneck_adapters_kwargs = w2v2_config["bottleneck_adapters_kwargs"] \
        if "bottleneck_adapters_kwargs" in w2v2_config.keys() else {}
    model_kwargs.update(bottleneck_adapters_kwargs)

    cnn_adapters_kwargs = w2v2_config["cnn_adapters_kwargs"] \
        if "cnn_adapters_kwargs" in w2v2_config.keys() else {}
    model_kwargs.update(cnn_adapters_kwargs)

    if args.use_target_vocab is True:
        vocab_path = os.path.join(args.output_dir, 'vocab.json')

        print(f"Writing created vocabulary to {vocab_path}")

        with open(vocab_path, 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)

        AutoConfig.from_pretrained(args.repo_path_or_name).save_pretrained(args.output_dir)
        tokenizer = Wav2Vec2CTCTokenizer(vocab_path)

    else:

        print("Using vocabulary from tokenizer ...")
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.repo_path_or_name)

    feature_extractor = Wav2Vec2FeatureExtractor(**feature_extractor_kwargs)

    processor = Wav2Vec2Processor(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor
    )

    processor.save_pretrained(args.output_dir)

    if args.use_target_vocab:
        model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_model_name_or_path=args.repo_path_or_name,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            **model_kwargs
        )

    else:
        model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_model_name_or_path=args.repo_path_or_name,
            **model_kwargs
        )

    model.freeze_feature_encoder()
    model.unfree_cnn_adapters()
    model.unfreeze_bottleneck_adapters()
    model.unfree_cnn_adapters()

    return model, processor

def configure_w2v2_for_inference(repo_path_or_name, cache_dir="tmp/"):

    if os.path.isdir(repo_path_or_name):
        cp_path   = glob.glob(os.path.join(repo_path_or_name, 'checkpoint-*'))[0]
        model     = Wav2Vec2ForCTC.from_pretrained(cp_path, cache_dir=cache_dir)
        processor = AutoProcessor.from_pretrained(repo_path_or_name, cache_dir=cache_dir)
    else:
        model     = Wav2Vec2ForCTC.from_pretrained(repo_path_or_name)
        processor = AutoProcessor.from_pretrained(repo_path_or_name)

    if torch.cuda.is_available():
        model.to("cuda")

    def predict(batch):
        input_values = processor(batch["speech"], return_tensors="pt", padding="longest", sampling_rate=16_000).input_values

        with torch.no_grad():
            logits = model(input_values.to("cuda")).logits if torch.cuda.is_available() else model(input_values).logits

        if type(processor).__name__ == 'Wav2Vec2ProcessorWithLM':
            transcription = processor.batch_decode(logits.cpu().numpy()).text
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)

        batch["transcription"] = transcription[0] if isinstance(transcription, list) else transcription

        return batch
    
    return model, processor, predict

def configure_lm(processor, arpa_path, output_dir):

    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path=arpa_path,
    )

    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder
    )

    processor_with_lm.save_pretrained(output_dir)

    return processor_with_lm
