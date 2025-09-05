#!/usr/bin/env python
import os
import sys

import json
import torch
import glob
import pickle

from torch import nn
import torchaudio
import torchaudio.transforms as at
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from types import SimpleNamespace

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import whisper
from transformers import (
    get_linear_schedule_with_warmup
)
from transformers import WhisperTokenizer
import random
import evaluate
# Set global constants
SAMPLE_RATE = 16000
BATCH_SIZE = 2
TRAIN_RATE = 0.8
AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 1000
SEED = 3407
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pl.seed_everything(SEED, workers=True)


# Audio loading function
def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

# Function to extract text from JSON structure
def extract_transcription(data):
    """Extracts transcription text from JSON structure"""
    # Check if segments exist and have text content
    if 'segments' in data and isinstance(data['segments'], list) and len(data['segments']) > 0:
        # Combine all segment texts (for multi-segment files)
        return ''.join(segment.get('text', '') for segment in data['segments'])
    return ''

# Main dataset processing function for a single dataset type
def process_dataset_type(
    json_dir,
    audio_dir,
    extension,
    text_max_length=200,
    audio_max_sample_length=480000,
    sample_rate=16000,
):
    audio_transcript_pair_list = []
    json_path_list = list(Path(json_dir).rglob("*.json"))
    
    print(f"\nProcessing {len(json_path_list)} JSON files from {json_dir}")
    print(f"Using audio extension: {extension}")
    
    for json_path in tqdm(json_path_list):
        # Load JSON content
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON {json_path}: {str(e)}")
            continue
        
        # Extract text from JSON structure
        text = extract_transcription(data)
        if not text:
            print(f"Missing text in {json_path}")
            continue
        
        # Construct audio path
        rel_path = json_path.relative_to(Path(json_dir))
        audio_path = Path(audio_dir) / rel_path.with_suffix(extension)
        
        # Validate paths
        if not audio_path.exists():
            print(f"Missing audio: {audio_path}")
            continue

        # Load and validate audio
        try:
            audio = load_wave(audio_path, sample_rate)[0]
            audio_length = len(audio)
            
            if len(text) > text_max_length:
                print(f"Skipped (text too long: {len(text)} chars) for {json_path.stem}")
                continue
                
            if audio_length > audio_max_sample_length:
                print(f"Skipped (audio too long: {audio_length} samples) for {json_path.stem}")
                continue
                
        except Exception as e:
            print(f"Error loading audio {audio_path}: {str(e)}")
            continue
        
        # Add valid pair
        audio_transcript_pair_list.append((json_path.stem, str(audio_path), text))
    
    print(f"Found {len(audio_transcript_pair_list)} valid pairs")
    return audio_transcript_pair_list

# Main function to process all datasets
def process_all_datasets(base_dir):
    all_pairs = []
    base_path = Path(base_dir)
    
    # Define dataset types and their configurations
    dataset_configs = [
        # {
        #     "name": "non-vocoded",
        #     "json_dir": base_path / "tedx_fr_random_3k_json",
        #     "audio_dir": base_path / "tedx_fr_random_3k",
        #     "extension": ".wav"
        # },
        {
            "name": "vocoded",
            "json_dir": base_path / "tedx_fr_random_3k_vocoded_json",
            "audio_dir": base_path / "tedx_fr_random_3k_vocoded",
            "extension": ".flac"
        }
    ]
    
    # Process each dataset type
    for config in dataset_configs:
        if not config["json_dir"].exists():
            print(f"\n⚠️ Missing JSON directory: {config['json_dir']}")
            continue
        if not config["audio_dir"].exists():
            print(f"\n⚠️ Missing audio directory: {config['audio_dir']}")
            continue
            
        print(f"\n{'='*40}")
        print(f"Processing {config['name']} dataset")
        print(f"JSON directory: {config['json_dir']}")
        print(f"Audio directory: {config['audio_dir']}")
        print(f"{'='*40}")
        
        pairs = process_dataset_type(
            json_dir=config["json_dir"],
            audio_dir=config["audio_dir"],
            extension=config["extension"],
            text_max_length=TEXT_MAX_LENGTH,
            audio_max_sample_length=AUDIO_MAX_LENGTH,
            sample_rate=SAMPLE_RATE
        )
        all_pairs.extend(pairs)
    
    return all_pairs

class TedXSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate, nmel) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.nmel = nmel

    def __len__(self):
        return len(self.audio_info_list)
    
    def __getitem__(self, id):
        audio_id, audio_path, text = self.audio_info_list[id]

        # audio
        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, self.nmel)
        
        raw_text = text

        # print(text)
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        # print(text)
        labels = text[1:] + [self.tokenizer.eot]
        # print(labels)
        # print("tooooooooooooooooooooooooooooooooooooooo")
        # print(self.tokenizer.decode(text))
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text,
            'raw_text': raw_text
        }

class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids, raw_texts = [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            raw_texts.append(f['raw_text'])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids
        batch['raw_texts'] = raw_texts
        return batch

class Config:
    learning_rate = 0.000003
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 16
    num_worker = 2
    num_train_epochs = 30
    gradient_accumulation_steps = 1
    sample_rate = 16000


class WhisperModelModule(LightningModule):
    def __init__(self, cfg:Config, model_name="base", lang="fr", train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="fr", task=self.options.task)

        # number of mels
        self.nmel = self.model.dims.n_mels

        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            # audio_features = whisper.log_mel_spectrogram(input_ids, n_mels=self.model.dims.n_mels).to(self.model.device)
            audio_features = self.model.encoder(input_ids)

        # out = self.model.decoder(dec_input_ids, audio_features)
        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        # audio_features = whisper.log_mel_spectrogram(input_ids, n_mels=self.model.dims.n_mels).to(self.model.device)
        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o))
            l_list.append(self.tokenizer.decode(l))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val_cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val_wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "val_cer": cer,
            "val_wer": wer,
            "val_loss": loss
        }

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
                          lr=self.cfg.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
    
    def train_dataloader(self):
        dataset = TedXSpeechDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate, self.nmel)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        dataset = TedXSpeechDataset(self.__eval_dataset, self.tokenizer, self.cfg.sample_rate, self.nmel)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

# checkpoint_path = "content/artifacts/checkpoint/checkpoint-epoch=0014-v1.ckpt"

# state_dict = torch.load(checkpoint_path)
# print(state_dict.keys())
# state_dict = state_dict['state_dict']

# whisper_model = WhisperModelModule(cfg)
# whisper_model.load_state_dict(state_dict)

# woptions = whisper.DecodingOptions(language="fr", without_timestamps=True)
# # dataset = TedXSpeechDataset(eval_pairs, wtokenizer, SAMPLE_RATE)
# dataset = TedXSpeechDataset(all_audio_transcript_pairs, wtokenizer, SAMPLE_RATE)

# loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())
# hf_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
# refs = []
# res = []

# for b in tqdm(loader):
#     input_ids = b["input_ids"].half().cuda()
#     labels = b["labels"].long().cuda()
#     with torch.no_grad():
#         # print(input_ids)
#         # print(labels)
#         # exit()
#         # decoded_with_special = hf_tokenizer.decode(labels[0], skip_special_tokens=False)
#         # decoded_str = hf_tokenizer.decode(labels[0], skip_special_tokens=True)
#         # print(f"Input:                 {input_ids}")
#         # print(f"Tokens:                {labels}")
#         # print(f"Token !!!!!!:          {50199}")
#         # print(f"Token !!!!!!!!:        {28618}")
#         # print(f"Decoded w/ special:    {decoded_with_special}")
#         # print(f"Decoded w/out special: {decoded_str}")
#         # print(f"Are equal:             {input_ids == decoded_str}")
#         # # audio_features = whisper_model.model.encoder(input_ids)
#         # out = whisper_model.model.decoder(enc_input_ids, audio_features)
#         results = whisper_model.model.decode(input_ids, woptions)
#         for r in results:
#             res.append(r.text)
        
#         for l in labels:
#             l[l == -100] = wtokenizer.eot
#             ref = wtokenizer.decode(l)
#             refs.append(ref)

# cer_metrics = evaluate.load("cer")
# cer_metrics.compute(references=refs, predictions=res)

# for k, v in zip(refs, res):
#     print("-"*10)
#     print(k)
#     print(v)



import torch
from tqdm import tqdm
import evaluate

vocoding_label = "!!!!!!"

# -----------------------------
# 1. Helper functions for vocoding evaluation
# -----------------------------
def extract_vocoding_labels(text: str):
    """
    Given a text with vocoded words prefixed by '!!!!!!',
    return a list of (label, word) where label is 'bonafide' or 'spoof'.
    """
    words = text.strip().split()
    labels = []
    i = 0
    while i < len(words):
        if words[i] == vocoding_label:
            if i + 1 < len(words):
                labels.append(("spoof", words[i + 1]))
                i += 2
            else:
                i += 1
        else:
            labels.append(("bonafide", words[i]))
            i += 1
    return labels

def remove_vocoding_labels(text: str):
    return text.replace(vocoding_label + ' ', '')

def remove_special_token(tiktoken_enc, tokens):
    return [x for x in tokens if x not in tiktoken_enc._special_tokens.values()]

def compute_detection_metrics(refs, preds):
    a = b = c = d = 0

    for ref_text, hyp_text in zip(refs, preds):
        ref_labels = extract_vocoding_labels(ref_text)
        hyp_labels = extract_vocoding_labels(hyp_text)

        min_len = min(len(ref_labels), len(hyp_labels))

        for i in range(min_len):
            gt, _ = ref_labels[i]
            pred, _ = hyp_labels[i]

            if gt == "bonafide" and pred == "bonafide":
                a += 1
            elif gt == "spoof" and pred == "spoof":
                b += 1
            elif gt == "bonafide" and pred == "spoof":
                c += 1  # false negative
            elif gt == "spoof" and pred == "bonafide":
                d += 1  # false positive

    total = a + b + c + d
    accuracy = (a + b) / total if total > 0 else 0
    false_positive_rate = d / (d + b) if (d + b) > 0 else 0
    false_negative_rate = c / (a + c) if (a + c) > 0 else 0

    return {
        "a_true_bonafide": a,
        "b_true_spoof": b,
        "c_false_negative": c,
        "d_false_positive": d,
        "accuracy": accuracy,
        "FPR": false_positive_rate,
        "FNR": false_negative_rate,
    }


# load configuration file
with open(sys.argv[1], encoding="utf-8") as fin:
    cfg = SimpleNamespace(**load_hyperpyyaml(fin))
    

# Dataset setup - point to your base directory
BASE_DIR = "tedx_test"
print(f"\n{'#'*50}")
print(f"Starting dataset processing in: {BASE_DIR}")
print(f"{'#'*50}\n")

# Process all datasets
all_audio_transcript_pairs = process_all_datasets(BASE_DIR)
print(f"\nTOTAL AUDIO-TEXT PAIRS FOUND: {len(all_audio_transcript_pairs)}")

# Split into train and eval
train_num = int(len(all_audio_transcript_pairs) * TRAIN_RATE)
_, eval_pairs = all_audio_transcript_pairs[:train_num], all_audio_transcript_pairs[train_num:]

print(f"EVAL DATASET SIZE: {len(eval_pairs)}")

if len(eval_pairs) > 0:
    print("\nSample eval item:")
    print(f"ID: {eval_pairs[0][0]}")
    print(f"Audio: {eval_pairs[0][1]}")
    print(f"Text: {eval_pairs[0][2][:50]}...")

# (Add your Whisper model and training code below)

woptions = whisper.DecodingOptions(language="fr", without_timestamps=True)
wtokenizer = whisper.tokenizer.get_tokenizer(True, language="fr", task=woptions.task)



model_name = cfg.model_name
lang = cfg.lang

project_dir = 'cp_whisper_{:s}_lr_{:3.0e}'.format(model_name, cfg.learning_rate)
log_output_dir = "{:s}/logs".format(project_dir)
check_output_dir = "{:s}/artifacts".format(project_dir)
cp_dir = "{:s}/checkpoint".format(check_output_dir)
save_dir = "{:s}/outputs".format(check_output_dir)

Path(save_dir).mkdir(parents=True, exist_ok=True)
train_name = "whisper_finetune_lr_{:3.0e}".format(cfg.learning_rate)
train_id = "all_finetune_lr_{:3.0e}".format(cfg.learning_rate)


# -----------------------------
# 2. Load checkpoint and prepare model
# -----------------------------
# sorted by epoch
checkpoint = sorted(glob.glob("*.ckpt", root_dir=cp_dir), 
                    key=lambda x: x.split('-')[1])[-1]
checkpoint_path = "{:s}/{:s}".format(cp_dir, checkpoint)
print("Use {:s}".format(checkpoint_path))

state_dict = torch.load(checkpoint_path)
state_dict = state_dict['state_dict']

whisper_model = WhisperModelModule(cfg, model_name, lang)
whisper_model.load_state_dict(state_dict)

woptions = whisper.DecodingOptions(language="fr", without_timestamps=True)

dataset = TedXSpeechDataset(eval_pairs, wtokenizer, SAMPLE_RATE, whisper_model.nmel)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())

#hf_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")

# -----------------------------
# 3. Run inference
# -----------------------------
refs = []
refs_raw = []
res = []
res_raw = []
for b in tqdm(loader):
    input_ids = b["input_ids"].half().cuda()
    labels = b["labels"].long()
    rawtexts = b['raw_texts']
    with torch.no_grad():
        results = whisper_model.model.decode(input_ids, woptions)
        for r in results:
            res.append(r.text)
            res_raw.append(remove_vocoding_labels(r.text))

        for l in labels:
            l[l == -100] = wtokenizer.eot
            ref = wtokenizer.decode(remove_special_token(wtokenizer.encoding, l))
            refs.append(ref)

        for x in rawtexts:
            refs_raw.append(remove_vocoding_labels(x))

# save the output
with open(Path(save_dir) / '{:s}.inference.txt.pkl'.format(checkpoint), 'wb') as file_ptr:
    pickle.dump([res, res_raw, refs, refs_raw], file_ptr)
# -----------------------------
# 4. Compute CER
# -----------------------------
cer_metrics = evaluate.load("cer")
cer = cer_metrics.compute(references=refs_raw, predictions=res_raw)
print(f"CER: {cer:.4f}")

# WER
wer_metrics = evaluate.load("wer")
wer = wer_metrics.compute(references=refs_raw, predictions=res_raw)
print(f"WER: {wer:.4f}")

# -----------------------------
# 5. Compute vocoded detection metrics
# -----------------------------
metrics = compute_detection_metrics(refs, res)
print("\nVocoded Word Detection Results:")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -----------------------------
# 6. Optional: print some examples
# -----------------------------
for k, v in zip(refs[:10], res[:10]):
    print("-" * 20)
    print("REF:", k)
    print("HYP:", v)

def count_phrases_and_words(texts):
    num_phrases = len(texts)
    num_words = 0
    for t in texts:
        words = t.strip().split()
        # Don't count the '!!!!!!' markers as words
        num_words += sum(1 for w in words if w != "!!!!!!")
    return num_phrases, num_words


# Count in references
num_phrases_ref, num_words_ref = count_phrases_and_words(refs)

# Count in predictions
num_phrases_pred, num_words_pred = count_phrases_and_words(res)

print(f"References:  {num_phrases_ref} phrases, {num_words_ref} words")
print(f"Predictions: {num_phrases_pred} phrases, {num_words_pred} words")
