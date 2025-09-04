#!/usr/bin/env python
import os
import sys

import json
import torch
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
            "dec_input_ids": text
        }

class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

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

        return batch

class Config:
    learning_rate = 0.000001
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 16
    num_worker = 2
    num_train_epochs = 100
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

# Dataset setup
# DATASET_DIR_SPOOF = "tedx_test/tedx_fr_random_3k_vocoded_json"
# DATASET_DIR_BONAFIDE = "tedx_test/tedx_fr_random_3k_json"
# DATASET_DIR = "tedx_test/"
# # DATASET_DIR = "tedx/tedx_fr_random_3k_json"
# dataset_dir_spoof = Path(DATASET_DIR_SPOOF)
# dataset_dir_bonafide = Path(DATASET_DIR_BONAFIDE)
# json_path_list_spoof = list(dataset_dir_spoof.rglob("*.json"))
# json_path_list_bonafide = list(dataset_dir_bonafide.rglob("*.json"))
# print(f"Found {len(json_path_list_spoof)} SPOOF JSON files")
# print(f"Found {len(json_path_list_bonafide)} BONA FIDE JSON files")

# json_path_list = json_path_list_spoof + json_path_list_bonafide
# print(f"Found {len(json_path_list)} JSON files")
# random.shuffle(json_path_list)
# Split dataset

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
train_pairs, eval_pairs = all_audio_transcript_pairs[:train_num], all_audio_transcript_pairs[train_num:]

print(f"\nTRAIN DATASET SIZE: {len(train_pairs)}")
print(f"EVAL DATASET SIZE: {len(eval_pairs)}")

# print("\nProcessing training set...")
# train_pairs = get_audio_file_list(
#     train_pairs,
#     TEXT_MAX_LENGTH,
#     AUDIO_MAX_LENGTH,
#     SAMPLE_RATE,
# )

# print("\nProcessing evaluation set...")
# eval_pairs = get_audio_file_list(
#     eval_pairs,
#     TEXT_MAX_LENGTH,
#     AUDIO_MAX_LENGTH,
#     SAMPLE_RATE
# )

# print(f"\nTRAIN AUDIO DATASET NUM: {len(train_pairs)}")
# print(f"EVAL AUDIO DATASET NUM: {len(eval_pairs)}")

# Sample output for verification
if len(train_pairs) > 0:
    print("\nSample training item:")
    print(f"ID: {train_pairs[0][0]}")
    print(f"Audio: {train_pairs[0][1]}")
    print(f"Text: {train_pairs[0][2][:50]}...")

# (Add your Whisper model and training code below)

woptions = whisper.DecodingOptions(language="fr", without_timestamps=True)
#wmodel = whisper.load_model("base")
wtokenizer = whisper.tokenizer.get_tokenizer(True, language="fr", task=woptions.task)

# dataset = TedXSpeechDataset(eval_pairs, wtokenizer, SAMPLE_RATE)
# loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())

# for b in loader:
#     # print(b.keys())
#     # exit()
#     print(b["labels"].shape)
#     print(b["input_ids"].shape)
#     print(b["dec_input_ids"].shape)

#     for token, dec in zip(b["labels"], b["dec_input_ids"]):
#         token[token == -100] = wtokenizer.eot
#         text = wtokenizer.decode(token)
#         print(text)

#         dec[dec == -100] = wtokenizer.eot
#         text = wtokenizer.decode(dec)
#         # print(text)
#         # print("tototototototottoto")


#     break

# with torch.no_grad():
#     audio_features = wmodel.encoder(b["input_ids"].cuda())
#     input_ids = b["input_ids"]
#     labels = b["labels"].long()
#     dec_input_ids = b["dec_input_ids"].long()

        
#     audio_features = wmodel.encoder(input_ids.cuda())
#     print(dec_input_ids)
#     print(input_ids.shape, dec_input_ids.shape, audio_features.shape)
#     print(audio_features.shape)
# out = wmodel.decoder(dec_input_ids.cuda(), audio_features)

# tokens = torch.argmax(out, dim=2)
# for token in tokens:
#     token[token == -100] = wtokenizer.eot
#     print(token)
#     text = wtokenizer.decode(token)
#     print(text)
    # decode = wtokenizer.decode("!!!!")
    # print(decode)

#cfg = Config()
model_name = cfg.model_name
lang = cfg.lang

project_dir = 'cp_whisper_{:s}_lr_{:3.0e}'.format(model_name, cfg.learning_rate)
log_output_dir = "{:s}/logs".format(project_dir)
check_output_dir = "{:s}/artifacts".format(project_dir)

train_name = "whisper_finetune_lr_{:3.0e}".format(cfg.learning_rate)
train_id = "all_finetune_lr_{:3.0e}".format(cfg.learning_rate)

# %%

Path(log_output_dir).mkdir(parents=True, exist_ok=True)
Path(check_output_dir).mkdir(parents=True, exist_ok=True)

tflogger = TensorBoardLogger(
    save_dir=log_output_dir,
    name=train_name,
    version=train_id
)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{check_output_dir}/checkpoint",
    filename="checkpoint-{epoch:04d}-{val_loss:.4f}-{val_cer:.4f}-{val_wer:.4f}",
    save_top_k=-1 # all model save
)
checkpoint_callback.CHECKPOINT_EQUALS_CHAR = '-'

callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
model = WhisperModelModule(cfg, model_name, lang, train_pairs, eval_pairs)

trainer = Trainer(
    # precision=16,
    accelerator="gpu",
    devices=[0],
    max_epochs=cfg.num_train_epochs,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    logger=tflogger,
    callbacks=callback_list
)

trainer.fit(model)

print("Training finished")
