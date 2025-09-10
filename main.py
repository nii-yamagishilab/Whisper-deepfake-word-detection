#!/usr/bin/env python
"""
Code to fine-tune whisper
"""

import os
import sys
import json
import glob
import random
import pickle
from types import SimpleNamespace

import torch
from torch import nn
import torchaudio
import torchaudio.transforms as at

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml

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
import evaluate
import evaluation


SEED = 3407
pl.seed_everything(SEED, workers=True)

def return_folder_name(cfg_name, model_name, learning_rate):
    #project_dir = 'project/cp_whisper_{:s}_lr_{:3.0e}'.format(model_name, learning_rate)
    project_dir = 'project/cp_whisper_{:s}'.format(cfg_name)
    log_output_dir = "{:s}/logs".format(project_dir)
    check_output_dir = "{:s}/artifacts".format(project_dir)
    cp_dir = "{:s}/checkpoint".format(check_output_dir)
    output_dir = "{:s}/outputs".format(check_output_dir)
    return project_dir, log_output_dir, cp_dir, output_dir


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

def remove_special_token(tiktoken_enc, tokens):
    return [x for x in tokens if x not in tiktoken_enc._special_tokens.values()]

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
    # fix the random seed
    pd_tmp = pd.DataFrame({'json': list(Path(json_dir).rglob("*.json"))}).sample(frac=1, random_state=SEED)
    json_path_list = pd_tmp['json'].to_list()
    
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
def process_all_datasets(base_dir, sample_rate, text_max_length=1000, audio_max_length=480000):
    all_pairs = []
    base_path = Path(base_dir)
    
    # Define dataset types and their configurations
    dataset_configs = [
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
            text_max_length=text_max_length,
            audio_max_sample_length=audio_max_length,
            sample_rate=sample_rate
        )
        all_pairs.extend(pairs)
    
    return all_pairs

class TedXSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, cfg, nmel, ps_tokens=[]) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = cfg.sample_rate
        self.tokenizer = tokenizer
        self.nmel = nmel
        
        # loss cross entropy weights
        self.w_voc_token = cfg.weight_vocoded_token if hasattr(cfg, 'weight_vocoded_token') else 1.0
        self.w_oth_token = cfg.weight_other_token if hasattr(cfg, 'weight_other_token') else 1.0
        self.ps_tokens = ps_tokens

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
        
        # weights for cross entropy loss
        mask = torch.ones_like(torch.tensor(labels)) * self.w_oth_token
        for ps_token in self.ps_tokens:
            mask[labels == ps_token] = self.w_voc_token
        
        # print(labels)
        # print("tooooooooooooooooooooooooooooooooooooooo")
        # print(self.tokenizer.decode(text))
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text,
            'mask': mask
        }

class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids, masks = [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            masks.append(f["mask"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        # 50257 is eot token id
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) \
                  for lab, lab_len in zip(labels, label_lengths)]
        masks = [np.pad(mask, (0, max_label_len - lab_len), 'constant', constant_values=0) \
                for mask, lab_len in zip(masks, label_lengths)]
        
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) \
                         for e, e_len in zip(dec_input_ids, dec_input_ids_length)] 
        
        batch = {
            "labels": labels,
            "masks": masks,
            "dec_input_ids": dec_input_ids
        }

        # input features
        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch


class WhisperModelModule(LightningModule):
    def __init__(self,
                 cfg,
                 model_name="base",
                 lang="fr",
                 train_dataset=[],
                 eval_dataset=[],
                 ps_tokens=[220, 50199]) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="fr", task=self.options.task)
        # flag of fine-tuning
        #  by default False
        self.finetune_encoder = cfg.finetune_encoder if hasattr(cfg, 'finetune_encoder') else False
        #  by default True
        self.finetune_decoder = cfg.finetune_decoder if hasattr(cfg, 'finetune_decoder') else True

        
        # number of mels
        self.nmel = self.model.dims.n_mels

        if not self.finetune_encoder:
            # not update encoder
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        if not self.finetune_decoder:
            # not update decoder
            for p in self.model.decoder.parameters():
                p.requires_grad = False
            
        # loss cross entropy weights
        self.w_voc_token = cfg.weight_vocoded_token  if hasattr(cfg, 'weight_vocoded_token') else 1.0
        self.w_oth_token = cfg.weight_other_token  if hasattr(cfg, 'weight_other_token') else 1.0
        
        weights = torch.ones(self.model.decoder.token_embedding.num_embeddings) * self.w_oth_token
        for ps_token in ps_tokens:
            weights[ps_token] = self.w_voc_token

        # just to be backward compatible
        #  otherwise, loss_fn.weight is required from the checkpoint
        if (not hasattr(cfg, 'weight_other_token')) and (not hasattr(cfg, 'weight_vocoded_token')):
            weights = None
        
        # cross entropy loss 
        self.loss_fn = nn.CrossEntropyLoss(weight = weights, ignore_index=-100)

        # loss for WER/CER
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        # 
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        masks = batch["masks"]
        dec_input_ids = batch["dec_input_ids"].long()

        # encoding
        #  whether fine-tune encoder
        if self.finetune_encoder:
            audio_features = self.model.encoder(input_ids)
        else:
            with torch.no_grad():
                audio_features = self.model.encoder(input_ids)

        # decoding
        out = self.model.decoder(dec_input_ids, audio_features)

        # compute loss
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

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
        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
    
    def train_dataloader(self):
        dataset = TedXSpeechDataset(self.__train_dataset, self.tokenizer,
                                    self.cfg, self.nmel)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        dataset = TedXSpeechDataset(self.__eval_dataset, self.tokenizer,
                                    self.cfg, self.nmel)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )



def train(cfg, cfg_name):
        # Dataset setup - point to your base directory
    data_dir = cfg.data_base_dir
    print(f"\n{'#'*50}")
    print(f"Starting dataset processing in: {data_dir}")
    print(f"{'#'*50}\n")

    # Process all datasets
    all_audio_transcript_pairs = process_all_datasets(data_dir, cfg.sample_rate)
    print(f"\nTOTAL AUDIO-TEXT PAIRS FOUND: {len(all_audio_transcript_pairs)}")

    # Split into train and eval
    train_num = int(len(all_audio_transcript_pairs) * cfg.train_ratio)
    train_pairs, eval_pairs = all_audio_transcript_pairs[:train_num], all_audio_transcript_pairs[train_num:]

    print(f"\nTRAIN DATASET SIZE: {len(train_pairs)}")
    print(f"EVAL DATASET SIZE: {len(eval_pairs)}")
    # Sample output for verification
    if len(train_pairs) > 0:
        print("\nSample training item:")
        print(f"ID: {train_pairs[0][0]}")
        print(f"Audio: {train_pairs[0][1]}")
        print(f"Text: {train_pairs[0][2][:50]}...")
        
    # (Add your Whisper model and training code below)

    woptions = whisper.DecodingOptions(language=cfg.lang, without_timestamps=True)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language=cfg.lang, task=woptions.task)
    
    model_name = cfg.model_name
    lang = cfg.lang

    project_dir, log_output_dir, cp_dir, _ = return_folder_name(cfg_name, model_name, cfg.learning_rate)
    
    train_name = "whisper_finetune_lr_{:3.0e}".format(cfg.learning_rate)
    train_id = "all_finetune_lr_{:3.0e}".format(cfg.learning_rate)

    # %%

    Path(log_output_dir).mkdir(parents=True, exist_ok=True)
    Path(cp_dir).mkdir(parents=True, exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=train_id
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=cp_dir,
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


def inference(cfg, cfg_name):
    # Process all datasets
    all_audio_transcript_pairs = process_all_datasets(cfg.data_base_dir, cfg.sample_rate)
    print(f"\nTOTAL AUDIO-TEXT PAIRS FOUND: {len(all_audio_transcript_pairs)}")

    # Split into train and eval
    train_num = int(len(all_audio_transcript_pairs) * cfg.train_ratio)
    _, eval_pairs = all_audio_transcript_pairs[:train_num], all_audio_transcript_pairs[train_num:]

    print(f"EVAL DATASET SIZE: {len(eval_pairs)}")
    
    if len(eval_pairs) > 0:
        print("\nSample eval item:")
        print(f"ID: {eval_pairs[0][0]}")
        print(f"Audio: {eval_pairs[0][1]}")
        print(f"Text: {eval_pairs[0][2][:50]}...")

    woptions = whisper.DecodingOptions(language="fr", without_timestamps=True)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="fr", task=woptions.task)

    model_name = cfg.model_name
    lang = cfg.lang

    project_dir, _, cp_dir, save_dir = return_folder_name(cfg_name, model_name, cfg.learning_rate)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    train_name = "whisper_finetune_lr_{:3.0e}".format(cfg.learning_rate)
    train_id = "all_finetune_lr_{:3.0e}".format(cfg.learning_rate)

    # -----------------------------
    # 2. Load checkpoint and prepare model
    # -----------------------------
    # sorted by epoch
    checkpoint = sorted(glob.glob("*.ckpt", root_dir=cp_dir), key=lambda x: x.split('-')[1])[-1]
    checkpoint_path = "{:s}/{:s}".format(cp_dir, checkpoint)
    print("Use {:s}".format(checkpoint_path))
    
    state_dict = torch.load(checkpoint_path)
    state_dict = state_dict['state_dict']

    whisper_model = WhisperModelModule(cfg, model_name, lang)
    whisper_model.load_state_dict(state_dict)

    woptions = whisper.DecodingOptions(language="fr", without_timestamps=True)

    dataset = TedXSpeechDataset(eval_pairs, wtokenizer, cfg, whisper_model.nmel)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())
    
    # -----------------------------
    # 3. Run inference
    # -----------------------------
    refs = []
    res = []

    for b in tqdm(loader):
        input_ids = b["input_ids"].half().cuda()
        labels = b["labels"].long()
        with torch.no_grad():
            results = whisper_model.model.decode(input_ids, woptions)
            for r in results:
                res.append(r.text)

            for l in labels:
                l[l == -100] = wtokenizer.eot
                ref = wtokenizer.decode(remove_special_token(wtokenizer.encoding, l))
                refs.append(ref)

    # save the output
    save_output = Path(save_dir) / '{:s}.inference.txt.pkl'.format(checkpoint)
    with open(save_output, 'wb') as file_ptr:
        pickle.dump([res, refs], file_ptr)
    print("Inference output saved to {:s}".format(str(save_output)))

    # evaluate
    evaluation.main(str(save_output))
    
if __name__ == "__main__":
    
    # load configuration file
    cfg_name = sys.argv[1]
    print("User {:s}".format(cfg_name))
    with open(cfg_name, encoding="utf-8") as fin:
        cfg = SimpleNamespace(**load_hyperpyyaml(fin))
    
    if sys.argv[2] == 'train':
        train(cfg, cfg_name.replace('/', '_'))
    else:
        inference(cfg, cfg_name.replace('/', '_'))
