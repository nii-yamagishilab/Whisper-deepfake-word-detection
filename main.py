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
import logging
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torchaudio
import torchaudio.transforms as at

import numpy as np
import pandas as pd

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
import dataio
from utils import misc

SEED = 3407
pl.seed_everything(SEED, workers=True)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
checkpoint_name = 'checkpoint'
lora_cp_name = 'lora.ckpt'

def return_folder_name(cfg_name, model_name, learning_rate):
    #project_dir = 'project/cp_whisper_{:s}_lr_{:3.0e}'.format(model_name, learning_rate)
    project_dir = 'project/cp_whisper_{:s}'.format(cfg_name)
    log_output_dir = "{:s}/logs".format(project_dir)
    check_output_dir = "{:s}/artifacts".format(project_dir)
    cp_dir = "{:s}/checkpoint".format(check_output_dir)
    output_dir = "{:s}/outputs".format(check_output_dir)
    return project_dir, log_output_dir, cp_dir, output_dir


def remove_special_token(tiktoken_enc, tokens):
    return [x for x in tokens if x not in tiktoken_enc._special_tokens.values()]

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

        # 50257 is eot token id
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) \
                  for lab, lab_len in zip(labels, label_lengths)]
        
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) \
                         for e, e_len in zip(dec_input_ids, dec_input_ids_length)] 
        
        batch = {
            "labels": labels,
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
                 ps_tokens=[220, 50199],
                 inference_flag=False) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)

        if hasattr(cfg, 'lora') and cfg.lora:
            lora_r = cfg.lora_r if hasattr(cfg, 'lora_r') else 8
            lora_alpha = cfg.lora_alpha if hasattr(cfg, 'lora_alpha') else 8
            lora_dropout = cfg.lora_dropout if hasattr(cfg, 'lora_dropout') else 0.0

            # inference, we merge the LoRA weights back to the original weights
            self.model = whisper.load_lora_model(
                model_name, lora_r, lora_alpha, lora_dropout, merge_weights = inference_flag)
        else:
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
        dataset = dataio.TedXSpeechDataset(self.__train_dataset, self.tokenizer,
                                    self.cfg, self.nmel)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        dataset = dataio.TedXSpeechDataset(self.__eval_dataset, self.tokenizer,
                                    self.cfg, self.nmel)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

def train(cfg, cfg_name):
    
    # Dataset setup - point to your base directory
    try:
        trn_list = cfg.trn_list
        val_list = cfg.val_list
        data_dir = cfg.data_base_dir
    except AttributeError:
        logger.error("Missing trn_list, val_list, or data_base_dir in yaml")
        sys.exit(1)

    logger.info(f"{'#'*50}")
    logger.info(f"Starting dataset processing in: {data_dir}")
    logger.info(f"{'#'*50}")

    # Process all datasets
    train_pairs = dataio.process_dataset(trn_list, data_dir, sample_rate = cfg.sample_rate)
    dev_pairs = dataio.process_dataset(val_list, data_dir, sample_rate = cfg.sample_rate)

    logger.info(f"TRAIN DATASET SIZE: {len(train_pairs)}")
    logger.info(f"DEV DATASET SIZE: {len(dev_pairs)}")
    
    # Sample output for verification
    if len(train_pairs) > 0:
        logger.info("Sample training item:")
        logger.info(f"ID: {train_pairs[0][0]}")
        logger.info(f"Audio: {train_pairs[0][1]}")
        logger.info(f"Text: {train_pairs[0][2][:50]}...")
        

    # whisper initialization
    woptions = whisper.DecodingOptions(language=cfg.lang, without_timestamps=True)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language=cfg.lang, task=woptions.task)
    
    model_name = cfg.model_name
    lang = cfg.lang
    project_dir, log_output_dir, cp_dir, _ = return_folder_name(cfg_name, model_name, cfg.learning_rate)
    
    train_name = "whisper_finetune_lr_{:3.0e}".format(cfg.learning_rate)
    train_id = "all_finetune_lr_{:3.0e}".format(cfg.learning_rate)

    Path(log_output_dir).mkdir(parents=True, exist_ok=True)
    Path(cp_dir).mkdir(parents=True, exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=train_id
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=cp_dir,
        filename=checkpoint_name + "-{epoch:04d}-{val_loss:.4f}-{val_cer:.4f}-{val_wer:.4f}",
        save_top_k=-1 # all model save
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = '-'

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    model = WhisperModelModule(cfg, model_name, lang, train_pairs, dev_pairs)

    # operation for lora
    if hasattr(cfg, 'lora') and cfg.lora:
        # follow https://github.com/microsoft/LoRA?tab=readme-ov-file
        # lora.mark_only_lora_as_trainable(model)
        whisper.set_lora_before_training(model.model)
    
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list
    )

    trainer.fit(model)
    logger.info("Training finished")

    if hasattr(cfg, 'lora') and cfg.lora:
        # in fact, no need to save model states, just need to save lora
        # but we keep the code compatible and save model states as well
        lora_cp_path = Path(cp_dir) / lora_cp_name
        whisper.save_lora_after_training(model.model, lora_cp_path)
    return


def inference(cfg, cfg_name):

    try:
        eval_list = cfg.eval_list
        data_dir = cfg.data_base_dir
    except AttributeError:
        logger.error("Missing eval_list or data_base_dir in yaml")
        sys.exit(1)

    eval_set_name = cfg.eval_set_name if hasattr(cfg, 'eval_set_name') else ''        
    eval_pairs = dataio.process_dataset(eval_list, data_dir, sample_rate = cfg.sample_rate)

    
    logger.info(f"EVAL DATASET SIZE: {len(eval_pairs)} from {eval_list}")
    
    if len(eval_pairs) > 0:
        logger.info("Sample eval item:")
        logger.info(f"ID: {eval_pairs[0][0]}")
        logger.info(f"Audio: {eval_pairs[0][1]}")
        logger.info(f"Text: {eval_pairs[0][2][:50]}...")

    woptions = whisper.DecodingOptions(language="fr", without_timestamps=True)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="fr", task=woptions.task)

    model_name = cfg.model_name
    lang = cfg.lang

    project_dir, _, cp_dir, save_dir = return_folder_name(cfg_name, model_name, cfg.learning_rate)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    #train_name = "whisper_finetune_lr_{:3.0e}".format(cfg.learning_rate)
    #train_id = "all_finetune_lr_{:3.0e}".format(cfg.learning_rate)

    # -----------------------------
    # 2. Load checkpoint and prepare model
    # -----------------------------
    # load pre-trained model
    whisper_model = WhisperModelModule(cfg, model_name, lang, inference_flag=True)
    
    if hasattr(cfg, 'lora') and cfg.lora:
        woptions = whisper.DecodingOptions(language=lang, without_timestamps=True, fp16=True)
        
        # if lora is on, no need to load checkpoint-epoch with whisper
        #whisper_model.load_state_dict(state_dict, strict=False)
        # load lora weights only
        lora_path = Path(cp_dir) / lora_cp_name
        lora = torch.load(lora_path)
        whisper_model.model.load_state_dict(lora, strict=False)

        # checkpoint, dummy for saving output
        checkpoint = lora_cp_name
        
    else:
        woptions = whisper.DecodingOptions(language=lang, without_timestamps=True)

        # without lora
        if hasattr(cfg, 'checkpoint') and os.path.isfile(cfg.checkpoint):
            # use specified checkpoint
            checkpoint_path = cfg.checkpoint
            checkpoint = os.path.basename(checkpoint_path)
        elif hasattr(cfg, 'checkpoint') and cfg.checkpoint == 'loss':
            # choose based on val loss
            checkpoint = sorted(glob.glob("{:s}*.ckpt".format(checkpoint_name), root_dir=cp_dir),
                                key=lambda x: x.split('-')[4])[0]
            checkpoint_path = "{:s}/{:s}".format(cp_dir, checkpoint)
        else:
            # choose last epoch
            checkpoint = sorted(glob.glob("{:s}*.ckpt".format(checkpoint_name), root_dir=cp_dir),
                                key=lambda x: x.split('-')[2])[-1]
            checkpoint_path = "{:s}/{:s}".format(cp_dir, checkpoint)
            
        logger.info("Use {:s}".format(checkpoint_path))
        state_dict = torch.load(checkpoint_path)
        state_dict = state_dict['state_dict']
        whisper_model.load_state_dict(state_dict)

    whisper_model.model.eval()
    dataset = dataio.TedXSpeechDataset(eval_pairs, wtokenizer, cfg, whisper_model.nmel, inf_flag=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding())
    
    # -----------------------------
    # 3. Run inference
    # -----------------------------
    refs = []
    res = []

    for b in tqdm(loader):

        input_ids = b["input_ids"].cuda()
        labels = b["labels"].long()

        # inference
        with torch.no_grad():
            # whisper decoding
            results = whisper_model.model.decode(input_ids, woptions)

            # save results
            for r in results:
                res.append(r.text)

            for l in labels:
                l[l == -100] = wtokenizer.eot
                ref = wtokenizer.decode(remove_special_token(wtokenizer.encoding, l))
                refs.append(ref)

    # save to output
    save_output = Path(save_dir) / '{:s}.inference.{:s}.txt.pkl'.format(checkpoint, eval_set_name)
    with open(save_output, 'wb') as file_ptr:
        pickle.dump([res, refs], file_ptr)
    logger.info("Inference output saved to {:s}".format(str(save_output)))

    # evaluate
    evaluation.main(str(save_output))
    
if __name__ == "__main__":
    
    # load configuration file
    cfg_name = sys.argv[1]
    logger.info("User {:s}".format(cfg_name))

    # parse the command line arguments
    parser = argparse.ArgumentParser()
    _, overrides = parser.parse_known_args(sys.argv[3:])
    overrides = misc.convert_to_yaml(overrides)
    
    # parse the config
    
    with open(cfg_name, encoding="utf-8") as fin:
        cfg = SimpleNamespace(**load_hyperpyyaml(fin, overrides))
    
    if sys.argv[2] == 'train':
        train(cfg, cfg_name.replace('/', '_'))
    else:
        inference(cfg, cfg_name.replace('/', '_'))
