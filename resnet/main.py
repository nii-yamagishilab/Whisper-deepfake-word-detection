#!/usr/bin/env python

import os
import sys
import random
import json
import math
import logging
import argparse
import pickle
from pathlib import Path
from typing import Union, Optional
from types import SimpleNamespace
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as at

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from hyperpyyaml import load_hyperpyyaml
import torchmetrics

from transformers import (
    get_linear_schedule_with_warmup
)

import dataio
import evaluation_seg
import evaluation
from utils import misc
from utils import feat as util_feat
from resnet import ResNet152

SEED = 3407
pl.seed_everything(SEED, workers=True)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
checkpoint_name = 'checkpoint'

def return_folder_name(cfg_name):
    project_dir = 'project/cp_resnet_{:s}'.format(cfg_name)
    log_output_dir = "{:s}/logs".format(project_dir)
    check_output_dir = "{:s}/artifacts".format(project_dir)
    cp_dir = "{:s}/checkpoint".format(check_output_dir)
    output_dir = "{:s}/outputs".format(check_output_dir)
    return project_dir, log_output_dir, cp_dir, output_dir


class FocalLoss(nn.Module):
    """Focal Loss for frame-wise classification."""
    def __init__(self, alpha=0.9, gamma=2.0, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
    
    def forward(self, inputs, targets):
        # inputs: (B, T, 2), targets: (B, T)
        probs = F.softmax(inputs, dim=-1)  # (B, T, 2)
        targets = targets.long()
        class_mask = torch.zeros_like(inputs).scatter_(2, targets.unsqueeze(-1), 1)
        probs = (probs * class_mask).sum(-1)  # (B, T)
        log_p = probs.log()
        loss = -self.alpha * (1 - probs).pow(self.gamma) * log_p
        return loss.mean() if self.size_average else loss.sum()


class ClassificationHead(nn.Module):
    """Frame-wise classification head: 256 -> 256 -> 2 (real/fake logits)."""
    def __init__(self, in_channels=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Binary: real (0), fake (1)
        )
    
    def forward(self, x):
        return self.layers(x)  # (B, T, 2)

class RegressionHead(nn.Module):
    """Frame-wise regression head: 256 -> 256 -> 2 (start/end offsets)."""
    def __init__(self, in_channels=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # Start/end offsets
            nn.ReLU()  # Ensure non-negative offsets
        )
    
    def forward(self, x):
        return self.layers(x)  # (B, T, 2)

class ResNetLocalization(nn.Module):
    """ResNet-152 for deepfake localization (Task 2)."""
    def __init__(self, in_channels=80, out_channels=256):
        super().__init__()
        self.feature_extractor = util_feat.MFB(
            sample_rate=16000, window_size=0.04, hop_size=0.02, n_mels=80
        )
        self.resnet = ResNet152(80, 256, pooling_func='TSTP', two_emb_layer=False)
        self.cls_head = ClassificationHead(out_channels)
        # self.reg_head = RegressionHead(out_channels)
    
    def forward(self, x, mask=None):
        # x: (B, T_samples), waveform
        # mask: (B, 1, T_frames) for padded frames, None if not needed
        x = self.feature_extractor(x)  # (B, 80, T_frames)
        _, features = self.resnet(x)  # (B, T_frames, 256)
        cls_scores = self.cls_head(features)  # (B, T_frames, 2)
        return cls_scores


class ResNetSpoof(pl.LightningModule):
    def __init__(self, cfg, resnet_model, train_dataset=[], eval_dataset=[], use_focal=False):
        super().__init__()
        self.save_hyperparameters(ignore=['resnet_model'])
        
        # Your ResNet152 + classification head
        self.model = resnet_model  
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self.cfg = cfg

        # loss cross entropy weights (default
        self.w_voc_token = cfg.weight_vocoded_token  if hasattr(cfg, 'weight_vocoded_token') else 0.9
        self.w_oth_token = cfg.weight_other_token  if hasattr(cfg, 'weight_other_token') else 0.1
        
        # Metrics
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([self.w_oth_token, self.w_voc_token]))
        logger.info(f"weight of vocoded {self.w_voc_token}")
        logger.info(f"weight of non-vocoded {self.w_oth_token}")
        
    def forward(self, x):
        return self.model(x)  # logits
    
    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self._train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )

    def train_dataloader(self):
        dataset = dataio.TedXSpeechDataset(self._train_dataset, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size,  
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=self._collate_fn
                          )
    
    def val_dataloader(self):
        dataset = dataio.TedXSpeechDataset(self._eval_dataset, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.cfg.batch_size, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=self._collate_fn
                          )

    def _collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.nn.utils.rnn.pad_sequence(
            [tensor.squeeze() for tensor in x], batch_first=True, padding_value=0.0
        )

        y = [torch.tensor(label, dtype=torch.long).squeeze() for label in y]  # list of 1D tensors
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0.0) 
        return x, y


    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        out = self.model(input_ids)

        # mininum length of data
        cnt_length = min([out.shape[1], labels.shape[1]])
        n_target = out.shape[-1]

        # flatten the temporal and batch dimensions
        loss = self.loss_fn(out[:, :cnt_length].view(-1, n_target), labels[:, :cnt_length].view(-1))
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        out = self.model(input_ids)
        
        # mininum length of data
        cnt_length = min([out.shape[1], labels.shape[1]])
        n_target = out.shape[-1]

        # flatten the temporal and batch dimensions
        loss = self.loss_fn(out[:, :cnt_length].view(-1, n_target), labels[:, :cnt_length].view(-1))
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        # return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        input_ids, labels = batch
        out = self.model(input_ids)
        # out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out, labels)
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch  # x: waveform, y: list of frame-wise or word-wise labels

        # Forward pass
        preds = self.model(x)  # (B, T, 2) if frame-level; adapt if word-level

        # Convert y from list of lists to padded tensor
        y = [torch.tensor(label, dtype=torch.long, device=preds.device) for label in y]
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-1)  # padding -1 to ignore

        # Convert predictions to class indices
        #pred_classes = preds.argmax(dim=-1)  # 0: bona fide, 1: spoofed
        y = y.squeeze(1)

        return {"preds": preds.cpu().numpy(), "labels": y.cpu().numpy()}

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
        logger.info(f"Label: {train_pairs[0][2][:50]}...")
    
    project_dir, log_output_dir, cp_dir, _ = return_folder_name(cfg_name)    
    Path(log_output_dir).mkdir(parents=True, exist_ok=True)
    Path(cp_dir).mkdir(parents=True, exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=cfg.train_name,
        version=cfg.train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cp_dir,
        filename=checkpoint_name + "-{epoch:04d}-{val_loss:.4f}",
        save_top_k=-1 # all model save
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = '-'
    
    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    
    model_resnet = ResNetLocalization(80, 256)
    model = ResNetSpoof(cfg, model_resnet, train_pairs, dev_pairs)
    
    trainer = Trainer(
        # precision=16,
        accelerator="gpu",
        devices=[0],
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list
    )

    # find checkpoint and resume training
    ckpts = glob.glob(os.path.join(cp_dir, "*.ckpt"))
    if len(ckpts):
        latest = max(ckpts, key=os.path.getmtime)
        trainer.fit(model, ckpt_path=latest)
    else:
        trainer.fit(model)
    return
    
@torch.no_grad()
def inference(cfg, cfg_name):
    """
    Lightning-native inference using trainer.predict().
    Loads model checkpoint, runs inference on all processed data,
    and saves spoofing scores per audio file.
    """

    eval_set_name = cfg.eval_set_name if hasattr(cfg, 'eval_set_name') else ''
    
    # ---------------------------
    # 0. find checkpoint
    # ---------------------------

    #checkpoint_path = cfg.ckpt_path
    #logger.info(f"✅ Using checkpoint: {checkpoint_path}")
    project_dir, _, cp_dir, save_dir = return_folder_name(cfg_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
        
    # without lora
    if hasattr(cfg, 'checkpoint') and cfg.checkpoint is not None and os.path.isfile(cfg.checkpoint):
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

    save_output = Path(save_dir) / '{:s}.inference.{:s}.score.pkl'.format(checkpoint, eval_set_name)
    if save_output.is_file():
        logger.info("Inference has been done: {:s}".format(str(save_output)))
        return
    
    # ---------------------------
    # 1. Process dataset
    # ---------------------------
    try:
        eval_list = cfg.eval_list
        data_dir = cfg.data_base_dir
    except AttributeError:
        logger.error("Missing eval_list or data_base_dir in yaml")
        sys.exit(1)
    
    eval_pairs = dataio.process_dataset(eval_list, data_dir, sample_rate = cfg.sample_rate, inference=True)
    logger.info(f"EVAL DATASET SIZE: {len(eval_pairs)} from {eval_list}")
    
    if len(eval_pairs) > 0:
        logger.info("Sample eval item:")
        logger.info(f"ID: {eval_pairs[0][0]}")
        logger.info(f"Audio: {eval_pairs[0][1]}")
        logger.info(f"Lebel: {eval_pairs[0][2][:50]}...")
    
    # ---------------------------
    # 2. Build dataset & dataloader
    # ---------------------------
    dataset = dataio.TedXSpeechDataset(eval_pairs, cfg.sample_rate)
    infer_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # inference per file
        num_workers=cfg.num_worker,
        collate_fn=lambda b: (
            torch.nn.utils.rnn.pad_sequence(
                [x.squeeze() for x, _ in b], batch_first=True
            ),
            [y for _, y in b]
        )
    )

    # ---------------------------
    # 3. Load model checkpoint
    # ---------------------------
    logger.info(f"✅ Using checkpoint: {checkpoint_path}")
    
    resnet_model = ResNetLocalization(80, 256)
    model = ResNetSpoof.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg,
        resnet_model=resnet_model,
        train_dataset=[],
        eval_dataset=[]
    )

    model.eval()
    model.freeze()

    # ---------------------------
    # 4. Run prediction
    # ---------------------------
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )

    logger.info("\n🚀 Running inference...")
    predictions = trainer.predict(model, dataloaders=infer_dataloader)


    # ---------------------------
    # 5. Save results
    # ---------------------------
    # results_path = Path(f"project/{cfg_name}/inference_results.csv")
    # with open(results_path, "w") as f:
    #     f.write("file,score\n")
    #     for (audio_id, _, _, _), score in zip(all_audio_transcript_pairs, predictions):
    #         f.write(f"{audio_id},{score:.6f}\n")
    
    #save_output = Path(save_dir) / '{:s}.inference.score.pkl'.format(checkpoint)
    save_output = Path(save_dir) / '{:s}.inference.{:s}.score.pkl'.format(checkpoint, eval_set_name)
    with open(save_output, 'wb') as file_ptr:
        pickle.dump(predictions, file_ptr)
    logger.info("Inference output saved to {:s}".format(str(save_output)))

    # evaluation
    if predictions[0]['labels'].size != 0:
        save_output_seg = evaluation_seg.main(str(save_output), eval_list, data_dir)
        evaluation.main(save_output_seg)
    else:
        logger.info("Skip final evaluation since there is no ground-truth label")
    return

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
