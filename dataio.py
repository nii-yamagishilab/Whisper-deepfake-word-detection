#!/usr/bin/env python
"""
Code to fine-tune whisper

"""
import os
import sys
import json
import glob
import random
import logging

from types import SimpleNamespace

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from utils import rawboost

import torch
import torchaudio
import torchaudio.transforms as at
import whisper

from evaluation import vocoding_label

default_vocoded_tag = ''
logger = logging.getLogger(__name__)

# Audio loading function
def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
    """waveform = load_wave(wav_path, sample_rate)
    input: wav_path, str, waveform path
    input: sample_rate, int, expected sampling rate

    output: waveform, torch.tensor
    """
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

# Function to extract text from JSON structure
def extract_transcription(data):
    """Extracts transcription text from JSON structure
    
    """
    # Check if segments exist and have text content
    if 'segments' in data and isinstance(data['segments'], list) and len(data['segments']) > 0:
        # Combine all segment texts (for multi-segment files)
        return ''.join(segment.get('text', '') for segment in data['segments'])
    return ''

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)        
    # Extract text from JSON structure
    text = extract_transcription(data)
    return text


# Functions to process text transcription
def modify_vocoded_tag(text, pad_mode):
    
    # print(text)
    if pad_mode == 1:
        # only remove space at beginning 
        text = text.rstrip().lstrip()
    elif pad_mode == 2:
        # remove every space before vocoded tag
        text = text.replace(' '+vocoding_label, vocoding_label)
    elif pad_mode == 3:
        # remove space after vocoded tag
        text = text.replace(vocoding_label+' ', vocoding_label)
    elif pad_mode == 4:
        # remove space around vocoded tag
        text = text.replace(' '+vocoding_label+' ', vocoding_label)
        
    return text

###
# dataset loader
###
    
def process_dataset(
        filelist_path, base_dir,
        text_max_length=1000, audio_max_sample_length=480000, sample_rate=16000):
    """
    input: filelist_path, str, path to file list (relative path to json)
    input: base_dir, str, path to base of data diretory
    
    output: audio_transcript_pair_list, list, list of (filename, audio_path, text transcription)
    """
    
    audio_transcript_pair_list = []
    
    # in case the filelist_path is a string separated by ,
    #json_path_list = []
    for filelist_path_ in filelist_path.split(','):
        pd_tmp = pd.read_csv(filelist_path_, names=['json', 'audio'])
        #json_path_list += pd_tmp['json'].to_list()

    logger.info(f"Processing {pd_tmp.shape[0]} JSON files")
    
    for index in tqdm(range(pd_tmp.shape[0])):
        
        row = pd_tmp.iloc[index]
        
        # Load JSON content
        json_path = Path(base_dir) / row['json']
        try:
            text = load_json(json_path)
        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {str(e)}")
            continue

        if not text:
            logger.warning(f"Missing text in {json_path}")
            continue
            
        if len(text) > text_max_length:
            logger.warning(f"Skipped (text too long: {len(text)} chars) for {json_path.stem}")
            continue

        
        # load audio
        audio_path = Path(base_dir) / row['audio']
        if not audio_path.exists():
            logger.warning(f"Missing audio: {audio_path}")
            continue
        
        try:
            audio_length = torchaudio.info(audio_path).num_frames
            
        except Exception as e:
            logger.warning(f"Error loading audio {audio_path}: {str(e)}")
            continue
        
        if audio_length > audio_max_sample_length:
            logger.warning(f"Skipped (audio too long: {audio_length} samples) for {json_path.stem}")
            continue
        
        # Add valid pair
        audio_transcript_pair_list.append((json_path.stem, str(audio_path), text))
    
    logger.info(f"Found {len(audio_transcript_pair_list)} valid pairs")
    return audio_transcript_pair_list
        
####
# dataset definition
####

class TedXSpeechDataset(torch.utils.data.Dataset):
    """Class definition for TedXSpeech Dataset
    """
    def __init__(self, audio_trans_pair_list, tokenizer, cfg, nmel, ps_tokens=[], inf_flag=False) -> None:
        super().__init__()

        # data list (returned by data IO)
        self.data_list = audio_trans_pair_list
        # sampling rate
        self.sample_rate = cfg.sample_rate
        # tokenizer
        self.tokenizer = tokenizer
        # number of Mel bins
        self.nmel = nmel

        # 
        self.ps_tokens = ps_tokens

        # trimming space
        self.trim_led_pad = cfg.trim_leading_pad if hasattr(cfg, 'trim_leading_pad') else 0
        
        # use augmentation 
        self.use_rawboost = cfg.use_rawboost if hasattr(cfg, 'use_rawboost') else False
        if inf_flag:
            self.use_rawboost = False
            
        if self.use_rawboost:
            self.rawboost_config = cfg.rawboost_config
        else:
            self.rawboost_config = None

        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, id):
        audio_id, audio_path, text = self.data_list[id]

        # load audio
        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())

        if self.use_rawboost and self.rawboost_config is not None:
            audio = rawboost.process_Rawboost_feature(
                audio, sr=self.sample_rate,
                args=self.rawboost_config,
                algo=self.rawboost_config['algo'])
            
        # mel
        mel = whisper.log_mel_spectrogram(audio, self.nmel)

        # text and tokenizing
        text = modify_vocoded_tag(text, self.trim_led_pad)
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]
        
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text
        }

