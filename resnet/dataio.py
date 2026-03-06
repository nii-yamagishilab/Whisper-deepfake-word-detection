#!/usr/bin/env python

import os
import sys
import random
import json
import math

from pathlib import Path
from typing import Union, Optional
from types import SimpleNamespace
import glob
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as at


logger = logging.getLogger(__name__)
default_vocoded_tag = "!!!!!!"

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


def get_label_length(wav_length, model_tag='ResNet152'):
    """feat_length = get_label_length(wav_length)
    input: wav_length, int, length of waveform
    input: model_tag, str, model name, ResNet152 by default
    output: feat_length, int, length after model processing
    """
    if model_tag == 'ResNet152':
        # feat_length after STFT & front-end
        # based on torch.stft document:
        # https://docs.pytorch.org/docs/stable/generated/torch.stft.html
        # not elegant to write a magic number here. 
        feat_length = 1 + wav_length // 320
        
        # due to ResNet152 Conv2D down-sampling
        # kensize = 3, stride = 2, padding 1. See eq in
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        func_down = lambda x: int(np.floor((x-1)/2+1))
        # down-sample three times
        feat_length = func_down(func_down(func_down(feat_length)))
        
        ## 10 because FB 80 dimensions -> downsample by 10 -> flatten to temporal dim
        feat_length *= 10
        
    else:
        raise NotImplementedError("Model_tag:{:s} not supported".format(model_tag))
    
    return feat_length


def lab_in_scale(labvec, shift):
    """ Convert segment-level labels to coarser temporal resolution.
    input:
    -------
      labvec: array, labels at base shift (0.01s).
      shift: int, frame shift factor (e.g., 4 for 0.04s).
    output:
    -------
      new_lab: array<int>, labels at coarser resolution ('0' for spoof, '1' for bona fide).
    """
    num_frames = math.ceil(len(labvec) / shift)
    new_lab = np.zeros(num_frames, dtype=int)
    for idx in np.arange(num_frames):
        st, et = int(idx * shift), int((idx + 1) * shift)
        new_lab[idx] = min(labvec[st:et])  # '0' for spoof
    return new_lab


def generate_labels_from_json(json_data, base_shift=0.01, target_shift=0.04):
    """
    Generate frame-wise labels from Whisper-style JSON.
    input:
    -------
      json_data: dict, Whisper JSON with word segments and replacements.
      base_shift: float, base frame shift (0.01s).
      target_shift: float, target frame shift (0.04s for ResNet).
    output:
    -------
      cls_labels: array<int>, frame-wise binary labels ('0' spoof, '1' bona fide).
    """
    # if no word_segments, fall back to segments
    word_segments = json_data.get('word_segments', json_data.get('segments', []))
    if not word_segments:
        raise ValueError("No word segments found in JSON.")

    # compute total duration
    total_duration = max(seg.get('end', 0) for seg in word_segments if 'end' in seg)
    num_frames = math.ceil(total_duration / target_shift)

    # base resolution labels
    base_num_frames = math.ceil(total_duration / base_shift)
    base_labels = np.ones(base_num_frames, dtype=int)

    # mark spoof regions from replacements
    for rep in json_data.get('replacements', []):
        start, end = rep['replacement_start'], rep['replacement_end']
        start_frame = int(start / base_shift)
        end_frame = min(int(end / base_shift) + 1, base_num_frames)
        base_labels[start_frame:end_frame] = 0

    # downsample to target resolution
    cls_labels = lab_in_scale(base_labels, shift=int(target_shift / base_shift))

    return cls_labels


# Function to extract text from JSON structure
def extract_transcription(data):
    """Extracts transcription text from JSON structure"""
    # Check if segments exist and have text content
    #if 'segments' in data and isinstance(data['segments'], list) and len(data['segments']) > 0:
    #    # Combine all segment texts (for multi-segment files)
    #    return ''.join(segment.get('text', '') for segment in data['segments'])

    if 'word_segments' in data and isinstance(data['word_segments'], list) \
       and len(data['word_segments']) > 0:

        # 
        spoof_words = data.get('replacements', [])
        
        temp = []
        for word_segment in data['word_segments']:

            flag_bona = True
            for spoof_word in spoof_words:
                if word_segment == spoof_word:
                    flag_bona = False
                    break

            if flag_bona is False:
                temp.append(default_vocoded_tag + word_segment['word'])
            else:
                temp.append(' ' + word_segment['word'])
        text = ''.join(temp)
        if text[0] == ' ':
            text = text[1:]
        return text
    
    return ''


def generate_label_tensor(
    json_data: Union[dict, str],
    waveform_length: Optional[int] = None,
    sr: Optional[int] = 16000,
):
    """
    Generate frame-wise label tensor from JSON annotations.

    Args:
        json_data (dict or str): JSON data or path to JSON file.
        waveform_duration (float, optional): Duration of waveform in seconds. 
            If None, inferred from the last word segment.
        time_resolution (float): Time per frame in seconds.
        return_one_hot (bool): Whether to return one-hot labels.

    Returns:
        labels: Tensor of shape (1, T_frames) with class indices (0: bonafide, 1: spoof)
        one_hot_labels (optional): Tensor of shape (1, T_frames, 2) if return_one_hot=True
    """
    # Load JSON if a file path is provided
    if isinstance(json_data, str):
        with open(json_data, 'r') as f:
            json_data = json.load(f)
    
    # Infer waveform duration if not provided
    if waveform_length is None:
        word_segments = json_data.get("word_segments", [])
        if not word_segments:
            raise ValueError("No word segments found and waveform_duration not provided.")
        waveform_duration = max([segment["end"] for segment in word_segments if 'end' in segment])
        waveform_length = int(np.ceil(waveform_duration * sr))
        
    # Compute number of frames (ceiling division)
    T_frames = get_label_length(waveform_length)
    timeresolution = waveform_length // T_frames
    
    # Initialize label tensor (1, T_frames) with 0 (bonafide)
    labels = torch.zeros(1, T_frames, dtype=torch.long)
    
    # Get replacement intervals (spoof segments)
    replacements = json_data.get("replacements", [])
    
    # Assign labels for spoof segments
    for replacement in replacements:
        start_time = replacement["replacement_start"]
        end_time = replacement["replacement_end"]
        start_frame = int(start_time * sr // timeresolution)
        end_frame = int(end_time * sr // timeresolution)
        labels[0, start_frame : end_frame + 1] = 1.0
        #for frame_idx in range(start_frame, min(end_frame + 1, T_frames)):
        #    labels[0, frame_idx] = 1
    
    return labels, timeresolution

def generate_label_tensor_fixed_frame(
    json_data: Union[dict, str],
    waveform_length: Optional[int] = None,
    sr: Optional[int] = 16000,
    frame_shift_s: float = 0.16,
):
    """
    Generate frame-wise label tensor from JSON annotations
    using fixed frame duration (frame_shift_s).
    
    Args:
        json_data: dict or path to JSON file
        waveform_length: number of samples in the waveform
        sr: sample rate
        frame_shift_s: duration of each frame in seconds
    
    Returns:
        labels: Tensor of shape (1, T_frames) with 0=bonafide, 1=spoof
    """
    if isinstance(json_data, str):
        with open(json_data, 'r') as f:
            json_data = json.load(f)

    if waveform_length is None:
        word_segments = json_data.get("word_segments", [])
        if not word_segments:
            raise ValueError("No word segments and waveform_length not provided")
        waveform_length = int(np.ceil(max(seg["end"] for seg in word_segments) * sr))

    # Compute number of frames with fixed frame shift
    frame_shift_samples = int(frame_shift_s * sr)
    T_frames = int(np.ceil(waveform_length / frame_shift_samples))
    
    labels = torch.zeros(1, T_frames, dtype=torch.long)

    # Get spoof segments
    for rep in json_data.get("replacements", []):
        start_frame = int(rep["replacement_start"] * sr // frame_shift_samples)
        end_frame = int(rep["replacement_end"] * sr // frame_shift_samples)
        labels[0, start_frame:end_frame+1] = 1

    return labels

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)        
    # Extract text from JSON structure
    text = extract_transcription(data)
    return text, data


###
# dataset loader
###

def process_dataset(filelist_path, base_dir,
                    text_max_length=1000, audio_max_sample_length=480000,
                    sample_rate=16000, inference=False):
    """
    input: filelist_path, str, path to file list (relative path to json)
    input: base_dir, str, path to base of data diretory
    
    output: audio_transcript_pair_list, list, list of (filename, audio_path, text transcription)
    """
    
    audio_transcript_pair_list = []

    # in case the filelist_path is a string separated by ,
    pd_tmp = []
    for filelist_path_ in filelist_path.split(','):
        pd_tmp.append(pd.read_csv(filelist_path_, names=['json', 'audio']))
    pd_tmp = pd.concat(pd_tmp, axis=0)
    
    logger.info(f"Processing {pd_tmp.shape[0]} JSON files")
    
    for idx in tqdm(range(pd_tmp.shape[0])):

        row = pd_tmp.iloc[idx]
        
        # Load JSON content
        json_path = Path(base_dir) / row['json']

        if inference:
            if json_path.is_file():
                try:
                    text, json_data = load_json(json_path)
                except Exception as e:
                    text = ''
                    json_data = {}
            else:
                text = ''
                json_data = {}
        else:
            try:
                text, json_data = load_json(json_path)
            except Exception as e:
                logger.error(f"Error loading JSON {json_path}: {str(e)}")
                continue
        
            if not text:
                logger.warning(f"Missing text in {json_path}")
                continue
            
            if len(text) > text_max_length:
                logger.warning(f"Skipped (text too long: {len(text)} chars) for {json_path.stem}")
                continue

            
        # Construct audio path
        audio_path = Path(base_dir) / row['audio']
        if not audio_path.exists():
            logger.warning(f"Missing audio: {audio_path}")
            continue

        # Load and validate audio
        try:
            audio_length = torchaudio.info(audio_path).num_frames
                                
        except Exception as e:
            logger.warning(f"Error loading audio {audio_path}: {str(e)}")
            continue

        if audio_length > audio_max_sample_length:
            logger.warning(f"Skipped (audio too long: {audio_length} samples) for {json_path.stem}")
            continue

        if inference and len(json_data) == 0:

            cls_labels = np.array([])
            reso = 0
        else:
            try:
                # cls_labels = generate_labels_from_json(
                #     data, base_shift=base_shift, target_shift=target_shift
                # )
                cls_labels, reso = generate_label_tensor(json_data, audio_length)
            
            except Exception as e:
                logger.warning(f"Error generating labels for {json_path}: {str(e)}")
                continue
        
        # Add valid pair
        audio_transcript_pair_list.append(
            (json_path.stem, json_path, str(audio_path), cls_labels.tolist(), reso)
        )
        
    logger.info(f"Found {len(audio_transcript_pair_list)} valid pairs")
    return audio_transcript_pair_list


####
# dataset definition
####

class TedXSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_trans_pair_list, sample_rate) -> None:
        super().__init__()

        self.data_list = audio_trans_pair_list
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, id):
        audio_id, _, audio_path, labels, _ = self.data_list[id]
        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        return audio, labels
