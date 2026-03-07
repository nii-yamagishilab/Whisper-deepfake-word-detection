#!/usr/bin/env python
import os
import sys
import pickle
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torchaudio
import numpy as np
from scipy.special import softmax

import dataio

default_vocoded_tag = dataio.default_vocoded_tag

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def logits_to_score(score):
    # probability of spoof
    return softmax(score, axis=-1)[:, -1]

def decision_policy(score_array, threshold=0.5):
    # probability of spoof > 0.5
    average_score = np.mean(score_array)
    return average_score > threshold


def predicted_score_to_decision(score, threshold=0.5):
    
    # score to prob
    probs = softmax(score, axis=-1)[:, -1]
    
    decision = np.zeros_like(probs, dtype=np.int32)
    # score larger than threshold
    decision[probs > threshold] = 1
    return decision

def main(input_file, eval_list, base_dir, sr=16000):

    # load predicted scores
    if input_file.endswith('pkl'):
        # pickle file
        with open(input_file, 'rb') as file_ptr:
            predictions = pickle.load(file_ptr)
            
    all_preds_list = []
    all_labels_list = []

    for p in predictions:
        preds = p["preds"]  # [B, T]
        labels_list = p["labels"]  # list of length B, each [T_i]

        for i in range(len(labels_list)):
            true_len = len(labels_list[i])
            all_preds_list.append(preds[i, :true_len])
            all_labels_list.append(labels_list[i])

    # load csv file file
    eval_pairs = dataio.process_dataset(eval_list, base_dir, sample_rate = sr)

    # check
    assert len(eval_pairs) == len(all_preds_list), \
        'Prediction output not compatible with file list'

    refs = []
    res = []
    
    # create marked text string
    for all_preds, eval_pair in tqdm(zip(all_preds_list, eval_pairs), total=len(eval_pairs)):
        # 
        all_decision = logits_to_score(all_preds)

        # 
        # load json (see dataio.py)
        json_path = eval_pair[1]
        try:
            _, json_data = dataio.load_json(json_path)
        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {str(e)}")
            continue

        # uniform reso
        time_reso = eval_pair[4]

        #
        word_segments = json_data.get("word_segments", [])
        if not word_segments:
            raise ValueError("No word segments found and waveform_duration not provided.")

        text_buffer = []
        presymbol = ''

        ref_buffer = []
        ref_presymbol = ''
        
        for segment in word_segments:
            if 'end' in segment:
                start_t = segment['start'] * sr
                end_t = segment['end'] * sr
                start_idx = int(np.floor(start_t / time_reso))
                end_idx = int(np.ceil(end_t / time_reso))

                # check whether the decision contains spoof
                if start_idx >= len(all_decision):
                    pass
                
                if start_idx < len(all_decision) and decision_policy(all_decision[start_idx:end_idx]):
                    presymbol = default_vocoded_tag
                else:
                    presymbol = ' '

                # check groud-truth
                ref_presymbol = ' '
                for gt in json_data['replacements']:
                    # replacement segment overlap with the word?
                    # for some databases, the time stamp label may
                    # not be exactly the same
                    gt_overlap = min([segment['end'], gt['replacement_end']]) - \
                        max([segment['start'], gt['replacement_start']])
                                            
                    if segment['word'] == gt['word'] and gt_overlap > 0:
                        ref_presymbol = default_vocoded_tag
                        break
                    
            else:
                presymbol = ' '
                ref_presymbol = ' '

            # not adding space at the beginning
            if len(text_buffer) == 0 and presymbol == ' ':
                presymbol = ''
            text_buffer.append(presymbol + segment['word'])

            # not adding space at the beginning
            if len(ref_buffer) == 0 and ref_presymbol == ' ':
                presymbol = ''
            ref_buffer.append(ref_presymbol + segment['word'])
            
            
        hyp_text = ''.join(text_buffer)
        ref_text = ''.join(ref_buffer)
        #ref_text = dataio.extract_transcription(json_data)
        res.append(hyp_text)
        refs.append(ref_text)

    save_output = input_file + '.txt2.pkl'
    with open(save_output, 'wb') as file_ptr:
        pickle.dump([res, refs], file_ptr)
    logger.info("Save output to {:s}".format(save_output))
        
    # Concatenate all frames
    all_preds = np.concatenate(all_preds_list)
    all_decisions = predicted_score_to_decision(all_preds)
    all_labels = np.concatenate(all_labels_list)

    # Now compute confusion matrix
    a = ((all_decisions == 0) & (all_labels == 0)).sum().item()
    b = ((all_decisions == 1) & (all_labels == 1)).sum().item()
    c = ((all_decisions == 1) & (all_labels == 0)).sum().item()
    d = ((all_decisions == 0) & (all_labels == 1)).sum().item()

    accuracy = (a + b) / (a + b + c + d)
    false_positive_rate = d / (d + b) if (d + b) > 0 else 0
    false_negative_rate = c / (a + c) if (a + c) > 0 else 0
    accuracy *= 100
    false_positive_rate *= 100
    false_negative_rate *= 100
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"False positive rate: {false_positive_rate:.4f}")
    logger.info(f"False negative rate: {false_negative_rate:.4f}")
    return save_output

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
