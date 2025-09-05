#!/usr/bin/env python
import os
import sys
import glob
import pickle
from pathlib import Path
import evaluate
import jiwer

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

def compute_detection_metrics(refs, preds, old_method=False):
    a = b = c = d = 0

    for ref_text, hyp_text in zip(refs, preds):
        ref_labels = extract_vocoding_labels(ref_text)
        hyp_labels = extract_vocoding_labels(hyp_text)


        # find the alignment
        ref_text = prepro(' '.join(x[1] for x in ref_labels))
        hyp_text = prepro(' '.join(x[1] for x in hyp_labels))
        align = jiwer.process_words([ref_text], [hyp_text]).alignments[0]

        
        if old_method:
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
        else:
            for chunk in align:
                ref_s, ref_e = chunk.ref_start_idx, chunk.ref_end_idx
                hyp_s, hyp_e = chunk.hyp_start_idx, chunk.hyp_end_idx
                
                if chunk.type == 'equal' or chunk.type == 'substitute':
                    # equal or substitute (aligned), then count directly
                    for ref_i, hyp_i in zip(range(ref_s, ref_e), range(hyp_s, hyp_e)):
                        gt, _ = ref_labels[ref_i]
                        pred, _ = hyp_labels[hyp_i]
                        if gt == "bonafide" and pred == "bonafide":
                            a += 1
                        elif gt == "spoof" and pred == "spoof":
                            b += 1
                        elif gt == "bonafide" and pred == "spoof":
                            c += 1  # false negative
                        elif gt == "spoof" and pred == "bonafide":
                            d += 1  # false positive
                        else:
                            assert 1==0, "Impossible"
                elif chunk.type == 'delete':
                    # prediction has no, reference has
                    for ref_i in range(ref_s, ref_e):
                        gt, _ = ref_labels[ref_i]
                        if gt == "bonafide":
                            # it is correct, the model does not predict spoof
                            a += 1
                        elif gt == "spoof":
                            d += 1  # false positive
                        else:
                            assert 1==0, "Impossible"
                elif chunk.type == 'insert':
                    for hyp_i in range(hyp_s, hyp_e):
                        pred, _ = hyp_labels[hyp_i]
                        if pred == "spoof":
                            # it is false, the model predict spoof
                            c += 1
                        elif pred == "bonafide":
                            a += 1
                        else:
                            assert 1==0, "Impossible"
                else:
                    assert 1==0, "Impossible"

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


def count_phrases_and_words(texts):
    num_phrases = len(texts)
    num_words = 0
    for t in texts:
        words = t.strip().split()
        # Don't count the '!!!!!!' markers as words
        num_words += sum(1 for w in words if w != vocoding_label)
    return num_phrases, num_words

def prepro(text):
    # to lower case
    # strip  
    return text.lower().strip()


if __name__ == "__main__":
    
    # 
    input_file = sys.argv[1]
    
    if input_file.endswith('pkl'):
        # pickle file
        with open(input_file, 'rb') as file_ptr:
            data = pickle.load(file_ptr)

    assert len(data) == 4, "The input is supposed to be [res, res_raw, refs, refs_raw]"

    

    res, res_raw, refs, refs_raw = data[0], data[1], data[2], data[3]
    # -----------------------------
    # Compute CER
    # -----------------------------
    cer_metrics = evaluate.load("cer")
    cer = cer_metrics.compute(references=[prepro(x) for x in refs_raw], 
                              predictions=[prepro(x) for x in res_raw])
    print(f"CER: {cer:.4f}")

    # WER
    wer_metrics = evaluate.load("wer")
    wer = wer_metrics.compute(references=[prepro(x) for x in refs_raw], 
                              predictions=[prepro(x) for x in res_raw])
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


    # Count in references
    num_phrases_ref, num_words_ref = count_phrases_and_words(refs)

    # Count in predictions
    num_phrases_pred, num_words_pred = count_phrases_and_words(res)

    print(f"References:  {num_phrases_ref} phrases, {num_words_ref} words")
    print(f"Predictions: {num_phrases_pred} phrases, {num_words_pred} words")

