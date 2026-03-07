
[![arXiv](https://img.shields.io/badge/arXiv-2507.08530-b31b1b.svg)](https://arxiv.org/abs/2602.22658)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)


This is the repository for paper [Deepfake Word Detection by Next-token Prediction using Fine-tuned Whisper](https://arxiv.org/abs/2602.22658)

```bibtex
Deepfake Word Detection by Next-token Prediction using Fine-tuned Whisper
Hoan My Tran, Xin Wang, Wanying Ge, Xuechen Liu, Junichi Yamagishi

@misc{tran2026deepfakeworddetectionnexttoken,
      title={Deepfake Word Detection by Next-token Prediction using Fine-tuned Whisper}, 
      author={Hoan My Tran and Xin Wang and Wanying Ge and Xuechen Liu and Junichi Yamagishi},
      year={2026},
      eprint={2602.22658},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2602.22658}, 
}
```

# Dependency

Install the tools via conda
```bash
conda env create -f requirements.yaml
```

Download the tools, in case that the GPU node cannot access Huggingface website during training / inference
```bash
cd whisper-tuned
git clone https://huggingface.co/spaces/evaluate-metric/wer
git clone https://huggingface.co/spaces/evaluate-metric/cer
cd ../resnet
git clone https://huggingface.co/spaces/evaluate-metric/wer
git clone https://huggingface.co/spaces/evaluate-metric/cer
```

- [ ] dependency installation for LoRA. This is not used for further experiments

# Folder structure

```bash
.
├── data
│   ├── tiny: a toy data set for demonstration
│
├── whisper-tuned: folder for fine-tuning whisper
│   ├── hparams: configuration YAML
│   ├── scripts: wrapper bash script 
│   ├── project: (to be produced after training) project folder
│   ├── utils: utility tools
│   ├── main.py: main function (model def, training loop)
│   ├── evaluation.py: tool to compute FPR/FRR and WER/CER
│   └── dataio.py: dataset definition, pre-processing (add token)
├── resnet: folder for resnet
├── README_YAML: document on the YAML file
├── requirements.yaml: python dependency
└── README.md
```


# Usage (toy example)

## Command 
```bash
cd whisper-tuned
bash scripts/tiny.sh
```

## What does the command do

This code will use the tiny dataset (one sample, repeated 16 times) to demonstrate the training and inference process.

1. Load `hparams/tiny.yaml`, which specifies
   * `data_base_dir`: path to the root base directory
   * `trn_list`: list of relative path to json file of training set
   * `val_list`: list of relative path to json file of dev set
   * `eval_list`: list of relative path to json file of evaluation set
   * The actual path to the file will be `data_base_dir/relative_path`
   * The file list is a two column csv file
     * 1st column is the json file
     * 2nd column is the wav file
     * see `data/tiny/tiny.lst` for example
   * See the `README_YAML.md` for details on yaml
 
2. Load model and do training

3. Load checkpoint and do inference
   * output is saved to `project/cp_whisper_<NAME_OF_CONFIG>`
   * output folder has structure
     * `artifacts/checkpoint`: folder to save checkpoint
     * `artifacts/outputs`: folder to save the inference output
   * The inference output is saved in `.pkl` format

4. Evaluation
   * For Whisper: main.py calls evaluation.py (or other tools) to compute detection errors and WER/CER
   * For ResNet: 
     * main.py calls evaluation_seg.py to count detect errors at the frame level (not used for paper). It also converts the frame-level scores to word scores and save to `txt2.pkl`
     * main.py then calls evaluation.py to evaluate `txt2.pkl` and produce the error rates on word level


To compute WER/CER and other metrics `python evaluation.py <path_to_pkl>`, where `<path_to_pkl>` is the path to the pkl file saved from inference.

## Sample log 

The log looks like

```bash
$ bash scripts/tiny.sh
##
# Training
##
[rank: 0] Seed set to 3407
INFO:__main__:User hparams/tiny.yaml
INFO:__main__:##################################################
INFO:__main__:Starting dataset processing in: ../data/tiny
INFO:__main__:##################################################
... 
INFO:__main__:TRAIN DATASET SIZE: 16
INFO:__main__:DEV DATASET SIZE: 16
INFO:__main__:Sample training item:
INFO:__main__:ID: o3d3yMxfioA_part_367_vocoded_replaced_3
INFO:__main__:Audio: ../data/tiny/o3d3yMxfioA_part_367_vocoded_replaced_3.flac
INFO:__main__:Text:  Versucht !!!!!! mittels !!!!!! stabilen !!!!!! Vo...
...
Epoch 1: 100%|█| 4/4 [00:13<00:00,  0.30it/s, v_num=e-05, train_loss=0.962, val_loss_step=0.902, val_ce
`Trainer.fit` stopped: `max_epochs=2` reached.     
Epoch 1: 100%|█| 4/4 [03:34<00:00,  0.02it/s, v_num=e-05, train_loss=0.962, val_loss_step=0.902, val_ce
INFO:__main__:Training finished
...

##
# Inference
##
...
INFO:__main__:Inference output saved to project/cp_whisper_hparams_tiny.yaml/artifacts/outputs/checkpoint-epoch-0001-val_loss-0.9019-val_cer-0.1091-val_wer-1.0000.ckpt.inference.tiny.txt.pkl
...

CER: 10.8108
WER: 100.0000

Vocoded Word Detection Results:
a_true_bonafide: 16
b_true_spoof: 0
c_false_negative: 0
d_false_positive: 48
accuracy: 25.0
FPR: 100.0
FNR: 0.0
--------------------
REF: Versucht!!!!!!mittels~~~!!!!!!stabilen~~~!!!!!!Vollkurven.~~~
HYP: Versucht, mittels, stabilen, Vollkurven,
--------------------
...
```

The above log shows the CER/WER and FRR/FNR on the tiny data set.

The last part shows the ground-truth (REF) and model output (HYP).


# How to use your own data

1. Follow `data/tiny` and prepare the json and wav files
2. Prepare the YAML file like `hparms/tiny.yaml`
3. Training `python main.py hparams/<yaml> train`
4. Inference & evaluation `python main.py hparams/<yaml> inference`. The best checkpoint (w.r.t error on validation set) will be loaded automatically 


Additional notes:
* `*/hparams/exp*.yaml` are the configuration files actually used in the paper.
* `*/scripts/qsub.sh` are the scripts to run training and inference on multiple test sets

# License

See [LICENSE](./LICENSE)

# Acknowledgement
This work is partially supported by JST, PRESTO Grant (JPMJPR23P9), and K Program Grant (JPMJKP24C2), Japan. It is partially done on TSUBAME4.0, Institute of Science Tokyo.

Contact: wangxin nii ac jp