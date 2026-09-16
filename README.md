
[![arXiv](https://img.shields.io/badge/arXiv-2507.08530-b31b1b.svg)](https://arxiv.org/abs/2602.22658)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)


This is the repository for paper [Deepfake Word Detection by Next-token Prediction using Fine-tuned Whisper](https://arxiv.org/abs/2602.22658)

```bibtex
Deepfake Word Detection by Next-token Prediction using Fine-tuned Whisper
Hoan My Tran, Xin Wang, Wanying Ge, Xuechen Liu, Junichi Yamagishi, Interspeech 2026 (accepted)


@inproceedings{tranDeepfake2026,
	title = {Deepfake {Word} {Detection} by {Next}-token {Prediction} using {Fine}-tuned {Whisper}},
	url = {https://arxiv.org/abs/2602.22658},
	booktitle = {Proc. {Interspeech}},
	author = {Tran, Hoan My and Wang, Xin and Ge, Wanying and Liu, Xuechen and Yamagishi, Junichi},
	month = oct,
	year = {2026},
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


Dependency installation for LoRA is not necessary anymore. It is not used for further experiments in the paper.

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
   * `projdir_name`: optional name to save output (or load pre-trained)
     * By default, it will be `project/cp_whisper_<NAME_OF_CONFIG>`
   * See the `README_YAML.md` for details on yaml
 
2. Load model and do training

3. Load checkpoint and do inference
   * output is saved to `project/cp_whisper_<NAME_OF_CONFIG>`
   * it has sub-folders
     * `artifacts/checkpoint`: folder to save checkpoint
     * `artifacts/outputs`: folder to save the inference output
     * `logs`: training logs
   * The inference output is saved in `.pkl` format in `artifacts/outputs`

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
INFO:__main__:ID: hifigan_589_1692_000106
INFO:__main__:Audio: data/tiny/hifigan_589_1692_000106.wav
INFO:__main__:Text: Nahm der Oehi den Peter ein !!!!!! wenig auf die S...
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

CER: 0.0000                            
WER: 0.0000         
                                       
Vocoded Word Detection Results:        
a_true_bonafide: 624                   
b_true_spoof: 0                        
c_false_negative: 0                    
d_false_positive: 64            
accuracy: 90.69767441860465
FPR: 100.0
FNR: 0.0
--------------------
REF: Nahm der Oehi den Peter ein!!!!!!wenig~~~auf die Seite, damit dieser verstehen könne, was er ihm zu!!!!!!sagen~~~hatte, denn die Geisten meckerten immer, eine stärker als!!!!!!die~~~andere, vor lauter Freude und Freundschaftsbezeugungen, sobald sie das!!!!!!Heide~~~in ihrer Mitte hatten.
HYP: nahm der Oehi den Peter ein wenig auf die Seite, damit dieser verstehen könne, was er ihm zu sagen hatte,
 denn die Geisten meckerten immer, eine stärker als die andere, vor lauter Freude und Freundschaftsbezeugungen, sobald sie das Heide in ihrer Mitte hatten.        
--------------------
...
References:  16 phrases, 560 words
Predictions: 16 phrases, 688 words
```

Depending on the GPU and data I/O, it may take a few minutes to run on the tiny dataset.

The above log shows the CER/WER and FRR/FNR on the tiny data set. The last part shows the ground-truth (REF) and model output (HYP).

It is not surprising that the Whisper fine-tuned on tiny dataset cannot detect no synthetic word (FRP: 100.0, which is false acceptance rate). You may try our pre-trained Whisper (see the following section).

## Error at running:

`torch.cuda.OutOfMemoryError: CUDA out of memory.`: Please reduce `batch_size: 8` in `hyparams/tiny.yaml`



# Notes:

## How to load pre-trained checkpoint

* download a pretrained checkpoint (for example from [hg](https://huggingface.co/nii-yamagishilab/whisper-deepfake-word-detection))
* run inference by adding `--checkpoint_folder <PATH_TO_CHECKPOINT_FOLDER>` to the python command (see inference commande in `script/tiny.sh`)
    * note that `<PATH_TO_CHECKPOINT_FOLDER>` should be the path to the folder that contains checkpoint `*.ckpt`, not the path to the checkpoint itself.
    * the code will find and load the checkpoint
    * the inference code will say `Set checkpoint folder to ..., from which checkpoint is saved or loaded.`


## YAML and scripts

* `*/hparams/exp*.yaml` are the configuration files actually used in the paper.
  * `exp_voc_mls.yaml`: `FT.Voc`
  * `exp_voctts_mls.yaml`: `FT.V+T`
  * `full_x02_llamapartial_l.yam`: `FT.TTS`
  * The path to the data can be found on NII internal server
* `*/scripts/qsub.sh` are the scripts to run training and inference on multiple test sets.
* to use pre-trained Whisper, run the inference command using `whisper-tuned/utils/main_pretrained_whisper.py`. The commandline is the same as inference using `main.py`.

## How to use your own data

1. Follow `data/tiny` and prepare the json and wav files
2. Prepare the YAML file like `hparms/tiny.yaml`
3. Training `python main.py hparams/<yaml> train`
4. Inference & evaluation `python main.py hparams/<yaml> inference`. The best checkpoint (w.r.t error on validation set) will be loaded automatically 


# License

See [LICENSE](./LICENSE)

# Acknowledgement
This work is partially supported by JST, PRESTO Grant (JPMJPR23P9), Japan. It is partially done on TSUBAME4.0, Institute of Science Tokyo. Hoan My Tran was supported by an NII MOU internship program. 

Contact: wangxin nii ac jp
