# Dependency

conda env create -f requirements.yaml

TODO: dependency installation for LoRA

# Usage demonstration

1. Change `data_base_dir` in hparams/tiny.yaml to the full path to $PWD/data/tiny

2. Training `python main.py hparams/tiny.yaml train`

3. Inference `python main.py hparams/tiny.yaml inference`. By default, it will load the checkpoint with the lowest loss on dev set. It will save the Whisper output to a pkl file and compute the WER/CER by calling evaluation.py.

4. You may also specify the path to the checkpoint (.pt) `python main.py hparams/tiny.yaml inference --checkpoint <PATH_TO_CHECKPOINT>`
   
This code will use the tiny dataset (one sample) to demonstrate the training and inference process.

To use your down data, please prepare the data and modify hparams/*.yaml
   * `trn_list`: list of relative path to json file of training set
   * `val_list`: list of relative path to json file of dev set
   * `eval_list`: list of relative path to json file of evaluation set
   * `data_base_dir`: path to the root base directory

To compute WER/CER and other metrics `python evaluation.py <path_to_pkl>`, where `<path_to_pkl>` is the path to the pkl file saved from inference.

