# Dependency

conda env create -f requirements.yaml

TODO: dependency installation for LoRA

# Usage

1. Prepare input via hparams/*.yaml
   * `trn_list`: list of relative path to json file of training set
   * `val_list`: list of relative path to json file of dev set
   * `eval_list`: list of relative path to json file of evaluation set
   * `data_base_dir`: path to the root base directory
   * `${data_base_dir}/relative_path` will be the full path to json file

2. TODO: the full path to audio file is currently inferred from the path to json file. The code is in `dataio.py:replace_json_dir_to_wav`

3. Training using `python main.py hparams/<name>.yaml train`

4. Testing using `python main.py hparams/<name>.yaml inference`. The latest checkpoint from the training process will be used. The reference and hypothesis (from the model) will be saved as a pkl file into the output folder.

5. Compute WER/CER and other metrics `python evaluation.py <path_to_pkl>`, where `<path_to_pkl>` is the path to the pkl file saved from step 4




