# Dependency

conda env create -f requirements.yaml

# Usage

Assume data structure like this
```bash
ls ../data/tedx_test
tedx_fr_random_3k          tedx_fr_random_3k_vocoded_json           vtt
tedx_fr_random_3k_json     tedx_fr_random_3k_vocoded_json_original  vtt_generated
tedx_fr_random_3k_vocoded  tedx_fr_random_3k_vocoded_original
```

Change `data_base_dir` in `hparams/*.yaml`


Training using
```bash
python main.py hparams/<name>.yaml train
```

Inference using
```bash
python main.py hparams/<name>.yaml inference
```
The latest checkpoint from the training process will be used. The reference and hypothesis (from the model) will be saved as a pkl file into the output folder.

If you want to compute WER/CER and other metrics on the reference and hypothesis
```bash
python evaluation.py <path_to_pkl>
```

