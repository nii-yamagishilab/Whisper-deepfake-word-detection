# On YAML

## Dataset

| Parameter | Description |
|---|---|
| `data_base_dir` | Base directory containing dataset files. |
| `trn_list` | CSV file listing training samples (audio path and transcription). |
| `val_list` | CSV file for validation during training. |
| `eval_list` | CSV file used for final evaluation. |
| `eval_set_name` | Name of the evaluation dataset used in logs or result files. |

## Model

| Parameter | Description |
|---|---|
| `model_name` | Whisper model size. `large` is the largest multilingual model with the best accuracy but highest compute cost. |
| `lang` | Language token used by Whisper during decoding (e.g., `fr` for French). |

For fine-tuned model, the language seems to be not so important. The model will be fine-tuned to learn the language in the fine-tuning set.

During inference, lang will be inferred by the Whisper and used it do decoding.

## Training Hyperparameters

| Parameter | Description |
|---|---|
| `learning_rate` | Initial learning rate for optimization. Small values help preserve pretrained knowledge. |
| `weight_decay` | L2 regularization coefficient to reduce overfitting. |
| `adam_epsilon` | Numerical stability term in the Adam optimizer. |
| `warmup_steps` | Number of steps used to gradually increase the learning rate at the start of training. |

## Training Setup

| Parameter | Description |
|---|---|
| `batch_size` | Number of samples processed per training step per GPU. |
| `num_worker` | Number of data loader workers used for parallel data loading. |
| `num_train_epochs` | Number of passes through the training dataset. |
| `gradient_accumulation_steps` | Number of steps to accumulate gradients before updating model weights. |

Effective batch size = `batch_size × gradient_accumulation_steps`.

If the GPU requirement is too large, please decrease the batch size.

## Audio Processing

| Parameter | Description |
|---|---|
| `sample_rate` | Sampling rate expected by the model. Whisper operates on 16 kHz audio. |

All audio inputs should be resampled to this rate.


## Fine-Tuning Behavior

| Parameter | Description |
|---|---|
| `finetune_encoder` | Whether to update encoder parameters during training. If `false`, only the decoder is trained. |

## Token Loss Weighting

| Parameter | Description |
|---|---|
| `weight_vocoded_token` | Loss weight applied to tokens associated with vocoded or synthetic speech. |
| `weight_other_token` | Loss weight for all other tokens. |

Both are set to `1.0`, meaning all tokens contribute equally to the loss.

## Input Processing

| Parameter | Description |
|---|---|
| `trim_leading_pad` | Number of leading padding tokens removed before computing the loss to avoid unnecessary loss contributions. |

## Checkpoint Selection

| Parameter | Description |
|---|---|
| `checkpoint` | Metric used to select the best model checkpoint. `loss` means the checkpoint with the lowest validation loss is saved. |

