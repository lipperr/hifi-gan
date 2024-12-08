# Vocoder with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a template for solving part of the TTS task with PyTorch. This template branch is a part of the [HSE DLA course](https://github.com/markovka17/dla) ASR homework. 

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw3_nv).

See wandb experiments with logged generated audio samples in the [task report](https://api.wandb.ai/links/lipperrdino/g6clt6d5
).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How to train and respoduce the results of the best model saved at [google drive link](https://drive.google.com/file/d/1bduk4t_2lpBduSBpkycQydCaINyp4wFU/view?usp=sharing):

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To reproduce the best model, just specify the path to the training dataset with wavs with **datasets.train.audio_dir=**. 
Dataset config **train**, which is used by default, needs two more paths to data: to validation dataset with txt queries and to test dataset with wavs to resynthesize. For training this is not nessesary, so to just train, specify **datasets=train_only**.

Directories to wavs/txt should contain data of the following format:

For wavs (LJSpeechDataset):
```bash
NameOfTheDirectory
└── wavs
    ├── UtteranceID1.wav
    ├── UtteranceID2.wav
    .
    .
    .
    └── UtteranceIDn.wav
```

For text queries (CustomDirDataset):
```bash
NameOfTheDirectory
└── transcriptions
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    .
    .
    .
    └── UtteranceIDn.txt

    # OR

    metadata.csv # to save the parsed text queries from this file you can provide a path for it with save_path=<your path>
```

**So, to train with default configs, run the following command:**
```bash
python3 train.py datasets.train.audio_dir=<your path> datasets.val.transcription_dir=<your path> datasets.test.audio_dir=<your path>
```

## How to run inference (synthesize audio)
Script *synthesize.py* generates wavs from wavs or from text queries and saves the predictions to the directory, specified by **inferencer.save_path=** (default is "generated").

To synthesize *audio from text* use **datasets=custom_text2mel**, to resynthesize *audio from audio* use **datasets=inference_wav2wav**. Don't forget to specify your paths (here use datasets.test. + audio_dir/transcriptions_dir).

To run inference on a pretrained model, specify path to the file with it's weights: **inferencer.from_pretrained=**.

To run inference on the best model, don't do anything, the script will automatically download the weights from gdrive in the directory "saved/model_best.pth" (or provide your path, if it doesn't exist, the best model will be loaded there, if it does, the program will assume that you provided your own weights).

So, to run inference on the best model, you need to chose the dataset config and provide the path to your data:

```bash
python3 synthesize.py datasets=custom_text2mel  datasets.test.transcriptions_dir=<your path>
```
or 
```bash
python3 synthesize.py datasets=inference_wav2wav  datasets.test.audio_dir=<your path>
```

**To synthesize your single text query, provide it as in the following command:**

```bash
python3 synthesize.py datasets=custom_text2mel  inferencer.query="<your query>"
```

## How to download best model weights 

Run this command with the path where to download the weights:

```bash
python3 download_best_model.py --path <your path>
```


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
