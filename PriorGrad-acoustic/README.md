# PriorGrad-acoustic

## Quick Start and Examples

1. Navigate to PriorGrad-acoustic root, install dependencies, and initialize submodule ([HiFi-GAN](https://github.com/jik876/hifi-gan) vocoder)
   ```bash
   # the codebase has been tested on Python 3.8 with PyTorch 1.8.2 LTS and 1.10.2 conda binaries
   pip install -r requirements.txt
   ```

Note: We release the pre-built LJSpeech binary dataset that can skip the preprocessing (step 2, 3 and 4). Refer to the [Pretrained Weights](https://huggingface.co/neuralspeech/priorgrad/blob/main/priorgrad_am.zip) section below.

2. Prepare the dataset (LJSpeech)
   ```bash
   mkdir -p data/raw/
   cd data/raw/
   wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
   tar -xvf LJSpeech-1.1.tar.bz2
   cd ../../
   python datasets/tts/lj/prepare.py
   ```
3. Forced alignment for duration predictor training
   ```bash
   # The following commands are tested on Ubuntu 18.04 LTS.
   sudo apt install libopenblas-dev libatlas3-base
   # Download MFA from https://montreal-forced-aligner.readthedocs.io/en/stable/aligning.html
   wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
   # Unzip to montreal-forced-aligner
   tar -zxvf montreal-forced-aligner_linux.tar.gz
   # See https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/149 regarding this fix
   cd montreal-forced-aligner/lib/thirdparty/bin && rm libopenblas.so.0 && ln -s ../../libopenblasp-r0-8dca6697.3.0.dev.so libopenblas.so.0
   cd ../../../../
   # Run MFA
   ./montreal-forced-aligner/bin/mfa_train_and_align \
   data/raw/LJSpeech-1.1/mfa_input \
   data/raw/LJSpeech-1.1/dict_mfa.txt \
   data/raw/LJSpeech-1.1/mfa_outputs \
   -t ./montreal-forced-aligner/tmp \
   -j 24
   ```

4. Build binary data and extract mean & variance for PriorGrad-acoustic. The mel-spectrogram is compatible with open-source [HiFi-GAN](https://github.com/jik876/hifi-gan)

   ```bash
   PYTHONPATH=. python datasets/tts/lj/gen_fs2_p.py \
   --config configs/tts/lj/priorgrad.yaml \
   --exp_name priorgrad
   ```

5. Train PriorGrad-acoustic
   ```bash
   # the following command trains PriorGrad-acoustic with default parameters defined in configs/tts/lj/priorgrad.yaml
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/priorgrad.py \
   --config configs/tts/lj/priorgrad.yaml \
   --exp_name priorgrad \
   --reset
   ```
   
   ### Optional feature: Monotonic alignment search (MAS) support
   Instead of MFA, PriorGrad also supports Monotonic Alignment Search (MAS) used in [Glow-TTS](https://github.com/jaywalnut310/glow-tts/) for duration predictor training.
      ```bash
      # install monotonic_align for MAS training
      cd monotonic_align && python setup.py build_ext --inplace && cd ..
      # The following command trains a variant of PriorGrad which uses MAS for training the duration predictor.
      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/priorgrad.py \
      --config configs/tts/lj/priorgrad.yaml \
      --hparams dur=mas \
      --exp_name priorgrad_mas \
      --reset
      ```

6. Download pre-trained HiFi-GAN vocoder
    ```
    mkdir hifigan_pretrained
    ```
    download `generator_v1`, `config.json` from [Google Drive](https://drive.google.com/drive/folders/1XtZ_AaYIsnx1zh_HxhrG5SZ6MUJV59gm) to `hifigan_pretrained/`

   
7. Inference (fast mode with T=12)
   ```bash
   # the following command performs test set inference along with a grid search of the reverse noise schedule. 
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/priorgrad.py \
   --config configs/tts/lj/priorgrad.yaml \
   --exp_name priorgrad \
   --reset \
   --infer \
   --fast --fast_iter 12
   ```
   
   When `--infer --fast`, the model applies grid search of beta schedules with the specified number of `--fast_iter` steps for the given model checkpoint.
   
   2, 6, and 12 `--fast_iter` are officially supported. If the value higher than 12 is provided, the model uses a linear beta schedule. Note that the linear schedule is expected to perform worse.
   
   `--infer` without `--fast` performs slow sampling with the same `T` as the forward diffusion used in training.

## Text-to-speech with User-given Text

`tasks/priorgrad_inference.py` provides the text-to-speech inference of PriorGrad-acoustic with user-given text file defined in `--inference_text`. Refer to `inference_text.txt` for example.
   ```bash
   # the following command performs text-to-speech inference from inference_text.txt
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/priorgrad_inference.py \
   --config configs/tts/lj/priorgrad.yaml \
   --exp_name priorgrad \
   --reset \
   --inference_text inference_text.txt \
   --fast --fast_iter 12
   ```

Samples are saved to folders with `inference_(fast_iter)_(train_step)`  created at `--exp_name`.

When using `--fast`, the grid-searched reverse noise schedule file is required. Refer to the inference section (step 7) of the examples above.  
