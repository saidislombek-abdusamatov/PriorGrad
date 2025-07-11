# HiFi-GAN vocoder

## 1. Installation and Setup for training

Build conda virtual environment
```
conda create --name <env_name> python=3.7
conda activate <env_name>
pip install -r requirements.txt
```
Build Monotonic Alignment Search Code (Cython)

Note : used only for glow-tts
```
bash install.sh
```

## Vocoder Training (hifi-gan)

### 1 Data Preparation

To prepare the data edit the scripts/hifi/prepare_data.sh file and change the following parameters
```
input_wav_path : absolute path to data/resampled_wav_folder_name
gender : female or male voice
```
To run:  
```bash
cd scripts/hifi/
bash prepare_data.sh
```
### 2 Training hifi-gan

To start the spectogram-training edit the scripts/hifi/train_hifi.sh file and change the following parameter:
```
gender : female or male voice
```
Make sure that the gender is same as that of the prepare_data.sh file

To start the training, run:  
```bash
cd scripts/hifi/
bash train_hifi.sh
```
