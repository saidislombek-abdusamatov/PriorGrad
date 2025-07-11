# PriorGrad

PriorGrad is an implementation of a diffusion-based acoustic model for neural speech synthesis, paired with a HiFi-GAN vocoder for high-fidelity waveform generation.

## Repository Structure

* `PriorGrad-acoustic`: Training and inference scripts for the PriorGrad acoustic model.
* `HiFi-GAN-vocoder`: HiFi-GAN implementation for waveform synthesis from mel-spectrograms.

## Requirements

* Python
* PyTorch
* torchaudio
* Additional dependencies: see each subdirectory's `requirements.txt`.

## References

* Lee et al., PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior, ICLR 2022.
* Kong et al., HiFi-GAN: Generative Adversarial Networks for Efficient and High-Fidelity Speech Synthesis, NeurIPS 2020.
