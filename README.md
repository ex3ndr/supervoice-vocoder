# ✨ SuperVoice Vocoder
Easy to use SOTA vocoder for speech synthesis from Mel Spectograms.

## Description
This repository contains easy to use SOTA vocoder that is uses BigVSAN original pre-trained weights with removed weight normalization and most of the code, but built to be used in a plug and play fashion. This model is tailored to a specific Mel-Spectogram parameters which is most common for voice synthesis tasks.

* Model: [BigVSAN](https://arxiv.org/abs/2309.02836)
* Sample rate: 24000 Hz
* Mel Spectogram:
  * Mel Number: `100`
  * Number of FFT: `1024`
  * Hop Length: `256`
  * Window Length: `1024`
  * Norm: `slaney`
  * Scale: `slaney`
  * Power: `1.0` aka amplitude spectogram
  * Center: `true`
  * Padding: `reflect`

## Evaluation

To evaluate model you can use [evaluation notebook](/eval.ipynb) which can run anywhere where `torch` and `torchaudio` are installed.
 
## How to use

This model is available using Torch Hub:

```python

# Load model
bigvsan = torch.hub.load(repo_or_dir='ex3ndr/supervoice-vocoder', model='bigvsan')

# Source mel spectogram
spec = torch.randn(100, 1234)

# Synthesized audio
audio = model.generate(spec)

```

## License

MIT
