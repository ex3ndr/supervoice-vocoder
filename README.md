# ✨ SuperVoice Vocoder
SOTA vocoder for speech synthesis.

## Description
This repository contains easy to use SOTA vocoder that is uses BigVSAN official pre-trained weights. This model is tailored to a specific Mel-Spectogram parameters which is most common for voice synthesis tasks.

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
 
## Install

TODO

## References

## License

MIT
