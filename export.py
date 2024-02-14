import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from supervoice_vocoder.bigvsan import BigVSAN

# Load source model
checkpoint = torch.load('bigvsan.pt', map_location='cpu')
state_dict = checkpoint['generator']

# Remove weight normalization
for key in list(state_dict.keys()):  # list(...) to make a copy of keys
    if 'weight_v' in key:
        layer_base = key.rsplit('.', 1)[0]  # Get the base name of the layer
        g_key = f'{layer_base}.weight_g'
        if g_key in state_dict:
            v = state_dict[key]
            g = state_dict[g_key]
            # Compute the recombined weight with proper norm calculation
            norm = torch.norm(v, p=2, dim=list(range(1, v.dim())), keepdim=True)
            recombined_weight = g * (v / norm)
            # Update the original weight and remove g and v
            state_dict[layer_base + '.weight'] = recombined_weight
            del state_dict[key]  # Remove weight_v
            del state_dict[g_key]  # Remove weight_g

# Try to load model
model = BigVSAN()
model.load_state_dict(state_dict)

# Synthesize sample
source, _ = torchaudio.load('sample.wav')
def spectogram(src):
    # Hann Window
    window = torch.hann_window(1024)

    # STFT
    stft = torch.stft(src, 
        n_fft = 1024, 
        hop_length = 256, 
        win_length = 1024,
        window = window, 
        center = True,
        return_complex = True
    )

    # magnitudes = stft[..., :-1].abs() ** 2 # Power
    magnitudes = stft[..., :-1].abs() # Amplitude (used by BigVSAN)

    # Mel Bank
    mel_filters = F.melscale_fbanks(
            n_freqs=int(1024 // 2 + 1),
            sample_rate=24000,
            f_min=0,
            f_max=12000,
            n_mels=100,
            norm="slaney",
            mel_scale="slaney"
    ).transpose(-1, -2)
    mel_spec = (mel_filters @ magnitudes)

    # Log
    log_spec = torch.clamp(mel_spec, min=1e-10).log()

    return log_spec
spec = spectogram(source)
resynth = model(spec).detach().squeeze(0)
model.eval()
torchaudio.save('resynth.wav', resynth, 24000)

# Export the model
torch.save(model.state_dict(), 'bigvsan_no_norm.pt')