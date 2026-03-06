"""
"""
import torch
import torch.nn as nn
import torchaudio

class MFB(nn.Module):
    """Compute 80D log Mel- filter bank features."""
    def __init__(self, sample_rate=16000, window_size=0.04, hop_size=0.02, n_mels=80):
        super().__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(sample_rate * window_size),
            win_length=int(sample_rate * window_size),
            hop_length=int(sample_rate * hop_size),
            n_mels=n_mels,
            window_fn=torch.hann_window,
            power=2.0
        )
    
    def forward(self, x, mask=None):
        # x: (B, T_samples), waveform at 16 kHz
        # mask: ignored for now (None)
        x = self.transform(x)  # (B, n_mels, T_frames)
        x = torch.log(x + 1e-6)  # Log-scale
        return x
