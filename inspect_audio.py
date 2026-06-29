import torch
import torchaudio
import numpy as np

def inspect_wav(filename):
    try:
        waveform, sample_rate = torchaudio.load(filename)
        print(f"File: {filename}")
        print(f"Shape: {waveform.shape}")
        print(f"Sample Rate: {sample_rate}")
        print(f"Max value: {waveform.max().item()}")
        print(f"Min value: {waveform.min().item()}")
        print(f"Mean value: {waveform.mean().item()}")
        print(f"Std dev: {waveform.std().item()}")
        
        # Check for NaNs or Infinite values
        if torch.isnan(waveform).any():
            print("Contains NaNs!")
        if torch.isinf(waveform).any():
            print("Contains Infs!")
            
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    inspect_wav("demo_expressive_1.wav")
