import torch
import torchaudio
import numpy as np

# Patch torch.load globally before any other imports
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

try:
    import torch.serialization
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np._core.multiarray.scalar])
except:
    pass
from bark import SAMPLE_RATE, generate_audio, preload_models

def run_expressive_demo():
    print("Loading pre-trained Bark weights (this may take a few minutes)...")
    preload_models()

    inputs = [
        ("Hindi: नमस्ते, आप कैसे हैं? [laugh]", "v2/hi_speaker_0"),
        ("Marathi: नमस्कार, तुमचं नाव काय आहे? [sigh]", "v2/hi_speaker_1"), # Bark uses hi_speaker for Marathi often
        ("English: Hello! This is high-quality expressive speech with a breath. [breath]", "v2/en_speaker_6")
    ]

    for i, (text, speaker) in enumerate(inputs):
        print(f"Generating: {text}")
        audio_array = generate_audio(text, history_prompt=speaker)
        
        # Save output
        filename = f"demo_expressive_{i}.wav"
        torchaudio.save(filename, torch.from_numpy(audio_array).unsqueeze(0), SAMPLE_RATE)
        print(f"Saved to {filename}")

if __name__ == "__main__":
    try:
        import bark
    except ImportError:
        print("Installing official Bark for demo...")
        import os
        os.system("pip install git+https://github.com/suno-ai/bark.git")
    
    run_expressive_demo()
