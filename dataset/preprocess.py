import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from models.codec_model import CodecModel
from dataset.text_processing import TextProcessor

class BarkDataset(Dataset):
    """
    Dataset for training Bark-style models.
    Expects metadata.csv with: audio_path, transcript, emotion, speaker
    """
    def __init__(self, metadata_path, codec: CodecModel, text_processor: TextProcessor, max_len=1024):
        self.df = pd.read_csv(metadata_path)
        self.codec = codec
        self.text_processor = text_processor
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['audio_path']
        transcript = row['transcript']
        emotion = row.get('emotion', 0)
        speaker = row.get('speaker', 0)
        
        # Load and normalize audio
        wav, sr = torchaudio.load(audio_path)
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            wav = resampler(wav)
        
        # Ensure mono
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        # 1. Get Acoustic Tokens (Target for Acoustic Models)
        with torch.no_grad():
            acoustic_tokens = self.codec.encode(wav.unsqueeze(0)) # [1, K, T]
            acoustic_tokens = acoustic_tokens.squeeze(0) # [K, T]
            
        # 2. Get Text Tokens (Input for Semantic Model)
        text_tokens = torch.tensor(self.text_processor.tokenize(transcript))
        
        # 3. Get Semantic Tokens (Target for Semantic Model / Input for Coarse)
        # In a real pipeline, we'd use a HuBERT model to extract semantic tokens.
        # For this implementation, we can use a simplified version or extract 
        # from the first codebook of EnCodec if HuBERT isn't available.
        # Bark uses HuBERT semantic tokens.
        semantic_tokens = acoustic_tokens[0] # Simplification for demo
        
        return {
            "text_tokens": text_tokens,
            "semantic_tokens": semantic_tokens,
            "acoustic_tokens": acoustic_tokens,
            "emotion": torch.tensor(emotion),
            "speaker": torch.tensor(speaker)
        }

def preprocess_audio(audio_path, codec):
    """
    Helper for inference or one-off preprocessing.
    """
    wav, sr = torchaudio.load(audio_path)
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000)
        wav = resampler(wav)
    # Get tokens
    tokens = codec.encode(wav.unsqueeze(0))
    return tokens

if __name__ == "__main__":
    # Example usage:
    # codec = CodecModel()
    # processor = TextProcessor()
    # ds = BarkDataset("dataset/metadata.csv", codec, processor)
    print("Dataset class implemented.")
