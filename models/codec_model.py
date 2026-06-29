import torch
import torch.nn as nn
from encodec import EncodecModel
from encodec.utils import convert_audio

class CodecModel(nn.Module):
    """
    Wrapper for EnCodec to handle audio tokenization and reconstruction.
    Bark typically uses 24kHz EnCodec.
    """
    def __init__(self, device="cpu"):
        super().__init__()
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0)  # Default bandwidth for 8 codebooks
        self.model.to(device)
        self.sample_rate = 24000
        self.num_codebooks = 8

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, device=None):
        """
        waveform: [B, 1, T]
        returns: (tokens, scale) where tokens is [B, K, T']
        """
        if device:
            waveform = waveform.to(device)
            self.model.to(device)
            
        # Ensure sample rate matches
        # Note: In a real training pipeline, we'd use convert_audio
        encoded_frames = self.model.encode(waveform)
        # Extract codes: [B, K, T']
        codes = torch.cat([x[0] for x in encoded_frames], dim=-1)
        return codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, device=None):
        """
        codes: [B, K, T']
        returns: reconstructed waveform [B, 1, T]
        """
        if device:
            codes = codes.to(device)
            self.model.to(device)
            
        # Encodec expects list of (codes, scale)
        # For simple inference, scale is None
        encoded_frames = [(codes, None)]
        out = self.model.decode(encoded_frames)
        return out

if __name__ == "__main__":
    # Test with dummy audio
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = CodecModel(device=device)
    dummy_wav = torch.randn(1, 1, 24000).to(device)
    tokens = codec.encode(dummy_wav)
    print(f"Tokens shape: {tokens.shape}")
    reconstructed = codec.decode(tokens)
    print(f"Reconstructed shape: {reconstructed.shape}")
