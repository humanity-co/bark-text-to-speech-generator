import torch
from models.semantic_model import SemanticModel
from models.coarse_model import CoarseModel
from models.fine_model import FineModel
from models.codec_model import CodecModel
from dataset.text_processing import TextProcessor
import torchaudio

class BarkInference:
    """
    Main API for generating speech from text using the Bark-style pipeline.
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.processor = TextProcessor()
        
        # Load models (In a real scenario, load pre-trained weights)
        self.semantic_model = SemanticModel(n_layer=6, n_head=8, n_embd=512).to(device).eval()
        self.coarse_model = CoarseModel(n_layer=6, n_head=8, n_embd=512).to(device).eval()
        self.fine_model = FineModel(n_layer=6, n_head=8, n_embd=512).to(device).eval()
        self.codec = CodecModel(device=device).eval()

    @torch.no_grad()
    def generate_speech(self, text, speaker=0, emotion=0, style=0, temperature=0.7):
        """
        Full pipeline: text -> semantic -> coarse -> fine -> waveform
        """
        # 1. Text to Semantic
        tokens = torch.tensor([self.processor.tokenize(text)]).to(self.device)
        style_tensor = torch.tensor([style]).to(self.device)
        emotion_tensor = torch.tensor([emotion]).to(self.device)
        
        # Autoregressive generation of semantic tokens (simplified loop)
        semantic_out = []
        curr_tokens = tokens
        for _ in range(100): # Max 100 semantic tokens
            logits, _ = self.semantic_model(curr_tokens, style=style_tensor, emotion=emotion_tensor)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            semantic_out.append(next_token)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
            if next_token.item() == 0: # Stop token
                break
        
        semantic_tokens = torch.cat(semantic_out, dim=1)
        
        # 2. Semantic to Coarse (Acoustic CB 0-1)
        # Simplified: Generate fixed length coarse tokens
        coarse_idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        for _ in range(200): # Max 200 acoustic tokens
            logits, _ = self.coarse_model(semantic_tokens, coarse_idx)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            coarse_idx = torch.cat([coarse_idx, next_token], dim=1)
            
        # Reshape to [1, 2, 100] (2 codebooks)
        coarse_codes = coarse_idx[:, 1:].view(1, 2, -1)
        
        # 3. Coarse to Fine (CB 2-7)
        codes = torch.zeros((1, 8, coarse_codes.shape[-1]), dtype=torch.long, device=self.device)
        codes[:, :2, :] = coarse_codes
        
        for cb_idx in range(2, 8):
            logits = self.fine_model(codes, target_codebook_idx=cb_idx)
            # Find max (non-autoregressive within codebook for simplicity)
            codes[:, cb_idx, :] = torch.argmax(logits, dim=-1)
            
        # 4. Waveform Generation
        waveform = self.codec.decode(codes)
        
        return waveform

if __name__ == "__main__":
    inf = BarkInference()
    wav = inf.generate_speech("Hello, this is a generated voice! [laugh]", emotion=1)
    print(f"Generated waveform shape: {wav.shape}")
    torchaudio.save("output_test.wav", wav.cpu().squeeze(0), 24000)
