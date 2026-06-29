import torch
import torch.nn as nn
from typing import Optional

class FineModel(nn.Module):
    """
    Non-autoregressive Transformer that predicts higher-level codebooks (2-7) 
    given the first two codebooks from the coarse model.
    """
    def __init__(self, 
                 num_acoustic_tokens: int = 1024, 
                 n_layer: int = 6, 
                 n_head: int = 8, 
                 n_embd: int = 512,
                 block_size: int = 1024,
                 num_codebooks: int = 8):
        super().__init__()
        
        self.num_codebooks = num_codebooks
        # Multiple embedding layers for different codebooks
        self.codebook_embs = nn.ModuleList([
            nn.Embedding(num_acoustic_tokens, n_embd) for _ in range(num_codebooks)
        ])
        
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, 
            nhead=n_head, 
            dim_feedforward=4 * n_embd, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # Heads for each remaining codebook (2 to num_codebooks-1)
        self.heads = nn.ModuleList([
            nn.Linear(n_embd, num_acoustic_tokens, bias=False) for _ in range(2, num_codebooks)
        ])
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
                codes: torch.Tensor, # [B, K, T] where K is number of codebooks
                target_codebook_idx: Optional[int] = None):
        """
        In Bark, for the fine model, we often predict codebook K from codebooks 0..K-1.
        codes: [B, K, T]
        target_codebook_idx: which codebook we are predicting (2 to 7)
        """
        b, k, t = codes.size()
        
        # Sum embeddings of input codebooks up to target_codebook_idx
        x = torch.zeros(b, t, self.codebook_embs[0].embedding_dim, device=codes.device)
        for i in range(target_codebook_idx):
            x += self.codebook_embs[i](codes[:, i, :])
            
        x = x + self.pos_emb[:, :t, :]
        
        # Fine model is often non-causal (bidirectional) within the sequence
        x = self.transformer(x)
        
        # Predict the target codebook
        # target_codebook_idx starts at 2, so index into self.heads is target_codebook_idx - 2
        logits = self.heads[target_codebook_idx - 2](x)
        
        return logits

if __name__ == "__main__":
    model = FineModel(n_layer=2, n_head=4, n_embd=256)
    # 8 codebooks, 100 timesteps
    dummy_codes = torch.randint(0, 1024, (1, 8, 100))
    # Predict codebook 2
    logits = model(dummy_codes, target_codebook_idx=2)
    print(f"Logits for CB 2 shape: {logits.shape}")
