import torch
import torch.nn as nn
from typing import Optional

class CoarseModel(nn.Module):
    """
    Autoregressive Transformer that maps semantic tokens to the first 2 codebooks of acoustic tokens.
    """
    def __init__(self, 
                 num_semantic_tokens: int = 10000, 
                 num_acoustic_tokens: int = 1024, # EnCodec codebook size
                 n_layer: int = 12, 
                 n_head: int = 12, 
                 n_embd: int = 768,
                 block_size: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.semantic_emb = nn.Embedding(num_semantic_tokens, n_embd)
        # We predict 2 codebooks simultaneously or flattened. 
        # Bark typically flattens them or uses a specific pattern.
        self.acoustic_emb = nn.Embedding(num_acoustic_tokens, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, 
            nhead=n_head, 
            dim_feedforward=4 * n_embd, 
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.ln_f = nn.LayerNorm(n_embd)
        # Head predicts acoustic tokens
        self.head = nn.Linear(n_embd, num_acoustic_tokens, bias=False)
        
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
                semantic_idx: torch.Tensor, 
                acoustic_idx: torch.Tensor, # [B, T_a] where T_a is acoustic sequence length
                targets: Optional[torch.Tensor] = None):
        
        device = semantic_idx.device
        b, t_s = semantic_idx.size()
        b, t_a = acoustic_idx.size()
        
        # Combine semantic and acoustic embeddings
        x_s = self.semantic_emb(semantic_idx)
        x_a = self.acoustic_emb(acoustic_idx)
        
        # Concat semantic then acoustic
        x = torch.cat([x_s, x_a], dim=1)
        t = x.size(1)
        
        assert t <= self.block_size, f"Sequence length {t} exceeds block size {self.block_size}"
        
        # Add positioning
        x = x + self.pos_emb[:, :t, :]
        x = self.drop(x)
        
        # Causal mask
        mask = torch.triu(torch.ones(t, t, device=device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            # Shift targets to align with logits for AR training
            # Usually targets are only for the acoustic part
            loss = nn.functional.cross_entropy(logits[:, t_s-1:-1, :].reshape(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

if __name__ == "__main__":
    model = CoarseModel(n_layer=4, n_head=4, n_embd=256)
    s_idx = torch.randint(0, 1000, (1, 50))
    a_idx = torch.randint(0, 1024, (1, 100))
    logits, _ = model(s_idx, a_idx)
    print(f"Logits shape: {logits.shape}")
