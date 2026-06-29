import torch
import torch.nn as nn
from typing import Optional

class SemanticModel(nn.Module):
    """
    GPT-style Transformer that takes text tokens and predicts semantic tokens.
    Semantic tokens bridge the gap between text and acoustic details.
    """
    def __init__(self, 
                 num_text_tokens: int = 10000, 
                 num_semantic_tokens: int = 10000,
                 n_layer: int = 12, 
                 n_head: int = 12, 
                 n_embd: int = 768,
                 block_size: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        # Token & Positional Embeddings
        self.text_emb = nn.Embedding(num_text_tokens, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(dropout)
        
        # Style & Emotion conditioning
        self.style_emb = nn.Embedding(20, n_embd)  # Example: 20 styles
        self.emotion_emb = nn.Embedding(10, n_embd) # Example: 10 emotions
        
        # Transformer Blocks
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
        
        # Prediction head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, num_semantic_tokens, bias=False)
        
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
                idx: torch.Tensor, 
                style: Optional[torch.Tensor] = None, 
                emotion: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is {self.block_size}"
        
        # Text embedding
        x = self.text_emb(idx)
        
        # Add style and emotion if provided (broadcasting across sequence)
        if style is not None:
            x = x + self.style_emb(style).unsqueeze(1)
        if emotion is not None:
            x = x + self.emotion_emb(emotion).unsqueeze(1)
            
        # Add positioning
        x = x + self.pos_emb[:, :t, :]
        x = self.drop(x)
        
        # Transformer
        # We use a causal mask for autoregressive generation of semantic tokens
        mask = torch.triu(torch.ones(t, t, device=device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

if __name__ == "__main__":
    model = SemanticModel(n_layer=4, n_head=4, n_embd=256)
    idx = torch.randint(0, 1000, (1, 50))
    style = torch.tensor([1])
    emotion = torch.tensor([2])
    logits, _ = model(idx, style=style, emotion=emotion)
    print(f"Logits shape: {logits.shape}")
