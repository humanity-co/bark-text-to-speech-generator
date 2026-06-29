import torch
from models.fine_model import FineModel
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

def train_fine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FineModel(n_layer=6, n_head=8, n_embd=512).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    print("Starting Fine Model Training Loop...")
    
    for epoch in range(5):
        # In each step, pick a random codebook index to predict (2 to 7)
        # target_cb = torch.randint(2, 8, (1,)).item()
        # with autocast():
        #     logits = model(codes, target_codebook_idx=target_cb)
        #     loss = cross_entropy(logits, targets)
        print(f"Epoch {epoch} complete (Dummy)")
        break

if __name__ == "__main__":
    train_fine()
