import torch
import os
from src.dumbGPT import nanoGPT, estimate_loss, get_batch
from config import *

class Trainer:
    def __init__(self, model_dir='checkpoints'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Data preparation
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Character encoding setup
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for i,ch in enumerate(self.chars)}
        
        # Prepare data splits
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9*len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        # Initialize model and optimizer
        self.model = self.load_or_create_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        
    def encode(self, s):
        return [self.stoi[c] for c in s]
        
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
    
    def load_or_create_model(self):
        checkpoint_path = os.path.join(self.model_dir, 'model_checkpoint.pt')
        
        if os.path.exists(checkpoint_path):
            print("Loading existing model checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            model = nanoGPT()
            model.load_state_dict(checkpoint['model_state'])
            model = model.to(DEVICE)
            return model
        else:
            print("Creating new model...")
            model = nanoGPT()
            model = model.to(DEVICE)
            return model
    
    def save_checkpoint(self, iter, loss):
        checkpoint = {
            'iter': iter,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': loss
        }
        checkpoint_path = os.path.join(self.model_dir, 'model_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
    def train(self):
        print(f"Training model with {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")
        
        for iter in range(MAX_ITERS):
            # Evaluation
            if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                self.save_checkpoint(iter, losses['train'])
            
            # Training step
            xb, yb = get_batch('train')
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

if __name__ == '__main__':
    torch.manual_seed(1337)
    trainer = Trainer()
    trainer.train()

