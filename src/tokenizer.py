import torch
import numpy as np

class FloatTokenizer:
    def __init__(self):
        # Create vocabulary with numbers 0-9, decimal point, and separator
        self.vocab = [str(i) for i in range(10)] + ['.', '|']  # Added separator token '|'
        self.stoi = {ch:i for i,ch in enumerate(self.vocab)}
        self.itos = {i:ch for i,ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.separator_token = self.stoi['|']
    
    def encode_float(self, num):
        """Convert a single float to a list of token indices"""
        # Convert float to string with 2 decimal places
        num_str = f"{num:.2f}"
        tokens = [self.stoi[c] for c in num_str]
        tokens.append(self.separator_token)  # Add separator after each number
        return tokens
    
    def encode_sequence(self, float_list):
        """Convert a list of floats to a sequence of token indices"""
        all_tokens = []
        for num in float_list:
            tokens = self.encode_float(num)
            all_tokens.extend(tokens)
        return torch.tensor(all_tokens, dtype=torch.long)
    
    def decode_sequence(self, tokens):
        """Convert a sequence of token indices back to a list of floats"""
        chars = [self.itos[int(t)] for t in tokens]
        text = ''.join(chars)
        number_strings = text.split('|')
        return [float(num_str) for num_str in number_strings if num_str.strip()]

    def visualize_training_chunk(self, chunk):
        """Visualize a single chunk of training data"""
        tokens = [self.itos[t.item() if torch.is_tensor(t) else t] for t in chunk]
        return ''.join(tokens)

    def prepare_training_data(self, float_sequences, block_size):
        """
        Prepare training data from sequences of floats.
        Each float_sequence is a list of floats.
        Returns: tensor of shape (num_sequences, block_size)
        """
        all_data = []
        for sequence in float_sequences:
            tokens = self.encode_sequence(sequence)
            for i in range(0, len(tokens) - block_size + 1):
                chunk = tokens[i:i + block_size]
                all_data.append(chunk)
        
        if not all_data:
            raise ValueError("No valid sequences found after tokenization")
        
        # Print example chunks before stacking
        print("Example training chunks:")
        for i in range(min(3, len(all_data))):
            print(f"Chunk {i}: {self.visualize_training_chunk(all_data[i])}")
        
        # Stack all chunks into a single tensor
        return torch.stack(all_data) 