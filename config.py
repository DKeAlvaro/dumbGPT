import torch
BLOCK_SIZE = 32
BATCH_SIZE = 16
MAX_ITERS = 100
EVAL_INTERVAL = 10
LEARNING_RATE = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0

