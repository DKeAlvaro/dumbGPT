from src.helpers import save_to_lists, divide_lists
from src.dumbGPT import dumbGPT, estimate_loss, get_batch
from src.tokenizer import FloatTokenizer
import torch
from config import BLOCK_SIZE, DEVICE, MAX_ITERS, EVAL_INTERVAL, LEARNING_RATE, N_EMBD
import os


def from_train_to_inference():

    print("\033[DATA PROCESSING:\033[0m")
    print("We first save the .txt observations to a list of lists")
    full_obs = save_to_lists()
    print(f"After saving them, we have a total of {len(full_obs)} lists of size {len(full_obs[0])} each")
    print(f"One list might look like this: {full_obs[0][0:10]} (first 10 elements) ")
    train_obs, test_obs = divide_lists(full_obs, 20)
    print(f"We are going to divide these {len(full_obs)} lists into train and test")
    print(f"We will train our model with {len(train_obs)} lists and test it with the rest ({len(test_obs)})")
    
    print()
    print("\033[1mTOKENIZATION:\033[0m")
    print("We first create a Tokenizer instance")
    print("We will use our custom Float Tokenizer")
    tokenizer = FloatTokenizer()

    print("Tokenization Example:")
    sample_sequence = train_obs[0][:5]
    print(f"Original sequence: {sample_sequence}")
    tokens = tokenizer.encode_sequence(sample_sequence)
    print(f"Token indices: {tokens}")
    print(f"Token characters: {[tokenizer.itos[t.item()] for t in tokens]}")
    decoded = tokenizer.decode_sequence(tokens)
    print(f"Decoded numbers: {decoded}")
    print(f"We now do a similar thing with all the train observations dividing sequences with size BLOCK_SIZE={BLOCK_SIZE}")
    data = tokenizer.prepare_training_data(train_obs, BLOCK_SIZE)
    sequence_tokens = tokenizer.encode_sequence(train_obs[0])
    print(f"'data' is roughly a tensor {len(train_obs)*(len(sequence_tokens)-BLOCK_SIZE)} times {BLOCK_SIZE}")
    print(data.shape)
    print(f"Number of differnt sequences across all training data, each with size {BLOCK_SIZE} (Thats why its 2D)")

    print()
    print("\033[1mMODEL CREATION: \033[0m")
    model = dumbGPT(vocab_size=tokenizer.vocab_size)
    model = model.to(DEVICE)
    model.print_model_size()
    vocab_size = tokenizer.vocab_size
    print(f"token_embedding_table is a matrix size (vocab_size, N_EMBD) = {vocab_size * N_EMBD}")   
    print(f"The vocab is the set of different tokens in the trainset")
    print(f"In our case it looks like this: {tokenizer.vocab}")
    print(f"There are vocab_size ({vocab_size}) entries represented with N_EMBD ({N_EMBD}) features.")
    print(f"The fact that token_embedding_table is an instance from nn.module implies that its values will be updated when training")
    print(f"Same goes for the rest of the components")

    print("\033[1mMODEL TRAINING: \033[0m") 
    print("We create our Training instance and we assign it our model params")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"Within our training set, we will still divide it 90% - 10% to visualize the loss")
    train_idx = int(0.9 * len(data))

    print(f"Every EVAL_INTERVAL({EVAL_INTERVAL}) iterations we will print the training and validation losses")
    best_val_loss = float('inf')
    for iter in range(MAX_ITERS):
        # Sample a batch of data
        xb, yb = get_batch(data, train_idx)
        
        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Logging
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, data, train_idx)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    print("\033[1mUSING THE MODEL \033[0m") 
    print("Setting the model to eval mode")
    model.eval()  
    context = tokenizer.encode_sequence(test_obs[0][0:10]) 
    print(f"Context (Tensor of token IDs): {context}")
    print(f"Context has shape: {context.shape}")
    print(f"However, we need the context to have shape (B, T) so we can feed it to our model")
    print(f"Where B is batch size and T is sequence length")
    print(f"(We have 1 sequence of size {context.size(0)})")
    print(f"We add the 'Batch dimension with' .unsqueeze(0)")
    print(f"This is like saying:")
    print("'Hey, I only have one sequence (one batch of data), so please treat it as a batch with size 1'")
    context = context.unsqueeze(0)
    print(f"Now, context has shape: {context.shape}")   
    generated = model.generate(context, max_new_tokens=10)[0]
    print(f"The model ultimately generates token IDs we need to decode: {generated[:10]}")
    print("Original sequence:")
    print(test_obs[0][0:10])
    print(f"Generated sequence:")
    print(tokenizer.decode_sequence(generated))

def main():

    from_train_to_inference()
if __name__ == "__main__":
    main()