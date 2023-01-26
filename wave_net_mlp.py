import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
d_model = 384
d_hidden = 2*d_model
n_layer = 1
dropout = 0.2
write_to_file = False
# ------------

@dataclass
class Config:
    pass

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # 256
    y = torch.stack([data[i+block_size] for i in ix]) # 1
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# RNN
class WaveNetMLPLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.ln_f = nn.LayerNorm(vocab_size) # final layer norm
        self.ln_1 = nn.LayerNorm(d_model // 2)
        self.ln_2 = nn.LayerNorm(d_model // 4)
        self.ln_3 = nn.LayerNorm(d_model // 8)

        self.linear_1 = nn.Linear(2 * d_model, d_model) # (2 * 384, 384)
        self.linear_2 = nn.Linear(2 * d_model, d_model) # (2 * 384, 384)
        self.linear_3 = nn.Linear(2 * d_model, d_model) # (2 * 384, 384)
        self.linear_4 = nn.Linear(2 * d_model, d_model) # (2 * 384, 384)
        self.linear_5 = nn.Linear(2 * d_model, d_model) # (2 * 384, 384)
        self.linear_6 = nn.Linear(2 * d_model, d_model) # (2 * 384, 384)

        self.ff = nn.Linear(4 * d_model, vocab_size)
        self.gelu = nn.GELU()

        self.dp_1 = nn.Dropout(dropout)
        self.dp_2 = nn.Dropout(dropout)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,d_model)

        # Do this here!!!

        x = tok_emb.view(B, T // 2, -1) 

        x = self.linear_1(x) # (B, T // 2, d_model), T  = 128

        x = x.view(B, T // 4, -1)

        x = self.linear_2(x) # (B, T // 4, d_model), T = 64

        x = x.view(B, T // 8, -1)

        x = self.linear_3(x) # (B, T // 8, d_model), T = 32

        x = x.view(B, T // 16, -1)

        x = self.linear_4(x) # (B, T // 16, d_model), T = 16

        x = x.view(B, T // 32, -1)

        x = self.linear_5(x) # (B, T // 32, d_model), T = 8

        x = x.view(B, T // 64, -1)

        x = self.linear_6(x) # (B, T // 64, d_model), T // 64 = 4

        x = x.view(B, -1) # (B, 4 * d_model)

        logits = self.ff(x) # (B, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, C = logits.shape

            assert(C == vocab_size)

            #print(logits.shape, targets.shape)
            
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step and use it to generate
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = WaveNetMLPLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for it in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
if write_to_file:
    open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
