import torch
from torch import nn


class SelfAttentionHead(nn.Module):
  def __init__(self, head_size, embedding_size, block_size, dropout):
    super().__init__()
    self.key = nn.Linear(embedding_size, head_size, bias = False)
    self.query = nn.Linear(embedding_size, head_size, bias = False)
    self.value = nn.Linear(embedding_size, head_size, bias = False)
    self.droupout = nn.Dropout(dropout)
    self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    Batch, Block, Size = x.shape
    
    keys = self.key(x)
    queries = self.query(x)

    weights = queries @ keys.transpose(-2, -1) * Size ** -0.5
    weights = weights.masked_fill(self.mask[:Block, :Block] == 0, float('-inf'))
    weights = nn.functional.softmax(weights, dim=-1)
    weights = self.droupout(weights)

    values = self.value(x)

    return weights @ values


class MultiHeadAttention(nn.Module):
  def __init__(self, head_size, num_heads, embedding_size, block_size, dropout):
    super().__init__()
    self.heads = nn.ModuleList(
      [SelfAttentionHead(head_size, embedding_size, block_size, dropout) for _ in range(num_heads)]
    )
    self.projection = nn.Linear(embedding_size, embedding_size)
    self.droupout = nn.Dropout(dropout)

  def forward(self, x):
    y = torch.cat([h(x) for h in self.heads], dim=-1)
    y = self.projection(y)
    y = self.droupout(y)
    return y


class FeedForwardLayer(nn.Module):
  def __init__(self, embedding_size, dropout):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(embedding_size, 4 * embedding_size),
      nn.ReLU(),
      nn.Linear(4 * embedding_size, embedding_size),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.network(x)


class DecoderBlock(nn.Module):
  def __init__(self, embedding_size, block_size, num_heads, dropout = 0.2):
    super().__init__()
    head_size = embedding_size // num_heads
    self.attention_heads = MultiHeadAttention(head_size, num_heads, embedding_size, block_size, dropout)
    self.feedforward_layer = FeedForwardLayer(embedding_size, droupout)
    self.layer_norm1 = nn.LayerNorm(embedding_size)
    self.layer_norm2 = nn.LayerNorm(embedding_size)

  def forward(self, x):
    x = x + self.attention_heads(x)
    x = x + self.feedforward_layer(x)
    return x
    


class SimpleModel(nn.Module):
  def __init__(self, vocab_size, embedding_size, block_size, layer_num=6, attention_heads=8, dropout=0.2):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
    self.position_embedding_table = nn.Embedding(block_size, embedding_size)
    self.decoder_blocks = nn.Sequential(
      *[DecoderBlock(embedding_size, block_size, num_heads=attention_heads, dropout=dropout) for _ in range(layer_num)],
      nn.LayerNorm(embedding_size)
    )
    self.linear_layer = nn.Linear(embedding_size, vocab_size)

  def forward(self, idx, targets=None):
    Batch, Block = idx.shape

    token_embeddings = self.token_embedding_table(idx)
    positional_embeddings = self.position_embedding_table(torch.arange(Block))
    
    embeddings = token_embeddings + positional_embeddings
    embeddings = self.decoder_blocks(embeddings)

    logits = self.linear_layer(embeddings)

    if targets is None:
      loss = None
    else:
      Batch, Block, Vocab = logits.shape
      logits = logits.view(Batch * Block, Vocab)
      targets = targets.view(Batch * Block)
      loss = nn.functional.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max):
    Batch, Block = idx.shape

    for _ in range(max):
      cropped_idx = idx[:, -Block:]
      logits, loss = self(cropped_idx)
      logits = logits [:,-1,:]
      probs = nn.functional.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx