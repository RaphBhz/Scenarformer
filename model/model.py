import torch
from torch import nn


class SelfAttentionHead(nn.Module):
  def __init__(self, head_size, embedding_size, block_size):
    super().__init__()
    self.key = nn.Linear(embedding_size, head_size, bias = False)
    self.query = nn.Linear(embedding_size, head_size, bias = False)
    self.value = nn.Linear(embedding_size, head_size, bias = False)
    self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    Batch, Block, Size = x.shape
    
    keys = self.key(x)
    queries = self.query(x)

    weights = queries @ keys.transpose(-2, -1) * Size ** -0.5
    weights = weights.masked_fill(self.mask[:Block, :Block] == 0, float('-inf'))
    weights = nn.functional.softmax(weights, dim=-1)

    values = self.value(x)

    return weights @ values


class SimpleModel(nn.Module):

  def __init__(self, vocab_size, embedding_size, block_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
    self.position_embedding_table = nn.Embedding(block_size, embedding_size)
    self.attention_head = SelfAttentionHead(embedding_size, embedding_size, block_size)
    self.linear_layer = nn.Linear(embedding_size, vocab_size)

  def forward(self, idx, targets=None):
    Batch, Block = idx.shape

    token_embeddings = self.token_embedding_table(idx)
    positional_embeddings = self.position_embedding_table(torch.arange(Block))
    
    embeddings = token_embeddings + positional_embeddings
    embeddings = self.attention_head(embeddings)

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