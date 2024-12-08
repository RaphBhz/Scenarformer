import torch
from torch import nn

class SimpleModel(nn.Module):

  def __init__(self, vocab_size, embedding_size, block_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
    self.position_embedding_table = nn.Embedding(block_size, embedding_size)
    self.linear_layer = nn.Linear(embedding_size, vocab_size)

  def forward(self, idx, targets=None):
    Batch, Block = idx.shape

    token_embeddings = self.token_embedding_table(idx)
    positional_embeddings = self.position_embedding_table(torch.arrange(Block))
    embeddings = token_embeddings + positional_embeddings
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
    for _ in range(max):
      logits, loss = self(idx)
      logits = logits [:,-1,:]
      probs = nn.functional.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx