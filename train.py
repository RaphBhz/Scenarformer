import tiktoken
import torch
import os.path
from pydracor import Corpus
from model.model import SimpleModel
from data.data import fetch_raw_data, get_raw_data, get_batch, split_files

# Loading pydracor API's corpus of french plays
corpus = Corpus('fre')
# We set a limit over the number of plays sampled
NUMBER_OF_PLAYS = 10
# Part of data used for validation
VALIDATION_RATIO = 0.1

# Using tiktoken's r50k_base encoder
enc = tiktoken.get_encoding("r50k_base")
# Number of sequences to train at a time
BATCH_SIZE = 128
# Size of each sequence
BLOCK_SIZE = 256
# Size of embeddings
EMBEDDING_SIZE = 512
# Number of episodes to train for
TRAIN_EPISODES = 50
# Number of steps in each episode
TRAIN_STEPS = 200
# Number of steps in the validation
VALIDATION_STEPS = 10
# Learning rate
LEARNING_RATE = 1e-4


# Getting the raw text data
if os.path.isfile('data/data.txt'):
  data_raw = get_raw_data()
else:
  data_raw = fetch_raw_data(corpus, NUMBER_OF_PLAYS)

# We encode the data
data = torch.tensor(enc.encode(data_raw), dtype=torch.long)

# Splitting the data into test and validation sets
train_files, validation_files = split_files()

# Creating the model
m = SimpleModel(
  vocab_size=enc.n_vocab,
  embedding_size=512,
  block_size = BLOCK_SIZE
)

# Using torch's Adam optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)
# Using a ReduceLROnPlateau scheduler
scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Training the model
for episode in range(TRAIN_EPISODES):
  print(f'EPISODE {episode+1}')
  m.train()
  for inp, out in get_batch(train_files, BATCH_SIZE, BLOCK_SIZE, episode, TRAIN_STEPS):
    logits, loss = m(inp, out)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  # Validation
  m.eval()
  validation_loss = 0
  with torch.no_grad():
    for val_inp, val_out in get_batch(validation_files, BATCH_SIZE, BLOCK_SIZE, episode, VALIDATION_STEPS):
      val_logits, val_loss = m(val_inp, val_out)
      validation_loss += val_loss.item()

    # Compute average validation loss
    validation_loss /= VALIDATION_STEPS

  # Step the scheduler
  scheduler.step(validation_loss)

  print(f'Loss = {loss.item():.4f} Validation Loss = {validation_loss:.4f}')


# Printing a sample of generation
print(enc.decode(m.generate(torch.zeros((1, 1), dtype=torch.long), 500)[0].tolist()))