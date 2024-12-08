import random
import tiktoken
import torch
from pydracor import Corpus, Play
from model.model import SimpleModel

# Loading pydracor API's corpus of french plays
corpus = Corpus('fre')
# We set a limit over the number of plays sampled
NUMBER_OF_PLAYS = 1
# Part of data used for validation
VALIDATION = 0.1

# Using tiktoken's r50k_base encoder
enc = tiktoken.get_encoding("r50k_base")
# Number of sequences to train at a time
BATCH_SIZE = 16
# Size of each sequence
BLOCK_SIZE = 32
# Number of episodes to train for
TRAIN_EPISODES = 5
# Number of steps in each episode
TRAIN_STEPS = 10
# Learning rate
LEARNING_RATE = 1e-3


def get_raw_data():
  # We will simply register all the data in a single string
  data_raw = ''
  # We sample a set of randomly chosen plays
  random_play_ids = random.sample(corpus.play_ids(), NUMBER_OF_PLAYS)

  # Extractnig the plays texts
  for id in random_play_ids:
    # Extracting a play's spoken text with the pydracor API
    play = Play(id)
    data_raw += f'\n{play.spoken_text()}'

  # Logging progress
  print(f'Extracted "{play.name}"')

  return data_raw


def get_batch(data, batch_size, block_size):
  start_idx = torch.randint(len(data) - block_size, (batch_size,))

  inputs = torch.stack([data[i:i+block_size] for i in start_idx])
  outputs = torch.stack([data[i+1:i+block_size+1] for i in start_idx])

  return inputs, outputs


# Getting the raw text data
data_raw = get_raw_data()

# We encode the data
data = torch.tensor(enc.encode(data_raw), dtype=torch.long)

# Splitting the data into test and validation sets
n = int(VALIDATION * len(data))
train = data[:-n]
validation = data[-n:]

# Generating a batch of inputs and outputs
batch_input, batch_output = get_batch(train, BATCH_SIZE, BLOCK_SIZE)

# Creating the model
m = SimpleModel(
  vocab_size=enc.n_vocab,
  embedding_size=512,
  block_size = BLOCK_SIZE
)
# Using torch's Adam optimizer
optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)

# Training the model
for episode in range(TRAIN_EPISODES):
  for steps in range(TRAIN_STEPS):
    inp, out = get_batch(train, BATCH_SIZE, BLOCK_SIZE)

    logits, loss = m(inp, out)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  print(f'Episode {episode+1}: Loss = {loss.item()}')


# Printing a sample of generation
print(enc.decode(m.generate(torch.zeros((1, 1), dtype=torch.long), 500)[0].tolist()))