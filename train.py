import tiktoken
import torch
import os.path
from pydracor import Corpus
from model.model import SimpleModel
from data.data import fetch_raw_data, get_raw_data, get_batch

# Loading pydracor API's corpus of french plays
corpus = Corpus('fre')
# We set a limit over the number of plays sampled
NUMBER_OF_PLAYS = 10
# Part of data used for validation
VALIDATION = 0.1

# Using tiktoken's r50k_base encoder
enc = tiktoken.get_encoding("r50k_base")
# Number of sequences to train at a time
BATCH_SIZE = 64
# Size of each sequence
BLOCK_SIZE = 128
# Number of episodes to train for
TRAIN_EPISODES = 10
# Number of steps in each episode
TRAIN_STEPS = 50
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