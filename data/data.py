import random
import torch
from pydracor import Play

def fetch_raw_data(corpus, play_num):
  with open('data/data.txt', 'w') as file:
    file.seek(0)

    # We will simply register all the data in a single string
    data_raw = ''
    # We sample a set of randomly chosen plays
    play_ids = corpus.play_ids()
    random_play_ids = random.sample(
      play_ids,
      len(play_ids) + 1 if play_num >= len(play_ids) else play_num
    )

    # Extractnig the plays texts
    for id in random_play_ids:
      # Extracting a play's spoken text with the pydracor API
      play = Play(id)
      data_raw += f'\n{play.spoken_text()}'

    # Logging progress
    print(f'Extracted "{play.name}"')

    file.write(data_raw)
    file.truncate()

    return data_raw


def get_raw_data():
  with open('data/data.txt') as file:
    data = file.read()
    return data if data else fetch_raw_data(1)


def get_batch(data, batch_size, block_size):
  start_idx = torch.randint(len(data) - block_size, (batch_size,))

  inputs = torch.stack([data[i:i+block_size] for i in start_idx])
  outputs = torch.stack([data[i+1:i+block_size+1] for i in start_idx])

  return inputs, outputs