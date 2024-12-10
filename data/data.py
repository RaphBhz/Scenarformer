import random
import torch
import os.path
from requests.exceptions import HTTPError
from pydracor import Play, Corpus
from concurrent.futures import ThreadPoolExecutor


def save_play(path, name, text, overwrite):
  file_path = f'{path+name}.txt'

  if not overwrite and os.path.isfile(file_path):
    print(f'Play {name} is already saved')
    return False

  with open(file_path, 'w') as file:
    file.seek(0)
    file.write(f'{name.upper()}\n\n{text}')
    file.truncate()

  print(f'Play {name} was successfully saved')
  return True


def fetch_play_text(play_id, save_to = 'data/plays/', overwrite = False, tries = 0):
  if tries >= 5:
    print('Maximum attempts exceeded while trying to fetch PyDraCor data')
    return None, None
    
  try:
    play = Play(play_id)
    name = play.name

    if not overwrite and os.path.isfile(file_path):
      print(f'Play {name} is already saved')
      return None, None

    text = play.spoken_text()

    save_play(save_to, name, text, overwrite)
    return name, text
  except HTTPError:
    tries += 1
    return fetch_play_text(play_id, save_to, overwrite, tries)


def fetch_raw_data(corpus, play_num, path = 'data/plays/', overwrite = False):
    # We sample a set of randomly chosen plays
    play_ids = corpus.play_ids()
    random_play_ids = play_ids if play_num >= len(play_ids) else random.sample(play_ids, play_num)

    # Using multithreading for faster fetching
    with ThreadPoolExecutor() as executor:
      results = executor.map(
        fetch_play_text,
        random_play_ids,
        [path] * len(random_play_ids),
        [overwrite] * len(random_play_ids)
      )

    extracted_num = len(list(filter(lambda x: x!= (None, None), results)))
    print(f'Successfully fetched {extracted_num}/{play_num} plays')


def load_play_text(path):
  with open(path) as file:
    data = file.read()
    return data


def get_batch(play_files, batch_size, block_size, episode, steps_num):
    num_plays = len(play_files)
    # Select a play based on episode index
    play_idx = episode % num_plays
    play_text = load_play_text(play_files[play_idx])
    
    # Convert the play text to tokens
    tokens = torch.tensor(enc.encode(play_text), dtype=torch.long)
    
    # We want to get a specific number of batches for this episode
    for _ in range(steps_num):
        # Generate random starting indices for the batch
        start_idx = torch.randint(len(tokens) - block_size, (batch_size,))
        inputs = torch.stack([tokens[i:i+block_size] for i in start_idx])
        outputs = torch.stack([tokens[i+1:i+block_size+1] for i in start_idx])
        yield inputs, outputs


def split_files(play_dir = 'data/plays/', validation_ratio=0.1):
  # Get all .txt files in the directory
  all_files = [os.path.join(root, file)
                for root, _, files in os.walk(play_dir)
                for file in files if file.endswith('.txt')]
  
  # Shuffle the files
  random.shuffle(all_files)
  
  # Calculate the split index
  split_idx = int(len(all_files) * (1 - validation_ratio))
  
  # Split into training and validation
  train_files = all_files[:split_idx]
  val_files = all_files[split_idx:]
  
  return train_files, val_files


corpus = Corpus('fre')
fetch_raw_data(corpus, 50, path='plays/')