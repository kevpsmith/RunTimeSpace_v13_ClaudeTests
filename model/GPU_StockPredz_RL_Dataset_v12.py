from torch.utils.data import IterableDataset
import torch
import random


class SequenceSelectionDataset(IterableDataset):
    def __init__(self, data, growth_rates, num_episodes, randomize_series=True):
        self.data = data
        self.growth_rates = growth_rates
        self.num_episodes = num_episodes
        self.num_sequences = data.shape[1]
        self.randomize_series = randomize_series

    def __iter__(self):
        episode_indices = list(range(self.num_episodes))
        random.shuffle(episode_indices)

        for episode_idx in episode_indices:
            base_state = torch.tensor(self.data[episode_idx, :, :-3], dtype=torch.float32, device='cuda')
            regime_state = torch.tensor(self.data[episode_idx, :, -3:], dtype=torch.float32, device='cuda')
            growth = torch.tensor(self.growth_rates[episode_idx], dtype=torch.float32, device='cuda')

            if self.randomize_series:
                perm = torch.randperm(base_state.shape[0], device='cuda')
                base_state = base_state[perm]
                regime_state = regime_state[perm]
                growth = growth[perm]

            yield (
                base_state.unsqueeze(0),
                regime_state.unsqueeze(0),
                growth.unsqueeze(0)
            )
