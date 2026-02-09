import gym
from gym import spaces
import numpy as np
import torch

class SequenceSelectionEnv(gym.Env):
    def __init__(self, data, growth_rates):
        super(SequenceSelectionEnv, self).__init__()
        self.data = data
        self.growth_rates = growth_rates
        self.num_episodes = data.shape[0]
        self.num_sequences = data.shape[1]
        self.sequence_length = data.shape[2]
        self.current_episode = 0
        self.observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'),
            shape=(self.num_sequences, self.sequence_length),
            dtype=np.float32  # Fixed: Use NumPy dtype only
        )
        self.state = None
        self.current_growth_rates = None

    def reset(self):
        if self.current_episode >= self.num_episodes:
            self.current_episode = 0

        # Extract base stock state (excluding last 3 columns)
        base_state = torch.tensor(self.data[self.current_episode, :, :-3], dtype=torch.float32, device='cuda')

        # Extract regime features (last 3 columns)
        regime_state = torch.tensor(self.data[self.current_episode, :, -3:], dtype=torch.float32, device='cuda')

        # Save state and regime features separately
        self.state = base_state
        self.regime_state = regime_state  # Store separately for model processing
        self.current_growth_rates = torch.tensor(self.growth_rates[self.current_episode], dtype=torch.float32, device='cuda')

        print("Current episode:", self.current_episode)
        print("Regime Features Shape:", self.regime_state.shape)  # Should be (num_sequences, 3)

        self.current_episode += 1
        return self.state, self.regime_state
    
    def step(self, action):
        select_action = action["select"]
        decline_action = action["decline"]
        double_digit_action = action["double_digit"]

        # True labels
        true_double_digit = (self.current_growth_rates > 10.0).float()

        # Compute rewards
        selection_reward = torch.sum(select_action * self.current_growth_rates) / torch.sum(select_action + 1e-6)
        decline_penalty = -torch.sum(decline_action * self.current_growth_rates) / torch.sum(decline_action + 1e-6)

        incorrect_double_digit = torch.sum(torch.abs(double_digit_action - true_double_digit)) / self.num_sequences
        double_digit_reward = -incorrect_double_digit  # Penalize wrong predictions

        total_reward = (
            0.4 * selection_reward +  
            0.2 * decline_penalty +    
            0.4 * double_digit_reward  # NEW DOUBLE-DIGIT REWARD
        )

        return self.state, total_reward, True, {}

    def get_true_select_action(self, size=10):
        if self.num_sequences < size:
            raise ValueError("Number of sequences must be at least 10 for top-k selection.")
        true_select_action = torch.zeros(self.num_sequences, dtype=torch.int, device='cuda')
        self.current_growth_rates = torch.tensor(self.current_growth_rates, dtype=torch.float32, device='cuda')
        top_k_indices = torch.topk(self.current_growth_rates, k=size).indices
        true_select_action[top_k_indices] = 1
        return true_select_action

    def get_true_decline_action(self):
        true_decline_action = (self.current_growth_rates < 0).int().to('cuda')
        return true_decline_action
    
    def render(self, mode='human'):
        print(f"State: {self.state}, Growth Rates: {self.growth_rates}")

