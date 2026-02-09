from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import torch
import random

class SequenceSelectionDataset(IterableDataset):
    def __init__(self, env, policy_model, num_episodes=500, randomize_series=True):
        self.env = env
        self.policy_model = policy_model
        self.num_episodes = num_episodes
        self.randomize_series = randomize_series

    def __iter__(self):
        episode_indices = list(range(self.num_episodes))
        random.shuffle(episode_indices)  # ✅ Shuffle episodes before training

        for episode_idx in episode_indices:  # Iterate over shuffled episodes
            # Reset the environment and get the initial state for this episode
            state, regime_state = self.env.reset()

            # Randomize if enabled
            if self.randomize_series:
                perm = torch.randperm(state.shape[0], device='cuda')
                state = state[perm]
                regime_state = regime_state[perm]

            # Convert to tensors
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to('cuda')
            regime_tensor = torch.tensor(regime_state, dtype=torch.float32).unsqueeze(0).to('cuda')

            # Use the policy model to generate probabilities without gradients
            with torch.no_grad():
                select_probs, decline_probs, double_digit_probs = self.policy_model(state_tensor, regime_tensor)

            # Sample actions from the generated probabilities (binary 0/1 decisions)
            top_select_indices = torch.topk(select_probs, k=10).indices
            select_action = torch.zeros(self.env.num_sequences, dtype=torch.float32).to('cuda')
            select_action[top_select_indices] = 1
            
            decline_threshold = 0.5
            decline_action = (decline_probs > decline_threshold).float().to('cuda')

            double_digit_threshold = 0.5  # Adjustable if needed
            double_digit_action = (double_digit_probs > double_digit_threshold).float().to('cuda')

            # Create the action dictionary based on model predictions
            train_actions = {
                'select': select_action,
                'decline': decline_action,
                'double_digit': double_digit_action  # NEW ACTION
            }

            # Handle randomized training properly
            if self.randomize_series:
                inv_perm = torch.argsort(perm)
                env_state = state[inv_perm]
                env_actions = {
                    'select': select_action[inv_perm.cpu()],
                    'decline': decline_action.squeeze()[inv_perm.cpu()],
                    'double_digit': double_digit_action.squeeze()[inv_perm.cpu()]  # ✅ Fix for KeyError
                }
            else:
                env_state = state
                env_actions = train_actions

            # Calculate the reward based on the model's predicted action
            _, reward, _, _ = self.env.step(env_actions)

            # Yield the state, action, and reward for training
            yield (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                torch.tensor(regime_state, dtype=torch.float32).unsqueeze(0),
                {key: torch.tensor(val, dtype=torch.float32) for key, val in train_actions.items()},
                torch.tensor(reward, dtype=torch.float32).to('cuda')
            )