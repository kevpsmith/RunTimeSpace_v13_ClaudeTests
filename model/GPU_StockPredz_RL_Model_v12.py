import torch
import torch.nn as nn
import pytorch_lightning as pl

class TransformerPolicyNetwork(pl.LightningModule):
    def __init__(self, train_env, val_env, d_model, nhead, lr, num_layers, size=10):
        super(TransformerPolicyNetwork, self).__init__()
        self.env = train_env
        self.train_env = train_env
        self.val_env = val_env
        self.d_model = d_model
        self.nhead = nhead
        self.lr = lr
        self.num_layers = num_layers
        self.size = size
        # Projection layers for stock and regime features
        self.stock_projection = nn.Linear(12, d_model)  # First 12 features
        self.regime_projection = nn.Linear(3, d_model)  # Last 3 features
        # Multi-head attention layers
        self.stock_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.regime_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # Regime-specific attention layer
        self.regime_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model * self.env.num_sequences, d_model)
        self.select_head = nn.Linear(d_model, self.env.num_sequences) 
        self.decline_head = nn.Linear(d_model, self.env.num_sequences)
        self.double_digit_head = nn.Linear(d_model, self.env.num_sequences)

        # Initialize weights
        nn.init.xavier_uniform_(self.stock_projection.weight)
        nn.init.xavier_uniform_(self.regime_projection.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        nn.init.xavier_uniform_(self.select_head.weight)  # ADDED
        nn.init.xavier_uniform_(self.decline_head.weight)  # ADDED
        nn.init.xavier_uniform_(self.double_digit_head.weight)  # ADDED

        # self.automatic_optimization=True # might not need this line

    def forward(self, stock_state, regime_state):
        # Normalize inputs
        print("Stock State Shape:", stock_state.shape)
        print("Regime State Shape:", regime_state.shape)
        stock_state = (stock_state - stock_state.mean()) / (stock_state.std() + 1e-6)
        regime_state = (regime_state - regime_state.mean()) / (regime_state.std() + 1e-6)
        # Project stock and regime features separately
        stock_embedded = self.stock_projection(stock_state)
        regime_embedded = self.regime_projection(regime_state)
        if torch.isnan(stock_embedded).any() or torch.isinf(stock_embedded).any() or torch.isnan(regime_embedded).any() or torch.isinf(regime_embedded).any():
            print("NaN or Inf detected after input_projection")
        if stock_embedded.dim() == 4 and stock_embedded.size(1) == 1:
            stock_embedded = stock_embedded.squeeze(1)
        if regime_embedded.dim() == 4 and regime_embedded.size(1) == 1:
            regime_embedded = regime_embedded.squeeze(1)
        # Apply multi-head attention to stock and regime embeddings
        stock_context, _ = self.stock_attention(stock_embedded, stock_embedded, stock_embedded)
        regime_context, _ = self.regime_attention(regime_embedded, regime_embedded, regime_embedded)
        # Fuse stock and regime representations
        fused_representation = stock_context + regime_context
        # Process through transformer
        x = self.transformer(fused_representation)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf detected after transformer")
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.leaky_relu(self.fc(x))
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf detected after fc layer")

        # Output probabilities
        select_probs = torch.sigmoid(self.select_head(x))
        decline_probs = torch.sigmoid(self.decline_head(x))
        double_digit_probs = torch.sigmoid(self.double_digit_head(x))  # NEW OUTPUT
        
        return select_probs, decline_probs, double_digit_probs
    
    def training_step(self, batch, batch_idx):
        ##CODENOTE: Need to uncomment this to print and monitor parameter adjustments
        #for name, param in self.named_parameters():
        #    print(f"Before update - {name}: {param.flatten()[0].item()}")
        state, regime_state, action, reward = batch
        regime_state = regime_state.to(self.device)
        state = state.to(self.device)
        reward = reward.to(self.device)
        action = {k: v.to(self.device) for k, v in action.items()}

        print("State shape:", state.shape)
        print("Action select shape:", action['select'].shape)
        print("Reward:", reward)

        select_probs, decline_probs, double_digit_probs = self(state, regime_state)
        true_select_action = torch.tensor(self.env.get_true_select_action(self.size), dtype=torch.float32, device=self.device)
        true_decline_action = torch.tensor(self.env.get_true_decline_action(), dtype=torch.float32, device=self.device)
        true_double_digit = (self.env.current_growth_rates > 10.0).float().to(self.device)

        print("select_probs shape:", select_probs.shape)
        print("true_select_action shape:", true_select_action.shape)
        print("select_probs values:", select_probs)
        print("true_select_action values:", true_select_action)

        true_select_action = true_select_action.unsqueeze(0)
        true_decline_action = true_decline_action.unsqueeze(0)
        true_double_digit = true_double_digit.unsqueeze(0)

        assert select_probs.shape == true_select_action.shape, "Shape mismatch in select_probs and true_select_action"
        assert decline_probs.shape == true_decline_action.shape, "Shape mismatch in decline_probs and true_decline_action"

        select_loss = nn.BCELoss()(select_probs, true_select_action)
        decline_loss = nn.BCELoss()(decline_probs, true_decline_action)
        double_digit_loss = nn.BCELoss()(double_digit_probs, true_double_digit)  # NEW LOSS
        loss = (select_loss / 3) + (decline_loss / 3) + (double_digit_loss / 3) + reward
        self.log('train_loss', loss)
        ##CODENOTE: Need to uncomment this to print and monitor gradient changes
        #for name, param in self.named_parameters():
        #    if param.grad is not None:
        #        print(f"Gradient for {name}: {param.grad}")  # Check if gradients are non-zero
        #    else:
        #        print(f"No gradient computed for {name}")
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        return loss

    def on_train_epoch_start(self):
        self.env = self.train_env
    
    def on_validation_epoch_start(self):
        self.env = self.val_env
        self.env.current_growth_rates = self.env.growth_rates[-1]

    def validation_step(self, batch, batch_idx):
        state, regime_state, action, _ = batch
        state = state.to(self.device)
        regime_state = regime_state.to(self.device)
        action = {k: v.to(self.device) for k, v in action.items()}
        growth_rates = torch.tensor(self.env.current_growth_rates, dtype=torch.float32, device=self.device)
        # Forward pass with regime state
        select_probs, decline_probs, double_digit_probs = self(state, regime_state)

        selected_indices = torch.topk(select_probs, k=10).indices.squeeze()
        selected_return = torch.mean(growth_rates[selected_indices])
        self.log('selected_top_10_return', selected_return, on_epoch=True, prog_bar=True)
        
        true_select_action = torch.tensor(self.env.get_true_select_action(size=10), dtype=torch.float32, device=self.device)
        true_decline_action = torch.tensor(self.env.get_true_decline_action(), dtype=torch.float32, device=self.device)
        predicted_double_digit = (double_digit_probs > 0.5).float()  # Generate predictions
        true_double_digit = (growth_rates > 10.0).float()  # 1 if stock grew >10%, else 0
        double_digit_accuracy = (true_double_digit == predicted_double_digit).float().mean()
        true_select_action = true_select_action.unsqueeze(0)
        true_decline_action = true_decline_action.unsqueeze(0)
        true_double_digit = true_double_digit.unsqueeze(0)

        assert select_probs.shape == true_select_action.shape, "Shape mismatch in select_probs and true_select_action"
        assert decline_probs.shape == true_decline_action.shape, "Shape mismatch in decline_probs and true_decline_action"

        select_loss = nn.BCELoss()(select_probs, true_select_action)
        decline_loss = nn.BCELoss()(decline_probs, true_decline_action)
        double_digit_loss = nn.BCELoss()(double_digit_probs, true_double_digit)  # NEW
        val_loss = (select_loss / 3) + (decline_loss / 3) + (double_digit_loss / 3)  # Balanced weight
        self.log('double_digit_accuracy', double_digit_accuracy, on_epoch=True, prog_bar=True)
        self.log('val_loss', val_loss)

        #Come back and add something to track your false positives
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer