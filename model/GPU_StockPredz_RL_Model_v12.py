import torch
import torch.nn as nn
import pytorch_lightning as pl


class TransformerPolicyNetwork(pl.LightningModule):
    def __init__(self, num_sequences, d_model, nhead, lr, num_layers, size=10):
        super(TransformerPolicyNetwork, self).__init__()
        self.num_sequences = num_sequences
        self.d_model = d_model
        self.nhead = nhead
        self.lr = lr
        self.num_layers = num_layers
        self.size = size

        self.stock_projection = nn.Linear(12, d_model)
        self.regime_projection = nn.Linear(3, d_model)

        self.stock_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.regime_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model * num_sequences, d_model)
        self.select_head = nn.Linear(d_model, num_sequences)
        self.decline_head = nn.Linear(d_model, num_sequences)
        self.double_digit_head = nn.Linear(d_model, num_sequences)

        nn.init.xavier_uniform_(self.stock_projection.weight)
        nn.init.xavier_uniform_(self.regime_projection.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.select_head.weight)
        nn.init.xavier_uniform_(self.decline_head.weight)
        nn.init.xavier_uniform_(self.double_digit_head.weight)

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
        double_digit_probs = torch.sigmoid(self.double_digit_head(x))

        return select_probs, decline_probs, double_digit_probs

    def _compute_true_labels(self, growth_rates):
        true_select = torch.zeros_like(growth_rates)
        if growth_rates.shape[-1] >= self.size:
            top_k_indices = torch.topk(growth_rates, k=self.size, dim=-1).indices
            true_select.scatter_(-1, top_k_indices, 1.0)
        true_decline = (growth_rates < 0).float()
        true_double_digit = (growth_rates > 10.0).float()
        return true_select, true_decline, true_double_digit

    def training_step(self, batch, batch_idx):
        state, regime_state, growth_rates = batch
        state = state.to(self.device)
        regime_state = regime_state.to(self.device)
        growth_rates = growth_rates.to(self.device)

        select_probs, decline_probs, double_digit_probs = self(state, regime_state)
        true_select, true_decline, true_double_digit = self._compute_true_labels(growth_rates)

        select_loss = nn.BCELoss()(select_probs, true_select)
        decline_loss = nn.BCELoss()(decline_probs, true_decline)
        double_digit_loss = nn.BCELoss()(double_digit_probs, true_double_digit)
        loss = (select_loss + decline_loss + double_digit_loss) / 3

        self.log('train_loss', loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        return loss

    def validation_step(self, batch, batch_idx):
        state, regime_state, growth_rates = batch
        state = state.to(self.device)
        regime_state = regime_state.to(self.device)
        growth_rates = growth_rates.to(self.device)

        select_probs, decline_probs, double_digit_probs = self(state, regime_state)
        true_select, true_decline, true_double_digit = self._compute_true_labels(growth_rates)

        selected_indices = torch.topk(select_probs, k=10, dim=-1).indices
        selected_return = torch.mean(torch.gather(growth_rates, -1, selected_indices))
        self.log('selected_top_10_return', selected_return, on_epoch=True, prog_bar=True)

        predicted_double_digit = (double_digit_probs > 0.5).float()
        double_digit_accuracy = (true_double_digit == predicted_double_digit).float().mean()
        self.log('double_digit_accuracy', double_digit_accuracy, on_epoch=True, prog_bar=True)

        select_loss = nn.BCELoss()(select_probs, true_select)
        decline_loss = nn.BCELoss()(decline_probs, true_decline)
        double_digit_loss = nn.BCELoss()(double_digit_probs, true_double_digit)
        val_loss = (select_loss + decline_loss + double_digit_loss) / 3
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
