import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from model import build_model, SmartNegativeSampler
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedDimensionDataset(Dataset):
    """
    Advanced dataset with rich geometric features and intelligent negative sampling.
    """
    def __init__(self, data, negative_sampler=None):
        self.data = data
        self.negative_sampler = negative_sampler or SmartNegativeSampler()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        view_data = self.data[idx]

        if not view_data['dimension_lines'] or view_data['num_nodes'] < 2:
            return None

        # Node features (rich geometric features)
        node_features = torch.tensor(view_data['node_features'], dtype=torch.float)
        num_nodes = view_data['num_nodes']

        # Graph structure from data processing
        edge_index = torch.tensor(view_data['edge_index'], dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(view_data['edge_features'], dtype=torch.float)

        # Node positions (first 2 features are normalized x, y)
        node_positions = node_features[:, :2]

        # Positive edges (dimension lines)
        positive_edges = view_data['dimension_lines']

        # Smart negative sampling
        negative_edges = self.negative_sampler.sample_negatives(
            positive_edges, node_positions, num_nodes, len(positive_edges)
        )

        if not negative_edges:
            return None

        # Combine positive and negative edges
        all_edges = positive_edges + negative_edges
        edge_candidates = torch.tensor(all_edges, dtype=torch.long).t().contiguous()

        # Labels
        labels = torch.cat([
            torch.ones(len(positive_edges), dtype=torch.float),
            torch.zeros(len(negative_edges), dtype=torch.float)
        ])

        return {
            'drawing_id': view_data.get('drawing_id', 'unknown'),
            'view_name': view_data.get('view_name', 'unknown'),
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'edge_candidates': edge_candidates,
            'labels': labels,
            'node_positions': node_positions,
            'num_nodes': num_nodes
        }

def collate_fn(batch):
    """Custom collate function that correctly batches graph data and metadata."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # If batch size is 1, the logic is simple
    if len(batch) == 1:
        # Still wrap strings in a list to be consistent
        item = batch[0]
        item['drawing_id'] = [item['drawing_id']]
        item['view_name'] = [item['view_name']]
        return item

    # For batch_size > 1, we need to handle things carefully
    # This is a simplified collate for this project, assuming batch_size=1 for evaluation
    # A more robust implementation would use torch_geometric.data.Batch

    # For now, we will just return a list of samples if batch size > 1
    # and the evaluation loop will handle it.
    return batch

def evaluate_model(model, data_loader, criterion, device):
    """Comprehensive model evaluation with multiple metrics."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue

            # Move data to device
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_features = batch['edge_features'].to(device)
            edge_candidates = batch['edge_candidates'].to(device)
            labels = batch['labels'].to(device)
            node_positions = batch['node_positions'].to(device)

            # Forward pass
            logits = model(node_features, edge_index, edge_candidates, edge_features, node_positions)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Collect predictions and labels
            predictions = torch.sigmoid(logits).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0

    # Calculate metrics
    if len(all_predictions) > 0:
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Binary predictions (threshold = 0.5)
        binary_preds = (all_predictions > 0.5).astype(int)

        precision = precision_score(all_labels, binary_preds, zero_division=0)
        recall = recall_score(all_labels, binary_preds, zero_division=0)
        f1 = f1_score(all_labels, binary_preds, zero_division=0)

        # AUC score
        try:
            auc = roc_auc_score(all_labels, all_predictions)
        except ValueError:
            auc = 0.0

        return avg_loss, precision, recall, f1, auc
    else:
        return avg_loss, 0, 0, 0, 0

if __name__ == '__main__':
    # Enhanced hyperparameters
    NODE_FEATURE_DIM = 7  # (x, y, min_dist, mean_dist, angular_variance, density_01, density_05)
    EDGE_FEATURE_DIM = 4  # (connection_type, length, angle, strength)
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 128
    LEARNING_RATE = 0.0005
    EPOCHS = 150
    VAL_SPLIT = 0.2
    WEIGHT_DECAY = 1e-5
    PATIENCE = 20  # Early stopping patience

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load processed data
    print("Loading processed data...")
    with open('data/prepared_data_v3.json', 'r') as f:
        full_data = json.load(f)

    print(f"Loaded {len(full_data)} samples")

    # Filter out samples with insufficient data
    filtered_data = [sample for sample in full_data if
                    len(sample['dimension_lines']) > 0 and sample['num_nodes'] >= 2]
    print(f"After filtering: {len(filtered_data)} samples")

    # Split data
    val_size = int(len(filtered_data) * VAL_SPLIT)
    train_size = len(filtered_data) - val_size
    train_data, val_data = random_split(filtered_data, [train_size, val_size])

    # Create datasets with smart negative sampling
    negative_sampler = SmartNegativeSampler(strategy='mixed')
    train_dataset = AdvancedDimensionDataset(train_data, negative_sampler)
    val_dataset = AdvancedDimensionDataset(val_data, negative_sampler)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Build model
    print("Building model...")
    model = build_model(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, HIDDEN_DIM, EMBEDDING_DIM)
    model = model.to(device)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Loss function with class weighting
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training tracking
    best_val_f1 = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0
        train_batches = 0

        for batch in train_loader:
            if batch is None:
                continue

            # Move data to device
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_features = batch['edge_features'].to(device)
            edge_candidates = batch['edge_candidates'].to(device)
            labels = batch['labels'].to(device)
            node_positions = batch['node_positions'].to(device)

            optimizer.zero_grad()
            logits = model(node_features, edge_index, edge_candidates, edge_features, node_positions)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        train_losses.append(avg_train_loss)

        # Validation phase
        val_loss, val_precision, val_recall, val_f1, val_auc = evaluate_model(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
        print(f'  F1: {val_f1:.4f}, AUC: {val_auc:.4f}')

        # Save best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss
            }, 'best_model_v3.pth')
            print(f'  -> New best model saved with F1: {best_val_f1:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        print("-" * 60)

    print(f"Training complete. Best F1 score: {best_val_f1:.4f}")
    print("Best model saved to 'best_model_v3.pth'")
