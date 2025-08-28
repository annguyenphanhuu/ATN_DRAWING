import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import build_model, SmartNegativeSampler
from train import AdvancedDimensionDataset, collate_fn, evaluate_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_model(model_path, node_feature_dim, edge_feature_dim, hidden_dim, embedding_dim):
    """Load a trained model from checkpoint."""
    model = build_model(node_feature_dim, edge_feature_dim, hidden_dim, embedding_dim)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def visualize_predictions(model, data_loader, device, num_samples=5):
    """Visualize model predictions on sample data."""
    model.eval()
    samples_shown = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if batch is None or samples_shown >= num_samples:
                continue
                
            # Move data to device
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_features = batch['edge_features'].to(device)
            edge_candidates = batch['edge_candidates'].to(device)
            labels = batch['labels'].to(device)
            node_positions = batch['node_positions'].to(device)
            
            # Get predictions
            logits = model(node_features, edge_index, edge_candidates, edge_features, node_positions)
            predictions = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()
            node_pos_np = node_positions.cpu().numpy()
            edge_candidates_np = edge_candidates.cpu().numpy()
            
            # Create visualization
            plt.figure(figsize=(12, 5))
            
            # Plot 1: Ground truth
            plt.subplot(1, 2, 1)
            plt.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='blue', s=50, alpha=0.7)
            
            # Draw positive edges (ground truth)
            positive_mask = labels_np == 1
            for i, (u, v) in enumerate(edge_candidates_np.T):
                if positive_mask[i]:
                    plt.plot([node_pos_np[u, 0], node_pos_np[v, 0]], 
                            [node_pos_np[u, 1], node_pos_np[v, 1]], 
                            'g-', linewidth=2, alpha=0.8, label='True Dimension' if i == 0 else "")
            
            plt.title('Ground Truth Dimension Lines')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Predictions
            plt.subplot(1, 2, 2)
            plt.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='blue', s=50, alpha=0.7)
            
            # Draw predicted edges (threshold = 0.5)
            predicted_mask = predictions > 0.5
            correct_predictions = (predicted_mask == positive_mask)
            
            for i, (u, v) in enumerate(edge_candidates_np.T):
                if predicted_mask[i]:
                    color = 'g' if correct_predictions[i] else 'r'
                    label = 'Correct Prediction' if correct_predictions[i] and i == 0 else 'Wrong Prediction' if not correct_predictions[i] and i == 0 else ""
                    plt.plot([node_pos_np[u, 0], node_pos_np[v, 0]], 
                            [node_pos_np[u, 1], node_pos_np[v, 1]], 
                            color + '-', linewidth=2, alpha=0.8, label=label)
            
            plt.title(f'Model Predictions (Accuracy: {np.mean(correct_predictions):.2f})')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'prediction_sample_{samples_shown + 1}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            samples_shown += 1

def analyze_model_performance(model, data_loader, device):
    """Comprehensive analysis of model performance."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    
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
            probabilities = torch.sigmoid(logits).cpu().numpy()
            
            # Collect data
            all_predictions.extend((probabilities > 0.5).astype(int))
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(probabilities)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['No Dimension', 'Dimension Line']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Dimension', 'Dimension Line'],
                yticklabels=['No Dimension', 'Dimension Line'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Confidence distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_confidences[all_labels == 0], bins=50, alpha=0.7, label='No Dimension', color='red')
    plt.hist(all_confidences[all_labels == 1], bins=50, alpha=0.7, label='Dimension Line', color='green')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by True Label')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    for threshold in thresholds:
        pred_at_threshold = (all_confidences > threshold).astype(int)
        accuracy = np.mean(pred_at_threshold == all_labels)
        accuracies.append(accuracy)
    
    plt.plot(thresholds, accuracies, 'b-', linewidth=2)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold (0.5)')
    plt.xlabel('Prediction Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Prediction Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return all_predictions, all_labels, all_confidences

if __name__ == '__main__':
    # Configuration
    NODE_FEATURE_DIM = 7
    EDGE_FEATURE_DIM = 4
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 128
    MODEL_PATH = 'best_model_v3.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading test data...")
    with open('data/prepared_data_v3.json', 'r') as f:
        full_data = json.load(f)
    
    # Use a subset for evaluation (last 20% as test set)
    test_data = full_data[-len(full_data)//5:]
    print(f"Evaluating on {len(test_data)} samples")
    
    # Create test dataset
    negative_sampler = SmartNegativeSampler(strategy='mixed')
    test_dataset = AdvancedDimensionDataset(test_data, negative_sampler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Load model
    print("Loading trained model...")
    model, checkpoint = load_model(MODEL_PATH, NODE_FEATURE_DIM, EDGE_FEATURE_DIM, HIDDEN_DIM, EMBEDDING_DIM)
    model = model.to(device)
    
    print(f"Model trained for {checkpoint['epoch']} epochs")
    print(f"Best validation F1: {checkpoint['val_f1']:.4f}")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    
    # Evaluate model
    print("\nEvaluating model performance...")
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss, test_precision, test_recall, test_f1, test_auc = evaluate_model(
        model, test_loader, criterion, device
    )
    
    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    # Detailed analysis
    print("\nPerforming detailed analysis...")
    predictions, labels, confidences = analyze_model_performance(model, test_loader, device)
    
    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, test_loader, device, num_samples=3)
    
    print("\nEvaluation complete! Check the generated plots for detailed analysis.")
