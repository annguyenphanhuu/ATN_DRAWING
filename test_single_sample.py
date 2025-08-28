import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import build_model, SmartNegativeSampler
from train import AdvancedDimensionDataset, collate_fn
import os

def load_best_model():
    """Load the best trained model."""
    NODE_FEATURE_DIM = 7
    EDGE_FEATURE_DIM = 4
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model = build_model(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, HIDDEN_DIM, EMBEDDING_DIM)
    
    # Load checkpoint
    if os.path.exists('best_model_v3.pth'):
        checkpoint = torch.load('best_model_v3.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully!")
        print(f"   Trained for: {checkpoint['epoch']} epochs")
        print(f"   Best F1 score: {checkpoint['val_f1']:.4f}")
        print(f"   Best validation loss: {checkpoint['val_loss']:.4f}")
    else:
        print("❌ No trained model found! Please run training first.")
        return None, device
    
    model = model.to(device)
    model.eval()
    return model, device

def show_available_samples():
    """Show available samples for testing."""
    print("📂 Loading available samples...")
    with open('data/prepared_data_v3.json', 'r') as f:
        full_data = json.load(f)
    
    print(f"\n📊 Available samples: {len(full_data)}")
    print("Sample details:")
    print("-" * 80)
    print(f"{'Index':<6} {'Drawing ID':<12} {'View Name':<20} {'Nodes':<6} {'Dim Lines':<10}")
    print("-" * 80)
    
    for i, sample in enumerate(full_data[:20]):  # Show first 20 samples
        drawing_id = sample['drawing_id']
        view_name = sample['view_name']
        num_nodes = sample['num_nodes']
        num_dim_lines = len(sample['dimension_lines'])
        print(f"{i:<6} {drawing_id:<12} {view_name:<20} {num_nodes:<6} {num_dim_lines:<10}")
    
    if len(full_data) > 20:
        print(f"... and {len(full_data) - 20} more samples")
    
    return full_data

def test_specific_sample(sample_index=None):
    """Test model on a specific sample."""
    print("🚀 Model Test on Specific Sample")
    print("=" * 50)
    
    # Load model
    model, device = load_best_model()
    if model is None:
        return
    
    # Load data
    full_data = show_available_samples()
    
    # Get sample index from user if not provided
    if sample_index is None:
        try:
            sample_index = int(input(f"\n🎯 Enter sample index (0-{len(full_data)-1}): "))
        except ValueError:
            print("❌ Invalid input! Using sample 0.")
            sample_index = 0
    
    if sample_index < 0 or sample_index >= len(full_data):
        print(f"❌ Invalid sample index! Using sample 0.")
        sample_index = 0
    
    # Get the specific sample
    selected_sample = full_data[sample_index]
    print(f"\n🔍 Testing Sample {sample_index}:")
    print(f"   Drawing ID: {selected_sample['drawing_id']}")
    print(f"   View Name: {selected_sample['view_name']}")
    print(f"   Nodes: {selected_sample['num_nodes']}")
    print(f"   Dimension Lines: {len(selected_sample['dimension_lines'])}")
    
    # Create dataset with just this sample
    negative_sampler = SmartNegativeSampler(strategy='mixed')
    test_dataset = AdvancedDimensionDataset([selected_sample], negative_sampler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Get the batch
    batch = next(iter(test_loader))
    
    if batch is None:
        print("❌ No valid data in this sample!")
        return
    
    # Test the model
    with torch.no_grad():
        # Move data to device
        node_features = batch['node_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_features = batch['edge_features'].to(device)
        edge_candidates = batch['edge_candidates'].to(device)
        labels = batch['labels'].to(device)
        node_positions = batch['node_positions'].to(device)
        
        print(f"\n📊 Sample Statistics:")
        print(f"   Node features shape: {node_features.shape}")
        print(f"   Edge features shape: {edge_features.shape}")
        print(f"   Edge candidates: {edge_candidates.shape[1]}")
        print(f"   Positive labels: {torch.sum(labels).item()}")
        
        # Get model predictions
        print(f"\n🧠 Running model inference...")
        logits = model(node_features, edge_index, edge_candidates, edge_features, node_positions)
        predictions = torch.sigmoid(logits)
        
        # Convert to numpy
        predictions_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        node_pos_np = node_positions.cpu().numpy()
        edge_candidates_np = edge_candidates.cpu().numpy()
        
        # Calculate metrics
        predicted_mask = predictions_np > 0.5
        positive_mask = labels_np == 1
        correct_predictions = (predicted_mask == positive_mask)
        
        accuracy = np.mean(correct_predictions)
        true_positives = np.sum(predicted_mask & positive_mask)
        false_positives = np.sum(predicted_mask & ~positive_mask)
        false_negatives = np.sum(~predicted_mask & positive_mask)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📈 Model Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   True Positives: {true_positives}")
        print(f"   False Positives: {false_positives}")
        print(f"   False Negatives: {false_negatives}")
        
        # Show top predictions
        print(f"\n🎯 Top 10 Predictions:")
        print("-" * 60)
        print(f"{'Rank':<5} {'Nodes':<10} {'Confidence':<12} {'True Label':<10} {'Correct':<8}")
        print("-" * 60)
        
        # Sort by confidence
        sorted_indices = np.argsort(predictions_np)[::-1]
        for i, idx in enumerate(sorted_indices[:10]):
            u, v = edge_candidates_np[:, idx]
            confidence = predictions_np[idx]
            true_label = "Dimension" if labels_np[idx] == 1 else "No Dim"
            predicted = "Dimension" if confidence > 0.5 else "No Dim"
            correct = "✓" if (confidence > 0.5) == (labels_np[idx] == 1) else "✗"
            
            print(f"{i+1:<5} ({u},{v}){'':<4} {confidence:.4f}{'':<6} {true_label:<10} {correct:<8}")
        
        # Create visualization
        print(f"\n🎨 Creating visualization...")
        
        plt.figure(figsize=(20, 6))
        
        # Plot 1: Original geometry
        plt.subplot(1, 4, 1)
        plt.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='lightblue', s=100, alpha=0.8, edgecolors='black')
        
        # Draw graph edges
        edge_index_np = edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            u, v = edge_index_np[:, i]
            plt.plot([node_pos_np[u, 0], node_pos_np[v, 0]], 
                    [node_pos_np[u, 1], node_pos_np[v, 1]], 
                    'gray', linewidth=1, alpha=0.5)
        
        # Add node labels
        for i, (x, y) in enumerate(node_pos_np):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title('Original Geometry')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Plot 2: Ground truth
        plt.subplot(1, 4, 2)
        plt.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='lightblue', s=100, alpha=0.8, edgecolors='black')
        
        for i, (u, v) in enumerate(edge_candidates_np.T):
            if positive_mask[i]:
                plt.plot([node_pos_np[u, 0], node_pos_np[v, 0]], 
                        [node_pos_np[u, 1], node_pos_np[v, 1]], 
                        'green', linewidth=3, alpha=0.8)
        
        plt.title(f'Ground Truth\n({np.sum(positive_mask)} dimension lines)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Plot 3: Predictions
        plt.subplot(1, 4, 3)
        plt.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='lightblue', s=100, alpha=0.8, edgecolors='black')
        
        for i, (u, v) in enumerate(edge_candidates_np.T):
            if predicted_mask[i]:
                color = 'green' if positive_mask[i] else 'red'
                plt.plot([node_pos_np[u, 0], node_pos_np[v, 0]], 
                        [node_pos_np[u, 1], node_pos_np[v, 1]], 
                        color, linewidth=3, alpha=0.8)
        
        plt.title(f'Predictions\n(Acc: {accuracy:.3f}, F1: {f1:.3f})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Plot 4: Confidence heatmap
        plt.subplot(1, 4, 4)
        plt.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='lightblue', s=100, alpha=0.8, edgecolors='black')
        
        # Draw all candidate edges with confidence-based colors
        for i, (u, v) in enumerate(edge_candidates_np.T):
            confidence = predictions_np[i]
            color = plt.cm.RdYlGn(confidence)  # Red to Green colormap
            linewidth = 1 + 3 * confidence
            plt.plot([node_pos_np[u, 0], node_pos_np[v, 0]], 
                    [node_pos_np[u, 1], node_pos_np[v, 1]], 
                    color=color, linewidth=linewidth, alpha=0.7)
        
        plt.title('Confidence Heatmap\n(Red=Low, Green=High)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('test_results', exist_ok=True)
        save_path = f'test_results/sample_{sample_index}_{selected_sample["drawing_id"]}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved to: {save_path}")
        
        plt.show()

if __name__ == '__main__':
    test_specific_sample()
