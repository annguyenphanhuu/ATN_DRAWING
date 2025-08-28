import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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
        print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
        print(f"   Best F1 score: {checkpoint['val_f1']:.4f}")
        print(f"   Best val loss: {checkpoint['val_loss']:.4f}")
    else:
        print("❌ No trained model found! Please run training first.")
        return None, device
    
    model = model.to(device)
    model.eval()
    return model, device

def visualize_single_sample(model, batch, device, save_path=None, json_path=None):
    """Visualize predictions for a single sample and save results as JSON."""
    if batch is None:
        print("❌ No valid batch data")
        return None

    with torch.no_grad():
        # Move data to device
        node_features = batch['node_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_features = batch['edge_features'].to(device)
        edge_candidates = batch['edge_candidates'].to(device)
        labels = batch['labels'].to(device)
        node_positions = batch['node_positions'].to(device)

        # Get model predictions
        logits = model(node_features, edge_index, edge_candidates, edge_features, node_positions)
        predictions = torch.sigmoid(logits).cpu().numpy()

        # Convert to numpy for visualization
        labels_np = labels.cpu().numpy()
        node_pos_np = node_positions.cpu().numpy()
        edge_candidates_np = edge_candidates.cpu().numpy()
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Original geometry (graph structure)
        ax1 = axes[0]
        ax1.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='lightblue', s=100, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Draw graph edges (geometric structure)
        edge_index_np = edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            u, v = edge_index_np[:, i]
            ax1.plot([node_pos_np[u, 0], node_pos_np[v, 0]], 
                    [node_pos_np[u, 1], node_pos_np[v, 1]], 
                    'gray', linewidth=1, alpha=0.5)
        
        # Add node indices
        for i, (x, y) in enumerate(node_pos_np):
            ax1.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, fontweight='bold')
        
        ax1.set_title('Original Geometry\n(Graph Structure)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: Ground Truth Dimension Lines
        ax2 = axes[1]
        ax2.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='lightblue', s=100, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Draw positive edges (ground truth dimension lines)
        positive_mask = labels_np == 1
        positive_count = 0
        for i, (u, v) in enumerate(edge_candidates_np.T):
            if positive_mask[i]:
                ax2.plot([node_pos_np[u, 0], node_pos_np[v, 0]], 
                        [node_pos_np[u, 1], node_pos_np[v, 1]], 
                        'green', linewidth=3, alpha=0.8)
                
                # Add edge label
                mid_x = (node_pos_np[u, 0] + node_pos_np[v, 0]) / 2
                mid_y = (node_pos_np[u, 1] + node_pos_np[v, 1]) / 2
                ax2.annotate(f'D{positive_count}', (mid_x, mid_y), 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                           fontsize=8, ha='center')
                positive_count += 1
        
        ax2.set_title(f'Ground Truth\n({positive_count} Dimension Lines)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Plot 3: Model Predictions
        ax3 = axes[2]
        ax3.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='lightblue', s=100, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Draw predicted edges with confidence colors
        predicted_mask = predictions > 0.5
        correct_predictions = (predicted_mask == positive_mask)
        
        true_positives = 0
        false_positives = 0
        
        for i, (u, v) in enumerate(edge_candidates_np.T):
            confidence = predictions[i]
            
            if predicted_mask[i]:  # Model predicts dimension line
                if positive_mask[i]:  # True positive
                    color = 'green'
                    label_text = f'TP{true_positives}'
                    true_positives += 1
                else:  # False positive
                    color = 'red'
                    label_text = f'FP{false_positives}'
                    false_positives += 1
                
                # Line thickness based on confidence
                linewidth = 1 + 3 * confidence
                
                ax3.plot([node_pos_np[u, 0], node_pos_np[v, 0]], 
                        [node_pos_np[u, 1], node_pos_np[v, 1]], 
                        color, linewidth=linewidth, alpha=0.8)
                
                # Add confidence label
                mid_x = (node_pos_np[u, 0] + node_pos_np[v, 0]) / 2
                mid_y = (node_pos_np[u, 1] + node_pos_np[v, 1]) / 2
                ax3.annotate(f'{label_text}\n{confidence:.2f}', (mid_x, mid_y), 
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor="lightgreen" if color=='green' else "lightcoral", 
                                   alpha=0.7),
                           fontsize=7, ha='center')
        
        # Calculate metrics
        accuracy = np.mean(correct_predictions)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / np.sum(positive_mask) if np.sum(positive_mask) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        ax3.set_title(f'Model Predictions\nAcc: {accuracy:.2f}, P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('X coordinate')
        ax3.set_ylabel('Y coordinate')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=3, label='True Positive (Correct)'),
            Line2D([0], [0], color='red', lw=3, label='False Positive (Wrong)'),
            Line2D([0], [0], color='gray', lw=1, alpha=0.5, label='Graph Structure')
        ]
        fig.legend(legend_elements, [elem.get_label() for elem in legend_elements], 
                  loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved visualization to {save_path}")
        plt.close(fig)  # Close the figure to free memory and prevent display

        # Post-processing to DimClient format
        predicted_lines = []
        for i, (u, v) in enumerate(edge_candidates_np.T):
            if predictions[i] > 0.5:  # If the model predicts a dimension line
                start_node_pos = node_pos_np[u]
                end_node_pos = node_pos_np[v]
                line_data = {
                    "StartPoint": {"X": float(start_node_pos[0]), "Y": float(start_node_pos[1])},
                    "EndPoint": {"X": float(end_node_pos[0]), "Y": float(end_node_pos[1])},
                    "StartPointOrigin": None,
                    "EndPointOrigin": None,
                    "Width": 0.0,
                    "Height": 0.0,
                    "WidthOrigin": 0.0,
                    "HeightOrigin": 0.0
                }
                predicted_lines.append(line_data)

        # Create the final JSON structure for the current view
        dim_client_output = {
            "ViewName": batch.get('view_name', 'unknown'),
            "Lines": predicted_lines
        }

        # Save JSON results if path provided
        if json_path:
            # In a real scenario, you might append to an existing file if multiple views exist for a drawing_id
            # For this script, we save one file per view.
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump([dim_client_output], f, indent=2, ensure_ascii=False) # Save as a list with one item
            print(f"💾 Saved prediction results to {json_path}")

        # Print detailed results
        print(f"\n📊 Sample Analysis:")
        print(f"   Nodes: {len(node_pos_np)}")
        print(f"   True dimension lines: {np.sum(positive_mask)}")
        print(f"   Predicted dimension lines: {np.sum(predicted_mask)}")
        print(f"   True Positives: {true_positives}")
        print(f"   False Positives: {false_positives}")
        print(f"   F1 Score: {f1:.4f}")

        # Create prediction results dictionary
        prediction_results = {
            'nodes': len(node_pos_np),
            'true_dimension_lines': int(np.sum(positive_mask)),
            'predicted_dimension_lines': int(np.sum(predicted_mask)),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'f1_score': float(f1),
            'dim_client_output': dim_client_output
        }

        return prediction_results

def test_multiple_samples(num_samples=5):
    """Test model on multiple samples and save visualizations."""
    print("🔄 Loading model...")
    model, device = load_best_model()
    
    if model is None:
        return
    
    print("📂 Loading test data...")
    # Load data
    with open('data/prepared_data_v3.json', 'r') as f:
        full_data = json.load(f)
    
    # Use last 20% as test data
    test_data = full_data[-len(full_data)//5:]
    print(f"📊 Test dataset: {len(test_data)} samples")
    
    # Create test dataset
    negative_sampler = SmartNegativeSampler(strategy='mixed')
    test_dataset = AdvancedDimensionDataset(test_data, negative_sampler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Create output directory
    os.makedirs('test_results', exist_ok=True)
    
    print(f"🎯 Testing on {num_samples} samples...")
    
    sample_count = 0
    for batch in test_loader:
        if batch is None:
            continue

        if sample_count >= num_samples:
            break

        print(f"\n🔍 Analyzing Sample {sample_count + 1}/{num_samples}")
        base_path = f'test_results/sample_{sample_count + 1}'
        save_path = f'{base_path}_prediction.png'
        json_path = f'{base_path}_prediction.json'

        visualize_single_sample(model, batch, device, save_path, json_path)
        sample_count += 1
    
    print(f"\n✅ Testing complete! Check 'test_results/' folder for visualizations.")

if __name__ == '__main__':
    print("🚀 Model Visualization Test")
    print("=" * 50)
    
    # Test on multiple samples
    test_multiple_samples(num_samples=5)
    
    print("\n🎉 All done! Check the generated images to see how well your model performs.")
