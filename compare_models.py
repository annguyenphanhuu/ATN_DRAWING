import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import build_model, SmartNegativeSampler
from train import AdvancedDimensionDataset, collate_fn
import os

def load_model_if_exists(model_path, model_name):
    """Load model if it exists."""
    NODE_FEATURE_DIM = 7
    EDGE_FEATURE_DIM = 4
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(model_path):
        model = build_model(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, HIDDEN_DIM, EMBEDDING_DIM)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✅ {model_name} loaded successfully!")
        print(f"   Epochs: {checkpoint['epoch']}")
        print(f"   F1 Score: {checkpoint['val_f1']:.4f}")
        print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
        
        return model, checkpoint
    else:
        print(f"❌ {model_name} not found at {model_path}")
        return None, None

def evaluate_model_performance(model, test_loader, device, model_name):
    """Evaluate model performance on test data."""
    if model is None:
        return None
    
    print(f"\n🔄 Evaluating {model_name}...")
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in test_loader:
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
    
    # Calculate metrics
    accuracy = np.mean(all_predictions == all_labels)
    
    true_positives = np.sum((all_predictions == 1) & (all_labels == 1))
    false_positives = np.sum((all_predictions == 1) & (all_labels == 0))
    false_negatives = np.sum((all_predictions == 0) & (all_labels == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # AUC calculation
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_confidences)
    except:
        auc = 0.0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences
    }
    
    print(f"📊 {model_name} Results:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   AUC:       {auc:.4f}")
    
    return results

def compare_models():
    """Compare old and new models."""
    print("🔄 Model Comparison Tool")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("\n📂 Loading models...")
    
    # Try to load both old and new models
    new_model, new_checkpoint = load_model_if_exists('best_model_v3.pth', 'New Model (v3)')
    old_model, old_checkpoint = load_model_if_exists('best_model.pth', 'Old Model (v1)')
    
    if new_model is None and old_model is None:
        print("❌ No models found! Please train models first.")
        return
    
    # Load test data
    print("\n📊 Loading test data...")
    with open('data/prepared_data_v3.json', 'r') as f:
        full_data = json.load(f)
    
    # Use last 20% as test data
    test_data = full_data[-len(full_data)//5:]
    print(f"Test dataset: {len(test_data)} samples")
    
    # Create test dataset
    negative_sampler = SmartNegativeSampler(strategy='mixed')
    test_dataset = AdvancedDimensionDataset(test_data, negative_sampler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Evaluate models
    results = {}
    
    if new_model is not None:
        results['new'] = evaluate_model_performance(new_model, test_loader, device, 'New Model (v3)')
    
    if old_model is not None:
        results['old'] = evaluate_model_performance(old_model, test_loader, device, 'Old Model (v1)')
    
    # Create comparison visualization
    if len(results) > 1:
        print("\n📈 Creating comparison visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Metrics comparison
        ax1 = axes[0, 0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        
        if 'new' in results and 'old' in results:
            new_values = [results['new'][m.lower().replace(' ', '_')] for m in metrics]
            old_values = [results['old'][m.lower().replace(' ', '_')] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, old_values, width, label='Old Model', alpha=0.8, color='lightcoral')
            ax1.bar(x + width/2, new_values, width, label='New Model', alpha=0.8, color='lightgreen')
            
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Score')
            ax1.set_title('Model Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (old_val, new_val) in enumerate(zip(old_values, new_values)):
                ax1.text(i - width/2, old_val + 0.01, f'{old_val:.3f}', ha='center', va='bottom', fontsize=8)
                ax1.text(i + width/2, new_val + 0.01, f'{new_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Confidence distribution comparison
        ax2 = axes[0, 1]
        if 'new' in results:
            new_conf = results['new']['confidences']
            new_labels = results['new']['labels']
            ax2.hist(new_conf[new_labels == 0], bins=30, alpha=0.5, label='New: No Dimension', color='red')
            ax2.hist(new_conf[new_labels == 1], bins=30, alpha=0.5, label='New: Dimension', color='green')
        
        if 'old' in results:
            old_conf = results['old']['confidences']
            old_labels = results['old']['labels']
            ax2.hist(old_conf[old_labels == 0], bins=30, alpha=0.3, label='Old: No Dimension', color='darkred', linestyle='--')
            ax2.hist(old_conf[old_labels == 1], bins=30, alpha=0.3, label='Old: Dimension', color='darkgreen', linestyle='--')
        
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Improvement visualization
        ax3 = axes[1, 0]
        if 'new' in results and 'old' in results:
            improvements = []
            metric_names = []
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                improvement = results['new'][metric] - results['old'][metric]
                improvements.append(improvement)
                metric_names.append(metric.replace('_', ' ').title())
            
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            bars = ax3.bar(metric_names, improvements, color=colors, alpha=0.7)
            
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_xlabel('Metrics')
            ax3.set_ylabel('Improvement (New - Old)')
            ax3.set_title('Performance Improvement')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
                        f'{imp:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # Summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "📊 COMPARISON SUMMARY\n\n"
        
        if 'new' in results and 'old' in results:
            new_f1 = results['new']['f1']
            old_f1 = results['old']['f1']
            improvement = new_f1 - old_f1
            
            summary_text += f"🎯 F1 Score Improvement: {improvement:+.4f}\n"
            summary_text += f"   Old Model: {old_f1:.4f}\n"
            summary_text += f"   New Model: {new_f1:.4f}\n\n"
            
            if improvement > 0:
                summary_text += "✅ NEW MODEL IS BETTER!\n\n"
            else:
                summary_text += "❌ Old model performs better\n\n"
            
            summary_text += "Key Improvements:\n"
            for metric in ['accuracy', 'precision', 'recall', 'auc']:
                imp = results['new'][metric] - results['old'][metric]
                summary_text += f"• {metric.title()}: {imp:+.4f}\n"
        
        elif 'new' in results:
            summary_text += "✅ NEW MODEL AVAILABLE\n\n"
            summary_text += f"F1 Score: {results['new']['f1']:.4f}\n"
            summary_text += f"Accuracy: {results['new']['accuracy']:.4f}\n"
            summary_text += f"AUC: {results['new']['auc']:.4f}\n"
        
        elif 'old' in results:
            summary_text += "📊 OLD MODEL ONLY\n\n"
            summary_text += f"F1 Score: {results['old']['f1']:.4f}\n"
            summary_text += f"Accuracy: {results['old']['accuracy']:.4f}\n"
            summary_text += f"AUC: {results['old']['auc']:.4f}\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save comparison
        os.makedirs('test_results', exist_ok=True)
        plt.savefig('test_results/model_comparison.png', dpi=300, bbox_inches='tight')
        print("💾 Comparison saved to: test_results/model_comparison.png")
        
        plt.show()
    
    print("\n✅ Model comparison complete!")

if __name__ == '__main__':
    compare_models()
