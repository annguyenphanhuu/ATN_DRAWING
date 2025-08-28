import json
import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from model import build_model, SmartNegativeSampler
from train import AdvancedDimensionDataset, collate_fn, evaluate_model
from prediction_converter import PredictionConverter

def create_evaluation_report(model, data_loader, device):
    """Performs a full evaluation and generates reports and plots."""
    print("\n→ Performing detailed analysis...")
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            if batch is None: continue
            
            # Move data to device and get predictions
            logits = model(
                batch['node_features'].to(device),
                batch['edge_index'].to(device),
                batch['edge_candidates'].to(device),
                batch['edge_features'].to(device),
                batch['node_positions'].to(device)
            )
            probabilities = torch.sigmoid(logits).cpu().numpy()
            
            all_predictions.extend((probabilities > 0.5).astype(int))
            all_labels.extend(batch['labels'].cpu().numpy())

    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['No Dimension', 'Dimension Line']))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Dimension', 'Dimension Line'],
                yticklabels=['No Dimension', 'Dimension Line'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('evaluation_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n🖼 Saved confusion matrix to 'evaluation_confusion_matrix.png'")

def visualize_and_save_predictions(model, data_loader, device, converter, num_samples=5, output_dir='evaluation_results'):
    """Visualize predictions, save plots, and save JSON results in original format."""
    print(f"\n→ Generating visualizations for {num_samples} samples...")
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    samples_processed = 0
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if samples_processed >= num_samples: break
            if batch is None: continue

            # Get model predictions
            logits = model(
                batch['node_features'].to(device),
                batch['edge_index'].to(device),
                batch['edge_candidates'].to(device),
                batch['edge_features'].to(device),
                batch['node_positions'].to(device)
            )
            predictions = torch.sigmoid(logits).cpu().numpy()
            
            # Prepare data for visualization and conversion
            node_pos_np = batch['node_positions'].cpu().numpy()
            edge_candidates_np = batch['edge_candidates'].cpu().numpy()
            labels_np = batch['labels'].cpu().numpy()
            # Safely extract drawing_id and view_name from the batch
            drawing_id = batch['drawing_id'][0] if batch.get('drawing_id') else f"unknown_id_{i}"
            view_name = batch['view_name'][0] if batch.get('view_name') else f"unknown_view_{i}"

            # --- Create and Save Visualization ---
            plt.figure(figsize=(10, 10))
            plt.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='lightblue', s=25, zorder=1)

            # Plot ground truth (True Positives and False Negatives)
            true_positives_mask = (predictions > 0.5) & (labels_np == 1)
            false_negatives_mask = (predictions <= 0.5) & (labels_np == 1)
            for j, (u, v) in enumerate(edge_candidates_np.T):
                if true_positives_mask[j]:
                    plt.plot([node_pos_np[u, 0], node_pos_np[v, 0]], [node_pos_np[u, 1], node_pos_np[v, 1]], 'g-', linewidth=2, alpha=0.8, label='Correct Prediction (TP)')
                elif false_negatives_mask[j]:
                    plt.plot([node_pos_np[u, 0], node_pos_np[v, 0]], [node_pos_np[u, 1], node_pos_np[v, 1]], 'm--', linewidth=1.5, alpha=0.7, label='Missed Prediction (FN)')

            # Plot model predictions (False Positives)
            false_positives_mask = (predictions > 0.5) & (labels_np == 0)
            for j, (u, v) in enumerate(edge_candidates_np.T):
                if false_positives_mask[j]:
                    plt.plot([node_pos_np[u, 0], node_pos_np[v, 0]], [node_pos_np[u, 1], node_pos_np[v, 1]], 'r-', linewidth=1.5, alpha=0.8, label='Wrong Prediction (FP)')

            plt.title(f'Prediction for {drawing_id} - {view_name}')
            plt.xlabel('X Coordinate (Normalized)')
            plt.ylabel('Y Coordinate (Normalized)')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.grid(True, linestyle='--', alpha=0.5)
            
            safe_view_name = view_name.replace(' ', '_')
            vis_path = os.path.join(output_dir, f'{drawing_id}_{safe_view_name}_visualization.png')
            plt.savefig(vis_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  🖼 Saved visualization to {vis_path}")

            # --- Save Predictions to JSON in Original Format ---
            predicted_indices = np.where(predictions > 0.5)[0]
            predicted_lines_data = []
            for idx in predicted_indices:
                u, v = edge_candidates_np[:, idx]
                start_node_pos = node_pos_np[u]
                end_node_pos = node_pos_np[v]
                line_data = {
                    "StartPoint": {"X": float(start_node_pos[0]), "Y": float(start_node_pos[1])},
                    "EndPoint": {"X": float(end_node_pos[0]), "Y": float(end_node_pos[1])}
                }
                predicted_lines_data.append(line_data)

            prediction_for_conversion = [{
                "ViewName": view_name,
                "Lines": predicted_lines_data
            }]
            
            # Use converter to get original format directly from memory
            json_output_path = os.path.join(output_dir, f'{drawing_id}_{safe_view_name}_predicted.json')
            converter.convert_prediction_to_original_format(
                prediction_for_conversion, # Pass data directly
                output_path=json_output_path,
                drawing_id=drawing_id,
                view_name=view_name
            )
            print(f"  💾 Saved JSON prediction to {json_output_path}")
            samples_processed += 1

def main():
    """Main function to run the complete evaluation process."""
    parser = argparse.ArgumentParser(description='Run a full evaluation of the model.')
    parser.add_argument('--model_path', type=str, default='best_model_v3.pth', help='Path to the trained model checkpoint.')
    parser.add_argument('--data_path', type=str, default='data/prepared_data_v3.json', help='Path to the prepared data.')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation outputs.')
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Data ---
    print("\n→ Loading test data...")
    with open(args.data_path, 'r') as f:
        full_data = json.load(f)
    
    test_split = max(1, len(full_data) // 5) # Ensure at least 1 sample
    test_data = full_data[-test_split:]
    print(f"Evaluating on {len(test_data)} samples.")

    sampler = SmartNegativeSampler(strategy='mixed')
    test_dataset = AdvancedDimensionDataset(test_data, sampler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # --- Load Model ---
    print("\n→ Loading trained model...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=device)
    model = build_model(7, 4, 256, 128) # Hardcoded dims, should match training
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded model from epoch {checkpoint['epoch']} with Val F1: {checkpoint.get('val_f1', 0):.4f}")

    # --- Run Quantitative Evaluation ---
    print("\n→ Running quantitative evaluation...")
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss, test_precision, test_recall, test_f1, test_auc = evaluate_model(model, test_loader, criterion, device)
    print("\n📊 Test Set Performance:")
    print(f"  ▪ F1 Score:  {test_f1:.4f}")
    print(f"  ▪ Precision: {test_precision:.4f}")
    print(f"  ▪ Recall:    {test_recall:.4f}")
    print(f"  ▪ AUC:       {test_auc:.4f}")
    print(f"  ▪ Loss:      {test_loss:.4f}")

    # --- Generate Detailed Reports and Visuals ---
    create_evaluation_report(model, test_loader, device)
    
    # --- Initialize Converter and Run Visualization/JSON Export ---
    converter = PredictionConverter(prepared_data_path=args.data_path)
    visualize_and_save_predictions(model, test_loader, device, converter, num_samples=args.num_samples, output_dir=args.output_dir)

    print(f"\n\n✨ Evaluation complete! All results are in the '{args.output_dir}' folder. ✨")

if __name__ == '__main__':
    main()

