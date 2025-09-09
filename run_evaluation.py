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


def group_data_by_drawing(data):
    """Groups a list of view data into a dictionary keyed by drawing_id."""
    drawing_groups = {}
    for item in data:
        drawing_id = item['drawing_id']
        if drawing_id not in drawing_groups:
            drawing_groups[drawing_id] = []
        drawing_groups[drawing_id].append(item)
    return drawing_groups

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

def visualize_and_save_drawing_predictions(model, drawing_id, drawing_views, device, converter, output_dir, sampler):
    """Visualize predictions for all views of a single drawing, save plots, and save a consolidated JSON result."""
    print(f"\n→ Processing drawing: {drawing_id} ({len(drawing_views)} views)...")
    model.eval()

    all_views_predictions_for_conversion = []

    with torch.no_grad():
        for view_data in drawing_views:
            # Manually create a batch for this single view
            dataset = AdvancedDimensionDataset([view_data], sampler)
            if not dataset[0]: continue
            batch = collate_fn([dataset[0]])

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
            view_name = batch['view_name'][0]

            # --- Create and Save 3-Panel Visualization for each view ---
            safe_view_name = view_name.replace(' ', '_')
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

            # Panel 1: Original Geometry
            edge_index_np = batch['edge_index'].cpu().numpy()
            ax1.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='lightblue', s=25, zorder=1)
            for u, v in edge_index_np.T:
                ax1.plot([node_pos_np[u, 0], node_pos_np[v, 0]], [node_pos_np[u, 1], node_pos_np[v, 1]], 'k-', linewidth=0.5, alpha=0.6)
            ax1.set_title(f'Original Geometry\n{drawing_id} - {view_name}')
            ax1.set_aspect('equal', adjustable='box')

            # Panel 2: Ground Truth
            ax2.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='lightblue', s=25, zorder=1)
            ground_truth_mask = labels_np == 1
            for j, (u, v) in enumerate(edge_candidates_np.T):
                if ground_truth_mask[j]:
                    ax2.plot([node_pos_np[u, 0], node_pos_np[v, 0]], [node_pos_np[u, 1], node_pos_np[v, 1]], 'b-', linewidth=1.5)
            ax2.set_title('Ground Truth Dimensions')
            ax2.set_aspect('equal', adjustable='box')

            # Panel 3: Model Prediction
            ax3.scatter(node_pos_np[:, 0], node_pos_np[:, 1], c='lightblue', s=25, zorder=1)
            true_positives_mask = (predictions > 0.5) & (labels_np == 1)
            false_positives_mask = (predictions > 0.5) & (labels_np == 0)
            for j, (u, v) in enumerate(edge_candidates_np.T):
                if true_positives_mask[j]:
                    ax3.plot([node_pos_np[u, 0], node_pos_np[v, 0]], [node_pos_np[u, 1], node_pos_np[v, 1]], 'g-', linewidth=1.5, label='Correct')
                elif false_positives_mask[j]:
                    ax3.plot([node_pos_np[u, 0], node_pos_np[v, 0]], [node_pos_np[u, 1], node_pos_np[v, 1]], 'r-', linewidth=1.5, label='Incorrect')
            ax3.set_title('Model Prediction')
            ax3.set_aspect('equal', adjustable='box')

            # Save the figure for the current view
            vis_path = os.path.join(output_dir, f'{drawing_id}_{safe_view_name}_comparison.png')
            plt.savefig(vis_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  🖼 Saved view comparison to {vis_path}")

            # --- Aggregate Predictions for JSON Output ---
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

            all_views_predictions_for_conversion.append({
                "ViewName": view_name,
                "Lines": predicted_lines_data
            })

    # --- Save Consolidated Predictions to a single JSON file for the drawing ---
    if all_views_predictions_for_conversion:
        json_output_path = os.path.join(output_dir, f'{drawing_id}_predicted.json')
        converter.convert_prediction_to_original_format(
            all_views_predictions_for_conversion,
            output_path=json_output_path,
            drawing_id=drawing_id
        )
        print(f"  💾 Saved consolidated JSON for drawing {drawing_id} to {json_output_path}")

def main():
    """Main function to run the complete evaluation process."""
    parser = argparse.ArgumentParser(description='Run a full evaluation of the model.')
    parser.add_argument('--model_path', type=str, default='best_model_v3.pth', help='Path to the trained model checkpoint.')
    parser.add_argument('--data_path', type=str, default='data/prepared_data_v3.json', help='Path to the prepared data.')
    parser.add_argument('--num_drawings', type=int, default=5, help='Number of drawings to visualize.')
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

    # --- Run Visualization and JSON Export for Drawings ---
    os.makedirs(args.output_dir, exist_ok=True)
    test_drawings = group_data_by_drawing(test_data)
    print(f"\n→ Found {len(test_drawings)} drawings to evaluate.")

    converter = PredictionConverter(prepared_data_path=args.data_path)

    drawings_processed = 0
    for drawing_id, drawing_views in test_drawings.items():
        if drawings_processed >= args.num_drawings:
            print(f"\nReached visualization limit of {args.num_drawings} drawings.")
            break

        visualize_and_save_drawing_predictions(
            model,
            drawing_id,
            drawing_views,
            device,
            converter,
            args.output_dir,
            sampler
        )
        drawings_processed += 1

    print(f"\n\n✨ Evaluation complete! All results are in the '{args.output_dir}' folder. ✨")

if __name__ == '__main__':
    main()

