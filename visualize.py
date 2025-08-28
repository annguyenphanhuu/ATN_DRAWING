import os
import json
import matplotlib.pyplot as plt

import re

def sanitize_filename(name):
    """Remove invalid characters from a string to make it a valid filename."""
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def plot_view_data(ax, view_data, color):
    """Plots all lines from a single view onto the given axes."""
    if not view_data or not view_data.get('Lines'):
        ax.text(0.5, 0.5, 'No data available for this view.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return

    for line in view_data['Lines']:
        start_point = line.get('StartPoint')
        end_point = line.get('EndPoint')

        if start_point and end_point:
            ax.plot([start_point['X'], end_point['X']],
                    [start_point['Y'], end_point['Y']],
                    color=color, linewidth=0.5)

def create_visualization_for_view(case_id, view_all, view_client, output_path):
    """Creates and saves a side-by-side visualization for a single view."""
    view_name = view_all.get('ViewName') if view_all else view_client.get('ViewName', 'Unknown View')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f'Visualization for Case: {case_id} - View: {view_name}')

    # Plot DimAll data
    ax1.set_title('DimAll Data')
    plot_view_data(ax1, view_all, 'blue')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True)

    # Plot DimClient data
    ax2.set_title('DimClient Data')
    plot_view_data(ax2, view_client, 'red')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close(fig)

def main():
    """Main function to process all cases and their views."""
    output_dir = os.path.join('data', 'output')
    visualization_dir = os.path.join('data', 'visualizations')

    if not os.path.exists(output_dir):
        print(f"Error: Input directory '{output_dir}' not found.")
        return

    case_ids = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    for case_id in case_ids:
        case_visualization_dir = os.path.join(visualization_dir, case_id)
        os.makedirs(case_visualization_dir, exist_ok=True)

        dim_all_path = os.path.join(output_dir, case_id, f'{case_id}_DimAll.json')
        dim_client_path = os.path.join(output_dir, case_id, f'{case_id}_DimClient.json')

        if os.path.exists(dim_all_path) and os.path.exists(dim_client_path):
            print(f'Processing case: {case_id}...')
            with open(dim_all_path, 'r') as f:
                data_all = json.load(f)

            with open(dim_client_path, 'r') as f:
                data_client = json.load(f)

            views_all = {view['ViewName']: view for view in data_all}
            views_client = {view['ViewName']: view for view in data_client}

            all_view_names = set(views_all.keys()) | set(views_client.keys())

            for view_name in all_view_names:
                view_all_data = views_all.get(view_name)
                view_client_data = views_client.get(view_name)

                sanitized_view_name = sanitize_filename(view_name)
                output_image_path = os.path.join(case_visualization_dir, f'{sanitized_view_name}_comparison.png')

                print(f'  - Visualizing view: {view_name}')
                create_visualization_for_view(case_id, view_all_data, view_client_data, output_image_path)
        else:
            print(f'Skipping case {case_id}: JSON files not found.')

    print('Visualization generation complete.')

if __name__ == '__main__':
    main()

