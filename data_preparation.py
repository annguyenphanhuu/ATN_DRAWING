import os
import json
import numpy as np
from collections import Counter, defaultdict
from scipy.spatial.distance import pdist, squareform

def calculate_geometric_features(points, lines_data):
    """
    Calculate rich geometric features for each point based on its context.
    """
    points_array = np.array(points)
    n_points = len(points)

    # Distance matrix
    dist_matrix = squareform(pdist(points_array))

    # For each point, calculate features
    features = []

    for i, point in enumerate(points):
        # Basic position features
        x, y = point

        # Distance-based features
        distances = dist_matrix[i]
        min_dist = np.min(distances[distances > 0]) if np.sum(distances > 0) > 0 else 0
        mean_dist = np.mean(distances[distances > 0]) if np.sum(distances > 0) > 0 else 0

        # Nearest neighbors
        nearest_indices = np.argsort(distances)[1:6]  # 5 nearest neighbors (excluding self)

        # Angular distribution of neighbors
        angles = []
        for j in nearest_indices:
            if j < len(points):
                dx = points_array[j][0] - x
                dy = points_array[j][1] - y
                angle = np.arctan2(dy, dx)
                angles.append(angle)

        # Angular variance (measure of how spread out neighbors are)
        angular_variance = np.var(angles) if angles else 0

        # Density features
        radius_01 = 0.1 * np.max(dist_matrix)  # 10% of max distance
        radius_05 = 0.5 * np.max(dist_matrix)  # 50% of max distance

        density_01 = np.sum(distances < radius_01) - 1  # exclude self
        density_05 = np.sum(distances < radius_05) - 1

        features.append([
            x, y,                    # Position
            min_dist, mean_dist,     # Distance features
            angular_variance,        # Angular distribution
            density_01, density_05,  # Local density
        ])

    return features

def extract_line_relationships(lines_data, point_to_index):
    """
    Extract relationships between lines and points for graph construction.
    """
    # Build line connectivity graph
    line_connections = defaultdict(list)  # point -> list of connected points
    line_properties = []  # properties of each line

    for line in lines_data:
        start = line.get("StartPointOrigin")
        end = line.get("EndPointOrigin")

        if start and end:
            start_point = tuple(start.values())
            end_point = tuple(end.values())

            if start_point in point_to_index and end_point in point_to_index:
                start_idx = point_to_index[start_point]
                end_idx = point_to_index[end_point]

                # Add bidirectional connection
                line_connections[start_idx].append(end_idx)
                line_connections[end_idx].append(start_idx)

                # Calculate line properties
                width = line.get("WidthOrigin", 0)
                height = line.get("HeightOrigin", 0)
                length = np.sqrt(width**2 + height**2)

                # Line orientation
                angle = np.arctan2(height, width)

                line_properties.append({
                    'start': start_idx,
                    'end': end_idx,
                    'length': length,
                    'angle': angle,
                    'width': width,
                    'height': height
                })

    return line_connections, line_properties

def build_geometric_graph(points, line_connections, line_properties):
    """
    Build a graph that captures geometric relationships in technical drawings.
    """
    n_points = len(points)
    edge_index = []
    edge_features = []

    # 1. Direct line connections (strongest relationships)
    for start_idx, connected_points in line_connections.items():
        for end_idx in connected_points:
            if start_idx < end_idx:  # avoid duplicates
                edge_index.append([start_idx, end_idx])

                # Find line properties for this connection
                line_props = None
                for prop in line_properties:
                    if (prop['start'] == start_idx and prop['end'] == end_idx) or \
                       (prop['start'] == end_idx and prop['end'] == start_idx):
                        line_props = prop
                        break

                if line_props:
                    edge_features.append([
                        1.0,  # connection type: direct line
                        line_props['length'],
                        line_props['angle'],
                        1.0   # connection strength
                    ])
                else:
                    edge_features.append([1.0, 0.0, 0.0, 1.0])

    # 2. Geometric proximity connections (weaker relationships)
    points_array = np.array(points)
    dist_matrix = squareform(pdist(points_array))

    # Connect points that are close but not directly connected
    threshold = np.percentile(dist_matrix[dist_matrix > 0], 20)  # 20th percentile

    for i in range(n_points):
        for j in range(i + 1, n_points):
            # Skip if already connected by line
            if j in line_connections.get(i, []):
                continue

            distance = dist_matrix[i, j]
            if distance < threshold:
                edge_index.append([i, j])

                # Calculate angle between points
                dx = points_array[j][0] - points_array[i][0]
                dy = points_array[j][1] - points_array[i][1]
                angle = np.arctan2(dy, dx)

                edge_features.append([
                    0.0,  # connection type: proximity
                    distance,
                    angle,
                    0.5   # connection strength
                ])

    return edge_index, edge_features

def process_data(root_dir):
    """
    Processes the raw JSON data with advanced geometric feature engineering.
    """
    processed_data = []

    for subdir, _, files in os.walk(root_dir):
        dim_all_path = None
        dim_client_path = None
        drawing_id = os.path.basename(subdir)

        for file in files:
            if file.endswith("_DimAll.json"):
                dim_all_path = os.path.join(subdir, file)
            elif file.endswith("_DimClient.json"):
                dim_client_path = os.path.join(subdir, file)

        if dim_all_path and dim_client_path:
            try:
                with open(dim_all_path, 'r') as f:
                    dim_all_data = json.load(f)
                with open(dim_client_path, 'r') as f:
                    dim_client_data = json.load(f)

                for view_all, view_client in zip(dim_all_data, dim_client_data):
                    if view_all.get("ViewName") != view_client.get("ViewName"):
                        print(f"Warning: View names do not match in {drawing_id}. Skipping view.")
                        continue

                    lines = view_all.get("Lines", [])
                    if not lines:
                        continue

                    # Extract all unique points
                    all_points = set()
                    for line in lines:
                        start = line.get("StartPointOrigin")
                        end = line.get("EndPointOrigin")
                        if start:
                            all_points.add(tuple(start.values()))
                        if end:
                            all_points.add(tuple(end.values()))

                    if len(all_points) < 2:
                        continue

                    point_cloud_coords = sorted(list(all_points))
                    point_to_index = {point: i for i, point in enumerate(point_cloud_coords)}

                    # Normalize coordinates
                    coords_array = np.array(point_cloud_coords, dtype=np.float32)
                    min_coords = np.min(coords_array, axis=0)
                    max_coords = np.max(coords_array, axis=0)
                    range_coords = max_coords - min_coords
                    range_coords[range_coords == 0] = 1
                    normalized_coords = (coords_array - min_coords) / range_coords
                    normalized_points = [tuple(coord) for coord in normalized_coords]

                    # Calculate rich geometric features
                    geometric_features = calculate_geometric_features(normalized_points, lines)

                    # Extract line relationships
                    line_connections, line_properties = extract_line_relationships(lines, point_to_index)

                    # Build geometric graph
                    edge_index, edge_features = build_geometric_graph(normalized_points, line_connections, line_properties)

                    # Process target dimension lines
                    dimension_lines = []
                    for line in view_client.get("Lines", []):
                        start = tuple(line["StartPointOrigin"].values())
                        end = tuple(line["EndPointOrigin"].values())
                        if start in point_to_index and end in point_to_index:
                            dimension_lines.append(sorted((point_to_index[start], point_to_index[end])))

                    # Convert numpy types to Python types for JSON serialization
                    def convert_to_python_types(obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, (np.float32, np.float64)):
                            return float(obj)
                        elif isinstance(obj, (np.int32, np.int64)):
                            return int(obj)
                        elif isinstance(obj, list):
                            return [convert_to_python_types(item) for item in obj]
                        elif isinstance(obj, dict):
                            return {key: convert_to_python_types(value) for key, value in obj.items()}
                        else:
                            return obj

                    processed_data.append({
                        "drawing_id": drawing_id,
                        "view_name": view_all.get("ViewName"),
                        "node_features": convert_to_python_types(geometric_features),
                        "edge_index": convert_to_python_types(edge_index),
                        "edge_features": convert_to_python_types(edge_features),
                        "dimension_lines": dimension_lines,
                        "num_nodes": len(point_cloud_coords)
                    })

            except json.JSONDecodeError:
                print(f"Error decoding JSON in {subdir}. Skipping.")
            except Exception as e:
                print(f"An error occurred while processing {subdir}: {e}")

    return processed_data

if __name__ == "__main__":
    output_dir = 'data/output'
    print("Processing data with advanced geometric features...")
    prepared_data = process_data(output_dir)

    # Save the processed data to a new file
    save_path = 'data/prepared_data_v3.json'
    with open(save_path, 'w') as f:
        json.dump(prepared_data, f, indent=2)

    print(f"Processed {len(prepared_data)} views and saved to {save_path}")

    # Print some statistics
    if prepared_data:
        total_nodes = sum(sample['num_nodes'] for sample in prepared_data)
        total_edges = sum(len(sample['edge_index']) for sample in prepared_data)
        total_dim_lines = sum(len(sample['dimension_lines']) for sample in prepared_data)

        print(f"Statistics:")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Total graph edges: {total_edges}")
        print(f"  Total dimension lines: {total_dim_lines}")
        print(f"  Average nodes per view: {total_nodes / len(prepared_data):.2f}")
        print(f"  Average dimension lines per view: {total_dim_lines / len(prepared_data):.2f}")

        # Sample feature dimensions
        sample = prepared_data[0]
        print(f"  Node feature dimensions: {len(sample['node_features'][0])}")
        if sample['edge_features']:
            print(f"  Edge feature dimensions: {len(sample['edge_features'][0])}")
        else:
            print(f"  Edge feature dimensions: 0 (no edges in first sample)")

    # Print some statistics
    if prepared_data:
        total_nodes = sum(sample['num_nodes'] for sample in prepared_data)
        total_edges = sum(len(sample['edge_index']) for sample in prepared_data)
        total_dim_lines = sum(len(sample['dimension_lines']) for sample in prepared_data)

        print(f"Statistics:")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Total graph edges: {total_edges}")
        print(f"  Total dimension lines: {total_dim_lines}")
        print(f"  Average nodes per view: {total_nodes / len(prepared_data):.2f}")
        print(f"  Average dimension lines per view: {total_dim_lines / len(prepared_data):.2f}")

        # Sample feature dimensions
        sample = prepared_data[0]
        print(f"  Node feature dimensions: {len(sample['node_features'][0])}")
        if sample['edge_features']:
            print(f"  Edge feature dimensions: {len(sample['edge_features'][0])}")
        else:
            print(f"  Edge feature dimensions: 0 (no edges in first sample)")

