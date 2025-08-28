import json
import numpy as np
import os
from typing import Dict, List, Tuple, Optional

class PredictionConverter:
    """
    Converts model predictions back to original data format.
    Reverses the preprocessing steps applied during data preparation.
    """
    
    def __init__(self, prepared_data_path: str = 'data/prepared_data_v3.json'):
        """
        Initialize converter with prepared data to get normalization parameters.

        Args:
            prepared_data_path: Path to the prepared data file
        """
        self.prepared_data_path = prepared_data_path
        self.normalization_params = {}
        self.sample_mapping = {}  # Maps sample indices to drawing_id and view_name
        self._load_normalization_params()
        self._create_sample_mapping()
    
    def _load_normalization_params(self):
        """Load normalization parameters from prepared data."""
        try:
            with open(self.prepared_data_path, 'r') as f:
                prepared_data = json.load(f)
            
            # Store normalization parameters for each drawing/view
            for sample in prepared_data:
                key = f"{sample['drawing_id']}_{sample['view_name']}"
                # We need to reconstruct the original coordinates from the prepared data
                # This requires access to the original data or storing normalization params
                self.normalization_params[key] = {
                    'drawing_id': sample['drawing_id'],
                    'view_name': sample['view_name'],
                    'num_nodes': sample['num_nodes']
                }
                
        except Exception as e:
            print(f"Warning: Could not load normalization parameters: {e}")

    def _create_sample_mapping(self):
        """Create mapping between test sample indices and actual drawing data."""
        try:
            with open(self.prepared_data_path, 'r') as f:
                prepared_data = json.load(f)

            # Create mapping for test samples (assuming test uses last samples)
            # This is a simplified approach - in practice you'd want to store this mapping
            # during the train/test split process
            for i, sample in enumerate(prepared_data):
                self.sample_mapping[i] = {
                    'drawing_id': sample['drawing_id'],
                    'view_name': sample['view_name']
                }

        except Exception as e:
            print(f"Warning: Could not create sample mapping: {e}")
    
    def load_original_coordinates(self, drawing_id: str, view_name: str) -> Optional[Dict]:
        """
        Load original coordinates from the source data files.
        
        Args:
            drawing_id: ID of the drawing
            view_name: Name of the view
            
        Returns:
            Dictionary containing original coordinates and normalization parameters
        """
        # Look for the original data file
        data_dir = 'data/output'
        drawing_path = os.path.join(data_dir, drawing_id)
        
        if not os.path.exists(drawing_path):
            print(f"Warning: Drawing path {drawing_path} not found")
            return None
        
        # Find DimAll file for this drawing
        dim_all_file = None
        for file in os.listdir(drawing_path):
            if file.endswith('_DimAll.json'):
                dim_all_file = os.path.join(drawing_path, file)
                break
        
        if not dim_all_file:
            print(f"Warning: DimAll file not found for {drawing_id}")
            return None
        
        try:
            with open(dim_all_file, 'r') as f:
                dim_all_data = json.load(f)
            
            # Find the matching view
            target_view = None
            for view in dim_all_data:
                if view.get('ViewName') == view_name:
                    target_view = view
                    break
            
            if not target_view:
                print(f"Warning: View {view_name} not found in {drawing_id}")
                return None
            
            # Extract original points and calculate normalization parameters
            lines = target_view.get('Lines', [])
            all_points = set()
            
            for line in lines:
                start = line.get("StartPointOrigin")
                end = line.get("EndPointOrigin")
                if start:
                    all_points.add(tuple(start.values()))
                if end:
                    all_points.add(tuple(end.values()))
            
            if len(all_points) < 2:
                return None
            
            # Sort points to match the order used in data preparation
            point_cloud_coords = sorted(list(all_points))
            coords_array = np.array(point_cloud_coords, dtype=np.float32)
            
            # Calculate normalization parameters
            min_coords = np.min(coords_array, axis=0)
            max_coords = np.max(coords_array, axis=0)
            range_coords = max_coords - min_coords
            range_coords[range_coords == 0] = 1
            
            return {
                'original_coords': point_cloud_coords,
                'min_coords': min_coords,
                'max_coords': max_coords,
                'range_coords': range_coords,
                'point_to_index': {point: i for i, point in enumerate(point_cloud_coords)}
            }
            
        except Exception as e:
            print(f"Error loading original coordinates: {e}")
            return None
    
    def denormalize_coordinates(self, normalized_coords: List[Tuple[float, float]], 
                              normalization_params: Dict) -> List[Tuple[float, float]]:
        """
        Convert normalized coordinates back to original scale.
        
        Args:
            normalized_coords: List of normalized coordinate tuples
            normalization_params: Parameters used for normalization
            
        Returns:
            List of original coordinate tuples
        """
        coords_array = np.array(normalized_coords)
        min_coords = normalization_params['min_coords']
        range_coords = normalization_params['range_coords']
        
        # Reverse normalization: original = normalized * range + min
        original_coords = coords_array * range_coords + min_coords
        
        return [tuple(coord) for coord in original_coords]
    
    def convert_prediction_to_original_format(self, prediction_input: any,
                                            output_path: str = None,
                                            drawing_id: Optional[str] = None,
                                            view_name: Optional[str] = None) -> Dict:
        """
        Convert prediction data back to the original DimClient format.

        Args:
            prediction_input: Prediction data, can be a file path (str) or in-memory data (list/dict).
            output_path: Optional path to save the converted result.
            drawing_id: The ID of the drawing.
            view_name: The name of the view.

        Returns:
            A dictionary in the original DimClient format.
        """
        try:
            if isinstance(prediction_input, str):
                with open(prediction_input, 'r') as f:
                    prediction_data = json.load(f)
            else:
                prediction_data = prediction_input
            
            # Extract the first (and usually only) view from prediction
            if isinstance(prediction_data, list) and len(prediction_data) > 0:
                view_data = prediction_data[0]
            else:
                view_data = prediction_data
            
            predicted_lines = view_data.get('Lines', [])

            # Prioritize passed arguments for drawing_id and view_name
            if not drawing_id or not view_name:
                # Fallback to extracting from path if not provided
                extracted_drawing_id, extracted_view_name = self._extract_drawing_info_from_path(prediction_json_path)
                drawing_id = drawing_id or extracted_drawing_id
                view_name = view_name or extracted_view_name

            # Get view_name from data if still not available
            view_name = view_name or view_data.get('ViewName', 'unknown')

            if not drawing_id:
                print("Warning: Could not determine drawing_id. Using 'unknown'")
                drawing_id = 'unknown'
            
            # Load original coordinates and normalization parameters
            norm_params = self.load_original_coordinates(drawing_id, view_name)
            
            if not norm_params:
                print("Warning: Could not load original coordinates. Returning prediction as-is.")
                return view_data
            
            # Convert predicted lines back to original format
            converted_lines = []
            for line in predicted_lines:
                # The coordinates in prediction are already in normalized space [0,1]
                # We need to map them back to original coordinate space
                start_point = (line['StartPoint']['X'], line['StartPoint']['Y'])
                end_point = (line['EndPoint']['X'], line['EndPoint']['Y'])

                # Denormalize coordinates from [0,1] space back to original space
                denormalized_coords = self.denormalize_coordinates(
                    [start_point, end_point], norm_params
                )

                original_start = denormalized_coords[0]
                original_end = denormalized_coords[1]
                
                # Create line in original format
                converted_line = {
                    "StartPoint": {"X": float(original_start[0]), "Y": float(original_start[1])},
                    "EndPoint": {"X": float(original_end[0]), "Y": float(original_end[1])},
                    "StartPointOrigin": {"X": float(original_start[0]), "Y": float(original_start[1])},
                    "EndPointOrigin": {"X": float(original_end[0]), "Y": float(original_end[1])},
                    "Width": float(original_end[0] - original_start[0]),
                    "Height": float(original_end[1] - original_start[1]),
                    "WidthOrigin": float(original_end[0] - original_start[0]),
                    "HeightOrigin": float(original_end[1] - original_start[1])
                }
                converted_lines.append(converted_line)
            
            # Create final result in original format
            result = {
                "ViewName": view_name,
                "Lines": converted_lines
            }
            
            # Save to file if output path provided
            if output_path:
                # Save as a list to match original DimClient format
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump([result], f, indent=2, ensure_ascii=False)
                print(f"✅ Converted prediction saved to {output_path}")
            
            return result
            
        except Exception as e:
            print(f"Error converting prediction: {e}")
            return None
    
    def _extract_drawing_info_from_path(self, file_path: str) -> tuple[Optional[str], Optional[str]]:
        """
        Try to extract drawing_id and view_name from file path or name.

        Args:
            file_path: Path to the prediction file

        Returns:
            Tuple of (drawing_id, view_name) or (None, None)
        """
        # Try to extract from file name patterns
        filename = os.path.basename(file_path)

        # Common patterns: sample_1_prediction.json, drawing_123_view_1.json, etc.
        if 'sample_' in filename:
            # Extract sample number from filename like "sample_1_prediction.json"
            import re
            match = re.search(r'sample_(\d+)', filename)
            if match:
                sample_idx = int(match.group(1)) - 1  # Convert to 0-based index
                if sample_idx in self.sample_mapping:
                    mapping = self.sample_mapping[sample_idx]
                    return mapping['drawing_id'], mapping['view_name']

        # Add more extraction logic based on your file naming conventions
        return None, None
    
    def batch_convert_predictions(self, predictions_dir: str, output_dir: str):
        """
        Convert all prediction files in a directory.
        
        Args:
            predictions_dir: Directory containing prediction JSON files
            output_dir: Directory to save converted files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        converted_count = 0
        for filename in os.listdir(predictions_dir):
            if filename.endswith('_prediction.json'):
                input_path = os.path.join(predictions_dir, filename)
                output_filename = filename.replace('_prediction.json', '_converted.json')
                output_path = os.path.join(output_dir, output_filename)
                
                result = self.convert_prediction_to_original_format(input_path, output_path)
                if result:
                    converted_count += 1
        
        print(f"✅ Converted {converted_count} prediction files")

def main():
    """Example usage of the converter."""
    converter = PredictionConverter()
    
    # Convert a single prediction file
    prediction_file = 'test_results/sample_1_prediction.json'
    if os.path.exists(prediction_file):
        result = converter.convert_prediction_to_original_format(
            prediction_file, 
            'test_results/sample_1_converted.json'
        )
        if result:
            print("✅ Single file conversion completed")
    
    # Batch convert all predictions in test_results
    converter.batch_convert_predictions('test_results', 'test_results/converted')

if __name__ == '__main__':
    main()
