import json
import numpy as np
from prediction_converter import PredictionConverter

def test_denormalization():
    """Test the denormalization process to ensure it works correctly."""
    
    # Load the converter
    converter = PredictionConverter()
    
    # Get sample mapping for the first sample
    if 0 in converter.sample_mapping:
        sample_info = converter.sample_mapping[0]
        drawing_id = sample_info['drawing_id']
        view_name = sample_info['view_name']
        
        print(f"Testing denormalization for:")
        print(f"  Drawing ID: {drawing_id}")
        print(f"  View Name: {view_name}")
        
        # Load original coordinates
        norm_params = converter.load_original_coordinates(drawing_id, view_name)
        
        if norm_params:
            print(f"\nOriginal coordinate range:")
            print(f"  Min coords: {norm_params['min_coords']}")
            print(f"  Max coords: {norm_params['max_coords']}")
            print(f"  Range: {norm_params['range_coords']}")
            
            # Test with some sample normalized coordinates
            test_normalized = [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)]
            denormalized = converter.denormalize_coordinates(test_normalized, norm_params)
            
            print(f"\nTest denormalization:")
            for i, (norm, denorm) in enumerate(zip(test_normalized, denormalized)):
                print(f"  {norm} -> {denorm}")
            
            # Load and compare with actual prediction
            with open('test_results/sample_1_prediction.json', 'r') as f:
                prediction = json.load(f)[0]
            
            print(f"\nFirst predicted line coordinates:")
            first_line = prediction['Lines'][0]
            start = (first_line['StartPoint']['X'], first_line['StartPoint']['Y'])
            end = (first_line['EndPoint']['X'], first_line['EndPoint']['Y'])
            print(f"  Start (normalized): {start}")
            print(f"  End (normalized): {end}")
            
            # Denormalize
            denorm_coords = converter.denormalize_coordinates([start, end], norm_params)
            print(f"  Start (denormalized): {denorm_coords[0]}")
            print(f"  End (denormalized): {denorm_coords[1]}")
            
            # Compare with original data
            original_file = f'data/output/{drawing_id}/{drawing_id}_DimClient.json'
            try:
                with open(original_file, 'r') as f:
                    original_data = json.load(f)
                
                # Find matching view
                original_view = None
                for view in original_data:
                    if view.get('ViewName') == view_name:
                        original_view = view
                        break
                
                if original_view and original_view.get('Lines'):
                    print(f"\nOriginal first line coordinates:")
                    orig_line = original_view['Lines'][0]
                    orig_start = (orig_line['StartPointOrigin']['X'], orig_line['StartPointOrigin']['Y'])
                    orig_end = (orig_line['EndPointOrigin']['X'], orig_line['EndPointOrigin']['Y'])
                    print(f"  Start (original): {orig_start}")
                    print(f"  End (original): {orig_end}")
                    
            except Exception as e:
                print(f"Could not load original data: {e}")
        
        else:
            print("Could not load normalization parameters")
    else:
        print("No sample mapping found")

if __name__ == '__main__':
    test_denormalization()
