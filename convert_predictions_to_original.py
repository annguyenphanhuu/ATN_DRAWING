#!/usr/bin/env python3
"""
Script để convert dữ liệu prediction về dạng ban đầu (original format).
Reverse lại tất cả các bước preprocessing đã được áp dụng trong quá trình train.
"""

import json
import os
import argparse
from prediction_converter import PredictionConverter

def main():
    """Main function to convert predictions back to original format."""
    
    parser = argparse.ArgumentParser(description='Convert AI predictions back to original format')
    parser.add_argument('--input', '-i', 
                       help='Input prediction file or directory', 
                       default='test_results')
    parser.add_argument('--output', '-o', 
                       help='Output directory for converted files', 
                       default='test_results/original_format')
    parser.add_argument('--prepared_data', '-p',
                       help='Path to prepared data file for normalization parameters',
                       default='data/prepared_data_v3.json')
    
    args = parser.parse_args()
    
    print("🔄 Khởi tạo Prediction Converter...")
    converter = PredictionConverter(args.prepared_data)
    
    # Tạo output directory nếu chưa có
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isfile(args.input):
        # Convert single file
        print(f"📄 Converting single file: {args.input}")

        # Get drawing ID from file path
        drawing_id, view_name = converter._extract_drawing_info_from_path(args.input)

        if drawing_id and drawing_id != 'unknown':
            # Tạo tên file an toàn cho view name
            safe_view_name = view_name.replace(' ', '_').replace('/', '_') if view_name else 'View1'
            output_filename = f"{drawing_id}_{safe_view_name}_DimClient_predicted.json"
        else:
            filename = os.path.basename(args.input)
            output_filename = filename.replace('_prediction.json', '_original_format.json')

        output_path = os.path.join(args.output, output_filename)

        result = converter.convert_prediction_to_original_format(args.input, output_path)
        
        if result:
            print(f"✅ Successfully converted: {output_path}")
            
            # Print summary
            print(f"📊 Conversion Summary:")
            print(f"   View Name: {result.get('ViewName', 'Unknown')}")
            print(f"   Number of predicted lines: {len(result.get('Lines', []))}")

            if result.get('Lines'):
                first_line = result['Lines'][0]
                print(f"   First line coordinates:")
                print(f"     Start: ({first_line['StartPointOrigin']['X']:.6f}, {first_line['StartPointOrigin']['Y']:.6f})")
                print(f"     End: ({first_line['EndPointOrigin']['X']:.6f}, {first_line['EndPointOrigin']['Y']:.6f})")
        else:
            print(f"❌ Failed to convert: {args.input}")
    
    elif os.path.isdir(args.input):
        # Convert all files in directory
        print(f"📁 Converting all prediction files in: {args.input}")
        
        prediction_files = [f for f in os.listdir(args.input) if f.endswith('_prediction.json')]
        
        if not prediction_files:
            print(f"❌ No prediction files found in {args.input}")
            return
        
        converted_count = 0
        total_lines = 0
        
        for filename in prediction_files:
            input_path = os.path.join(args.input, filename)

            # Get drawing ID from file path
            drawing_id, view_name = converter._extract_drawing_info_from_path(input_path)

            if drawing_id and drawing_id != 'unknown':
                # Tạo tên file an toàn cho view name
                safe_view_name = view_name.replace(' ', '_').replace('/', '_') if view_name else 'View1'
                output_filename = f"{drawing_id}_{safe_view_name}_DimClient_predicted.json"
            else:
                output_filename = filename.replace('_prediction.json', '_original_format.json')

            output_path = os.path.join(args.output, output_filename)

            print(f"   Converting: {filename} -> {output_filename}")
            result = converter.convert_prediction_to_original_format(input_path, output_path)
            
            if result:
                converted_count += 1
                total_lines += len(result.get('Lines', []))
                print(f"     ✅ {len(result.get('Lines', []))} lines converted")
            else:
                print(f"     ❌ Failed")
        
        print(f"\n🎉 Batch Conversion Complete!")
        print(f"   Files converted: {converted_count}/{len(prediction_files)}")
        print(f"   Total dimension lines: {total_lines}")
        print(f"   Output directory: {args.output}")
    
    else:
        print(f"❌ Input path not found: {args.input}")

def create_usage_examples():
    """Create a usage examples file."""
    examples = """
# Cách sử dụng convert_predictions_to_original.py

## 1. Convert một file prediction duy nhất:
python convert_predictions_to_original.py -i test_results/sample_1_prediction.json -o converted_results

## 2. Convert tất cả prediction files trong một thư mục:
python convert_predictions_to_original.py -i test_results -o converted_results

## 3. Sử dụng prepared data file khác:
python convert_predictions_to_original.py -i test_results -o converted_results -p data/prepared_data_v2.json

## 4. Chỉ convert và in ra console (không lưu file):
python convert_predictions_to_original.py -i test_results/sample_1_prediction.json

## Kết quả:
- File output sẽ có format giống như file DimClient gốc
- Tọa độ đã được denormalize về scale ban đầu
- Có thể import trực tiếp vào hệ thống CAD/Drawing

## Cấu trúc file output:
[
  {
    "ViewName": "Drawing View1",
    "Lines": [
      {
        "StartPoint": {"X": ..., "Y": ...},
        "EndPoint": {"X": ..., "Y": ...},
        "StartPointOrigin": {"X": ..., "Y": ...},
        "EndPointOrigin": {"X": ..., "Y": ...},
        "Width": ...,
        "Height": ...,
        "WidthOrigin": ...,
        "HeightOrigin": ...
      }
    ]
  }
]
"""
    
    with open('USAGE_EXAMPLES.md', 'w', encoding='utf-8') as f:
        f.write(examples)
    
    print("📝 Created usage examples file: USAGE_EXAMPLES.md")

if __name__ == '__main__':
    main()
    
    # Tạo file hướng dẫn sử dụng
    if not os.path.exists('USAGE_EXAMPLES.md'):
        create_usage_examples()
