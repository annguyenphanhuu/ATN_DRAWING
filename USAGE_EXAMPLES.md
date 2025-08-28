
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
