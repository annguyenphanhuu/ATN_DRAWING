# DFM ATN Drawing - Dimension Detection System

Hệ thống AI phát hiện và phân tích kích thước trong bản vẽ kỹ thuật sử dụng Graph Neural Networks (GNN) và PyTorch.

## Mô tả dự án

Dự án này sử dụng các mô hình học sâu tiên tiến để tự động phát hiện và phân tích các đường kích thước trong bản vẽ kỹ thuật. Hệ thống được xây dựng trên nền tảng PyTorch với kiến trúc Graph Neural Network để xử lý dữ liệu hình học phức tạp.

## Yêu cầu hệ thống

### Phần mềm cần thiết
- Python 3.8 hoặc cao hơn
- CUDA-compatible GPU (khuyến nghị cho training)
- Git

### Thư viện Python
Xem file `requirements.txt` để biết danh sách đầy đủ các thư viện cần thiết.

## Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd DFM_ATN_DRAWING
```

### 2. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies

**Lưu ý quan trọng**: Do các thư viện PyTorch phụ thuộc vào phiên bản CUDA của hệ thống, bạn nên cài đặt thủ công:

#### Cài đặt PyTorch và các thư viện liên quan:
```bash
# Kiểm tra phiên bản CUDA
nvidia-smi

# Cài đặt PyTorch (thay đổi theo phiên bản CUDA của bạn)
# Ví dụ cho CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt PyTorch Geometric
pip install torch-geometric

# Cài đặt các thư viện bổ sung cho PyTorch Geometric
pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

#### Cài đặt các thư viện còn lại:
```bash
pip install numpy scipy scikit-learn matplotlib seaborn
```

### 4. Chuẩn bị dữ liệu

Đảm bảo cấu trúc thư mục như sau:
```
DFM_ATN_DRAWING/
├── data/
│   ├── input/          # Dữ liệu đầu vào (bản vẽ gốc)
│   ├── output/         # Kết quả xử lý
│   └── prepared_data_v3.json  # Dữ liệu đã được tiền xử lý
├── evaluation_results/ # Kết quả đánh giá mô hình
└── best_model_v3.pth  # Mô hình đã được huấn luyện
```

## Cách sử dụng

### 1. Tiền xử lý dữ liệu
```bash
python data_preparation.py
```

### 2. Huấn luyện mô hình
```bash
python train.py
```

### 3. Đánh giá mô hình
```bash
python evaluate.py
```

### 4. Chạy đánh giá chi tiết
```bash
python run_evaluation.py
```

### 5. Kiểm tra mẫu đơn lẻ
```bash
python test_single_sample.py
```

### 6. So sánh các mô hình
```bash
python compare_models.py
```

### 7. Trực quan hóa kết quả
```bash
python visualize.py
```

### 8. Chuyển đổi dự đoán về định dạng gốc
```bash
python convert_predictions_to_original.py
```

## Cấu trúc dự án

- `model.py`: Định nghĩa kiến trúc mô hình GNN
- `train.py`: Script huấn luyện mô hình
- `evaluate.py`: Script đánh giá mô hình
- `data_preparation.py`: Tiền xử lý dữ liệu
- `visualize.py`: Trực quan hóa kết quả
- `prediction_converter.py`: Chuyển đổi định dạng dự đoán
- `requirements.txt`: Danh sách thư viện cần thiết

## Tham số mô hình

Mô hình sử dụng các tham số chính:
- **Node feature dimension**: 15 (đặc trưng hình học của điểm)
- **Edge feature dimension**: 8 (đặc trưng của cạnh nối)
- **Hidden dimension**: 256 (kích thước lớp ẩn)
- **Embedding dimension**: 128 (kích thước vector nhúng)

## Kết quả

Kết quả đánh giá và so sánh được lưu trong thư mục `evaluation_results/` bao gồm:
- Hình ảnh so sánh dự đoán vs thực tế
- File JSON chứa dự đoán chi tiết
- Báo cáo metrics đánh giá

## Xử lý sự cố

### Lỗi CUDA
Nếu gặp lỗi liên quan đến CUDA:
```bash
# Kiểm tra CUDA có sẵn
python -c "import torch; print(torch.cuda.is_available())"

# Cài đặt phiên bản CPU nếu không có GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Lỗi thư viện PyTorch Geometric
```bash
# Cài đặt lại PyTorch Geometric
pip uninstall torch-geometric torch-cluster torch-scatter torch-sparse
pip install torch-geometric
pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Lỗi thiếu dữ liệu
Đảm bảo file `data/prepared_data_v3.json` tồn tại. Nếu không, chạy:
```bash
python data_preparation.py
```

## Đóng góp

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## Giấy phép

Dự án này được phân phối dưới giấy phép [LICENSE]. Xem file `LICENSE` để biết thêm thông tin.

## Liên hệ

Để được hỗ trợ hoặc báo cáo lỗi, vui lòng tạo issue trên GitHub repository.
