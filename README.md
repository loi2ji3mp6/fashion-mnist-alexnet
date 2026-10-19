# AlexNet for Fashion-MNIST

## 1. Giới thiệu

Dự án triển khai mô hình học sâu AlexNet để phân loại hình ảnh trong bộ dữ liệu Fashion-MNIST bằng PyTorch.

Mô hình được huấn luyện trong 50 epoch và sử dụng GPU CUDA khi khả dụng.

## 2. Công nghệ sử dụng

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- CUDA (nếu có GPU tương thích)

## 3. Mô hình

AlexNet được điều chỉnh cho bài toán phân loại ảnh Fashion-MNIST gồm 10 lớp:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Các kỹ thuật được sử dụng:

- ReLU activation
- Max Pooling
- Dropout để hạn chế overfitting
- Data Augmentation với Random Horizontal Flip, Random Rotation và Random Affine
- Adam optimizer
- Cross-Entropy Loss

Ảnh đầu vào được chuyển thành 3 kênh và resize về 96 × 96 pixel để phù hợp với kiến trúc AlexNet được triển khai.

## 4. Kết quả huấn luyện

- Số epoch: 50
- Test Accuracy cao nhất: 90.46%
- Thời gian huấn luyện: khoảng 83.4 phút
- GPU: NVIDIA GeForce GTX 1060 3GB

Độ chính xác theo từng lớp:

| Class | Accuracy |
|---|---:|
| T-shirt/top | 87.70% |
| Trouser | 98.10% |
| Pullover | 84.80% |
| Dress | 90.30% |
| Coat | 87.00% |
| Sandal | 97.70% |
| Shirt | 68.80% |
| Sneaker | 96.00% |
| Bag | 98.10% |
| Ankle boot | 96.10% |

## 5. Cấu trúc dự án

```text
fashion-mnist-alexnet/
├── train.py
├── requirements.txt
├── .gitignore
├── data/
└── outputs/
    ├── best_model.pth
    ├── last_model.pth
    ├── history.json
    ├── results.json
    └── training_curves.png
```

Thư mục `data/` chứa dữ liệu Fashion-MNIST tải tự động khi chạy chương trình. Thư mục `.venv/` là môi trường Python cục bộ và không được đưa lên GitHub.

## 6. Cài đặt và chạy

Tạo môi trường Python và cài đặt các thư viện được liệt kê trong `requirements.txt`.

Chạy chương trình:

```bash
python train.py
```

Nếu máy có GPU CUDA tương thích, chương trình sẽ tự động sử dụng GPU; nếu không, chương trình sẽ chạy bằng CPU.

## 7. Kết quả đầu ra

Sau khi huấn luyện, thư mục `outputs/` chứa:

- `best_model.pth`: checkpoint có test accuracy cao nhất trong quá trình huấn luyện.
- `last_model.pth`: trọng số tại cuối lần chạy.
- `training_curves.png`: biểu đồ Loss và Accuracy.
- `history.json`: lịch sử huấn luyện.
- `results.json`: kết quả đánh giá.

## 8. Ghi chú

Dự án được thực hiện cho mục đích học tập và thực hành Deep Learning.