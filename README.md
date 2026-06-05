# 🍎 PomegranateAI — Hệ thống Nhận diện Bệnh Quả Lựu

Bài tập lớn môn **Nông nghiệp thông minh**  
Sử dụng Deep Learning (EfficientNetB0) để phân loại 5 loại bệnh trên quả lựu từ ảnh chụp.

---

## 📋 Tổng quan

| Hạng mục | Chi tiết |
|---|---|
| Mô hình | EfficientNetB0 (Transfer Learning) |
| Dataset | Pomegranate Disease — 5,099 ảnh |
| Số class | 5 (4 bệnh + 1 healthy) |
| Accuracy (test set) | ~93% |
| Backend | Flask + Python |
| Frontend | HTML/CSS/JS + Bootstrap 5 |
| Triển khai | Google Colab + ngrok |

---

## 🦠 5 Loại Bệnh Được Nhận Diện

| Class | Tên bệnh | Tác nhân | Mức độ |
|---|---|---|---|
| `Healthy_Pomegranate` | Quả khỏe mạnh | — | ✅ Bình thường |
| `Alternaria_Pomegranate` | Đốm nâu Alternaria | Nấm *Alternaria punicae* | ⚠️ Trung bình |
| `Anthracnose_Pomegranate` | Thán thư | Nấm *Colletotrichum* | 🚨 Nguy hiểm |
| `Bacterial_Blight_Pomegranate` | Cháy lá vi khuẩn | Vi khuẩn *Xanthomonas* | 🚨 Nguy hiểm |
| `Cercospora_Pomegranate` | Đốm Cercospora | Nấm *Cercospora punicae* | ⚠️ Trung bình |

---

## 📂 Cấu Trúc Project

```
pomegranate-ai/
├── templates/
│   └── index.html          # Giao diện web
├── static/
│   └── uploads/            # Ảnh upload (tự tạo)
├── model/
│   └── pomegranate_final.h5  # Model đã train (copy từ Drive)
├── app.py                  # Flask backend
├── history.json            # Lịch sử dự đoán (tự tạo)
└── requirements.txt
```

---

## 🚀 Hướng Dẫn Chạy

### Cách 1 — Chạy trên Google Colab (Khuyến nghị)

Không cần download model về máy, chạy thẳng trên Colab nơi model đã được lưu.

**Bước 1:** Đăng ký tài khoản ngrok miễn phí tại [dashboard.ngrok.com](https://dashboard.ngrok.com) và copy Auth Token.

**Bước 2:** Thêm cell sau vào Google Colab (sau cell training):

```python
!pip install -q flask flask-cors pyngrok werkzeug

from pyngrok import ngrok, conf
import threading, os

NGROK_TOKEN = "paste_token_của_bạn_vào_đây"
conf.get_default().auth_token = NGROK_TOKEN
```

**Bước 3:** Chạy Flask server trên Colab:

```python
def run_flask():
    os.system("python /content/app.py")

thread = threading.Thread(target=run_flask, daemon=True)
thread.start()

import time; time.sleep(5)

public_url = ngrok.connect(5000)
print(f"🌐 API URL: {public_url}")
```

**Bước 4:** Copy URL ngrok vào ô **"Nhập ngrok URL"** trên giao diện web và nhấn **Kết nối**.

> ⚠️ Giữ tab Colab mở trong suốt quá trình demo. URL ngrok thay đổi mỗi lần chạy lại.

---

### Cách 2 — Chạy Local (Nếu ổ C còn trống)

**Bước 1:** Cài thư viện:
```bash
pip install -r requirements.txt
```

**Bước 2:** Copy model vào thư mục `model/`:
```
model/pomegranate_final.h5
```

**Bước 3:** Chạy server:
```bash
python app.py
```

**Bước 4:** Truy cập [http://127.0.0.1:5000](http://127.0.0.1:5000)

Khi chạy local, xóa phần `API_BASE` trong `index.html` — các `fetch()` dùng đường dẫn tương đối `/predict` là được.

---

## ⚙️ Quy Trình Hoạt Động

```
[User upload ảnh]
       ↓
[Frontend gửi POST /predict]
       ↓
[Flask nhận file → lưu vào static/uploads]
       ↓
[Tiền xử lý: resize 224×224 + EfficientNet preprocess_input]
       ↓
[EfficientNetB0 inference → softmax 5 class]
       ↓
[Trả về: class, confidence, khuyến nghị điều trị]
       ↓
[Lưu lịch sử vào history.json]
       ↓
[Hiển thị kết quả + biểu đồ confidence]
```

---

## 🧠 Chi Tiết Mô Hình

### Kiến trúc
```
Input (224×224×3)
    ↓
EfficientNetB0 (pretrained ImageNet, 30 layer cuối fine-tuned)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dropout(0.3)
    ↓
Dense(256, relu)
    ↓
Dropout(0.2)
    ↓
Dense(5, softmax)  ← Output
```

### Chiến lược huấn luyện

| Phase | Epochs | Learning Rate | Backbone |
|---|---|---|---|
| Phase 1 (Warm-up) | 10 | 1e-3 | Đóng băng |
| Phase 2 (Fine-tune) | 20 | 1e-5 | Mở 30 layer cuối |

### Xử lý class imbalance

Dùng **class weights** vì dataset mất cân bằng:

| Class | Ảnh | Weight |
|---|---|---|
| Healthy | 1,450 | 0.703 |
| Anthracnose | 1,166 | 0.875 |
| Bacterial_Blight | 966 | 1.056 |
| Alternaria | 886 | 1.151 |
| Cercospora | 631 | 1.617 |

---

## 📊 Kết Quả Đánh Giá

| Class | Nhận xét |
|---|---|
| Anthracnose | ✅ Nhận diện tốt — triệu chứng đặc trưng rõ ràng |
| Bacterial Blight | ✅ Nhận diện tốt — vết nứt toác dễ phân biệt |
| Healthy | ✅ Nhận diện tốt |
| Alternaria | ⚠️ Khó hơn — vết nâu khô khó phân biệt qua ảnh 2D |
| Cercospora | ⚠️ Khó hơn — triệu chứng giống vết côn trùng cắn |

### Hạn chế

Alternaria và Cercospora có độ chính xác thấp hơn do:
- Camera 2D không ghi được độ sần sùi, texture 3D của vỏ quả
- Hai bệnh này có triệu chứng màu sắc tương đồng với các bệnh khác
- Bào tử nấm Cercospora không thể thấy bằng camera thường

**Hướng cải thiện:** Kết hợp cảm biến đa phổ (multispectral) hoặc ảnh cận hồng ngoại NIR để tăng khả năng phân biệt.

---

## 🔌 API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/predict` | Upload ảnh, trả về kết quả dự đoán |
| `GET` | `/history` | Lấy danh sách lịch sử dự đoán |
| `POST` | `/history/clear` | Xóa toàn bộ lịch sử |

### Ví dụ response `/predict`:
```json
{
  "success": true,
  "prediction": "Anthracnose",
  "confidence": 97.84,
  "recommendation": {
    "status": "danger",
    "icon": "🚨",
    "title": "Bệnh Thán Thư Anthracnose",
    "description": "Nấm Colletotrichum...",
    "actions": ["Thu gom tiêu hủy ngay...", "Phun Copper Oxychloride..."]
  },
  "image_url": "/uploads/abc123_image.jpg",
  "timestamp": "2024-08-13 15:17:00"
}
```

---

## 📦 Requirements

```
flask==3.0.3
tensorflow==2.17.0
numpy
Pillow
werkzeug
flask-cors
pyngrok
```

---

## 👨‍💻 Công Nghệ Sử Dụng

- **AI/ML:** TensorFlow 2.x, Keras, EfficientNetB0
- **Backend:** Python, Flask, Flask-CORS
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla), Bootstrap 5
- **Training:** Google Colab (GPU T4)
- **Triển khai:** ngrok tunnel
- **Lưu trữ lịch sử:** JSON file
