# 🍎 PomegranateAI — Hệ thống Nhận diện Bệnh Quả Lựu bằng AI

> **Môn học:** Nông nghiệp thông minh  
> **Trường:** Đại học Đại Nam — Khoa Công nghệ thông tin  
> **Công nghệ cốt lõi:** EfficientNetB0 Transfer Learning + Flask + Firebase + Telegram Bot

---

## 🎯 Giới thiệu đề tài

Bệnh cây là một trong những nguyên nhân chính gây thất thu trong nông nghiệp. Việc phát hiện bệnh sớm và chính xác giúp người nông dân can thiệp kịp thời, giảm thiệt hại kinh tế. Tuy nhiên, chẩn đoán thủ công đòi hỏi chuyên môn cao và tốn nhiều thời gian.

**PomegranateAI** là hệ thống ứng dụng trí tuệ nhân tạo để tự động nhận diện 5 loại bệnh phổ biến trên quả lựu chỉ từ ảnh chụp hoặc video, trả kết quả trong vài giây kèm theo hướng dẫn điều trị cụ thể.

### Những điểm nổi bật của dự án

- ✅ **Độ chính xác ~93%** trên tập kiểm tra 5,099 ảnh
- 🎬 **Phân tích video** — quay video vườn lựu, hệ thống tự trích frame và đưa ra kết luận tổng hợp
- 🔔 **Cảnh báo Telegram tức thì** — nhận thông báo ngay khi phát hiện bệnh dù không ngồi trước máy tính
- ☁️ **Lưu trữ đám mây** — lịch sử dự đoán được lưu tự động lên Firebase Firestore
- 🌐 **Giao diện web hiện đại** — không cần cài app, dùng trình duyệt là được

---

## 📊 Thông số kỹ thuật tổng quan

| Hạng mục | Chi tiết |
|---|---|
| Mô hình AI | EfficientNetB0 (Transfer Learning từ ImageNet) |
| Tập dữ liệu | Pomegranate Disease Dataset — 5,099 ảnh |
| Số lớp phân loại | 5 (4 bệnh + 1 healthy) |
| Độ chính xác (test set) | ~93% |
| Framework AI | TensorFlow 2.17 / Keras |
| Backend | Python 3.10 + Flask 3.0 |
| Frontend | HTML5 + CSS3 + JavaScript (Vanilla) |
| Cơ sở dữ liệu | Firebase Firestore (NoSQL đám mây) |
| Thông báo | Telegram Bot API |
| Xử lý video | OpenCV 4.8 |
| Môi trường train | Google Colab (GPU Tesla T4) |

---

## 🦠 5 Loại Bệnh Được Nhận Diện

| # | Nhãn lớp | Tên bệnh | Tác nhân sinh học | Mức độ nguy hiểm |
|---|---|---|---|---|
| 1 | `Healthy_Pomegranate` | Quả khỏe mạnh | — | ✅ Bình thường |
| 2 | `Alternaria_Pomegranate` | Đốm nâu Alternaria | Nấm *Alternaria punicae* | ⚠️ Trung bình |
| 3 | `Anthracnose_Pomegranate` | Bệnh thán thư | Nấm *Colletotrichum gloeosporioides* | 🚨 Nguy hiểm |
| 4 | `Bacterial_Blight_Pomegranate` | Cháy lá vi khuẩn | Vi khuẩn *Xanthomonas axonopodis* | 🚨 Nguy hiểm |
| 5 | `Cercospora_Pomegranate` | Đốm Cercospora | Nấm *Cercospora punicae* | ⚠️ Trung bình |

---

## 📂 Cấu Trúc Thư Mục

```
POMEGRANATES-DISEASE/
│
├── 📁 model/
│   └── pomegranate_final.h5      ← File model đã huấn luyện (copy từ Google Drive về đây)
│
├── 📁 templates/
│   └── index.html                ← Giao diện web (Flask dùng thư mục này)
│
├── 📁 static/
│   └── uploads/                  ← Ảnh/video upload sẽ được lưu ở đây (tự tạo khi chạy)
│
├── app.py                        ← Flask backend — file chính, chạy file này
├── firebase_key.json             ← Khóa kết nối Firebase (KHÔNG đưa lên GitHub)
├── check_model.py                ← Script kiểm tra model có load được không
├── requirements.txt              ← Danh sách thư viện cần cài
└── README.md                     ← File này
```

> ⚠️ **Lưu ý quan trọng:** File `firebase_key.json` chứa thông tin xác thực nhạy cảm. Tuyệt đối **không commit** file này lên GitHub. Hãy thêm vào `.gitignore`.

---

## 🛠️ Hướng Dẫn Cài Đặt và Chạy

Có hai cách chạy dự án. Nếu bạn đang train model trên **Google Colab** thì dùng Cách 1. Nếu máy tính có đủ RAM và ổ cứng thì dùng Cách 2.

---

### ▶️ Cách 1 — Chạy trên Google Colab + ngrok (Khuyến nghị cho demo)

Cách này phù hợp khi model của bạn đang nằm trên Google Drive và bạn không muốn tải về máy (file .h5 khá nặng).

**Bước 1 — Đăng ký ngrok (miễn phí)**

Truy cập [dashboard.ngrok.com](https://dashboard.ngrok.com), đăng ký tài khoản và copy **Auth Token** của bạn.

**Bước 2 — Cài thư viện trong Colab**

Thêm cell mới và chạy:
```python
!pip install -q flask flask-cors pyngrok werkzeug firebase-admin opencv-python-headless
```

**Bước 3 — Copy các file dự án lên Colab**

```python
# Upload app.py, templates/index.html, firebase_key.json lên /content/
# Hoặc clone từ GitHub:
!git clone https://github.com/your-username/pomegranate-ai.git /content/pomegranate-ai
%cd /content/pomegranate-ai
```

**Bước 4 — Cấu hình ngrok và chạy server**

```python
from pyngrok import ngrok, conf
import threading, os, time

# Dán Auth Token của bạn vào đây
NGROK_TOKEN = "paste_token_của_bạn_vào_đây"
conf.get_default().auth_token = NGROK_TOKEN

# Chạy Flask trong thread riêng
def run_flask():
    os.system("python /content/pomegranate-ai/app.py")

thread = threading.Thread(target=run_flask, daemon=True)
thread.start()
time.sleep(5)  # Chờ Flask khởi động

# Tạo tunnel public
public_url = ngrok.connect(5000)
print(f"🌐 Truy cập ứng dụng tại: {public_url}")
```

**Bước 5 — Mở giao diện**

Copy URL ngrok vừa in ra (dạng `https://xxxx.ngrok.io`) và mở trên trình duyệt. Ứng dụng đã sẵn sàng!

> ⚠️ Giữ tab Colab **luôn mở** trong suốt quá trình demo. URL ngrok sẽ thay đổi mỗi lần chạy lại.

---

### ▶️ Cách 2 — Chạy Local trên máy tính cá nhân

Cách này phù hợp nếu bạn đã tải model về máy và muốn chạy không cần internet (ngoại trừ Firebase/Telegram).

**Bước 1 — Clone dự án**

```bash
git clone https://github.com/your-username/pomegranate-ai.git
cd pomegranate-ai
```

**Bước 2 — Tạo môi trường ảo (khuyến nghị)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**Bước 3 — Cài đặt thư viện**

```bash
pip install -r requirements.txt
```

**Bước 4 — Copy file model**

Tải file `pomegranate_final.h5` từ Google Drive về và đặt vào thư mục `model/`:
```
model/
└── pomegranate_final.h5   ← Đặt file vào đây
```

**Bước 5 — Đặt file Firebase Key**

Tải `firebase_key.json` từ Firebase Console (Project Settings → Service Accounts → Generate new private key) và đặt vào thư mục gốc dự án.

**Bước 6 — Chạy ứng dụng**

```bash
python app.py
```

**Bước 7 — Mở trình duyệt**

Truy cập [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ⚙️ Quy Trình Hoạt Động Chi Tiết

### Luồng phân tích ảnh

```
Người dùng chọn ảnh (JPG/PNG/WEBP, ≤16MB)
        ↓
Frontend hiển thị preview, người dùng nhấn "Phân tích"
        ↓
Gửi HTTP POST /predict (multipart/form-data)
        ↓
Flask kiểm tra định dạng + kích thước file
        ↓
Đổi tên file bằng UUID → lưu vào static/uploads/
        ↓
Tiền xử lý: load_img → resize 224×224 → img_to_array
            → expand_dims → preprocess_input (pixel về [-1, +1])
        ↓
EfficientNetB0 model.predict() → vector xác suất (1, 5)
        ↓
argmax() → lấy lớp có xác suất cao nhất + confidence %
        ↓
get_recommendation() → khuyến nghị điều trị tương ứng
        ↓
Lưu kết quả lên Firebase Firestore (song song)
Gửi cảnh báo Telegram nếu có bệnh (song song)
        ↓
Trả về JSON response → giao diện hiển thị kết quả
```

### Luồng phân tích video

```
Người dùng upload video (MP4/AVI/MOV, ≤100MB)
        ↓
cv2.VideoCapture đọc metadata (fps, tổng frame, thời lượng)
        ↓
Tính sample_count = min(30, max(5, số giây video))
Tính interval = total_frames ÷ sample_count
        ↓
Trích frame tại vị trí 0, interval, 2×interval, ...
        ↓
Mỗi frame: BGR→RGB → resize 224×224 → preprocess → predict
        ↓
Thu thập kết quả tất cả frame vào dict đếm tần suất
        ↓
Majority Voting: chọn nhãn xuất hiện nhiều nhất
Tính avg_confidence của nhãn thắng
        ↓
Lưu Firestore + Gửi Telegram (thumbnail frame đại diện)
        ↓
Trả về JSON: nhãn cuối, confidence, phân bổ frame, khuyến nghị
```

---

## 🧠 Chi Tiết Mô Hình AI

### Tại sao chọn EfficientNetB0?

EfficientNetB0 được Google đề xuất năm 2019, sử dụng kỹ thuật **Compound Scaling** — mở rộng đồng thời chiều rộng, chiều sâu và độ phân giải của mạng theo một hệ số thống nhất. Kết quả là mô hình đạt độ chính xác cao hơn ResNet50 và MobileNetV2 nhưng với số lượng tham số ít hơn đáng kể (~5.3M), phù hợp với tài nguyên phần cứng hạn chế.

### Kiến trúc mô hình

```
Input (224 × 224 × 3)
        ↓
[BACKBONE — Đóng băng ở Phase 1, mở 30 layer cuối ở Phase 2]
EfficientNetB0 (pretrained ImageNet)
  Stem Conv → MBConv Blocks × 7 nhóm → Head Conv
  Output: feature map 7×7×1280
        ↓
[CUSTOM HEAD — Luôn huấn luyện]
GlobalAveragePooling2D    → vector 1280 chiều
BatchNormalization
Dropout(0.3)              → chống overfitting
Dense(256, activation='relu')
Dropout(0.2)
Dense(5, activation='softmax')   ← Output: xác suất 5 lớp
```

### Chiến lược huấn luyện 2 giai đoạn

| Giai đoạn | Epochs | Learning Rate | Backbone | Mục đích |
|---|---|---|---|---|
| Phase 1 — Warm-up | 10 | 1e-3 | Đóng băng toàn bộ | Học nhanh Custom Head, tránh phá vỡ trọng số pretrained |
| Phase 2 — Fine-tune | 20 | 1e-5 | Mở 30 layer cuối | Tinh chỉnh đặc trưng bậc cao phù hợp với ảnh bệnh lựu |

### Xử lý mất cân bằng dữ liệu

Dataset có sự chênh lệch số lượng giữa các lớp nên sử dụng **Class Weights** để model không bị thiên vị về lớp đa số:

| Lớp | Số ảnh | Class Weight | Ý nghĩa |
|---|---|---|---|
| Healthy | 1,450 | 0.703 | Nhiều ảnh nhất → trọng số thấp nhất |
| Anthracnose | 1,166 | 0.875 | — |
| Bacterial Blight | 966 | 1.056 | — |
| Alternaria | 886 | 1.151 | — |
| Cercospora | 631 | 1.617 | Ít ảnh nhất → trọng số cao nhất để tăng chú ý |

---

## 🔌 API Endpoints

| Method | Endpoint | Mô tả | Input |
|---|---|---|---|
| `GET` | `/` | Trả về giao diện web | — |
| `POST` | `/predict` | Phân tích ảnh | `multipart/form-data` field `file` |
| `POST` | `/predict_video` | Phân tích video | `multipart/form-data` field `file` |
| `GET` | `/history` | Lấy lịch sử dự đoán | — |
| `POST` | `/history/clear` | Xóa toàn bộ lịch sử | — |

### Ví dụ response `/predict` (thành công)

```json
{
  "success": true,
  "prediction": "Anthracnose",
  "confidence": 97.84,
  "recommendation": {
    "status": "danger",
    "icon": "🚨",
    "title": "Bệnh Thán Thư Anthracnose",
    "description": "Gây ra bởi nấm Colletotrichum gloeosporioides...",
    "actions": [
      "Thu gom và tiêu hủy ngay quả bệnh",
      "Phun Copper Oxychloride 0.3% toàn vườn",
      "Phun phòng định kỳ 7 ngày/lần"
    ],
    "severity": "high"
  },
  "image_url": "/static/uploads/abc123_image.jpg",
  "timestamp": "2024-08-13 15:17:00"
}
```

### Ví dụ response `/predict_video` (thành công)

```json
{
  "success": true,
  "source": "video",
  "duration": 12.5,
  "frame_count": 12,
  "final_disease": "Alternaria",
  "final_conf": 88.3,
  "disease_counts": {
    "Alternaria": 8,
    "Healthy": 3,
    "Cercospora": 1
  },
  "frame_results": [
    { "frame_idx": 0, "time_sec": 0.0, "prediction": "Alternaria", "confidence": 91.2, "status": "warning" }
  ],
  "recommendation": { "..." : "..." },
  "timestamp": "2024-08-13 15:20:00"
}
```

---

## 📊 Kết Quả Đánh Giá Mô Hình

| Lớp bệnh | Precision | Recall | F1-Score | Nhận xét |
|---|---|---|---|---|
| Healthy | 0.960 | 0.960 | 0.960 | ✅ Tốt — đặc trưng vỏ đỏ đều rõ ràng |
| Anthracnose | 0.958 | 0.960 | 0.959 | ✅ Tốt — vết đen hoại tử đặc trưng |
| Bacterial Blight | 0.917 | 0.880 | 0.898 | ✅ Khá — vết nứt toác dễ nhận |
| Alternaria | 0.920 | 0.920 | 0.920 | ⚠️ Trung bình — vết nâu khô dễ nhầm |
| Cercospora | 0.920 | 0.920 | 0.920 | ⚠️ Trung bình — triệu chứng giống vết côn trùng |
| **Macro Avg** | **0.935** | **0.928** | **0.931** | **Overall Accuracy: ~93%** |

### Tại sao Alternaria và Cercospora khó nhận diện hơn?

- Camera 2D không ghi lại được độ sần sùi và texture 3D của bề mặt vỏ quả
- Màu sắc vết bệnh của hai lớp này tương đồng với nhau và với Bacterial Blight giai đoạn sớm
- Bào tử nấm Cercospora quá nhỏ, không thể phân biệt bằng camera thường

**Hướng cải thiện:** Kết hợp cảm biến đa phổ (multispectral) hoặc ảnh cận hồng ngoại NIR để tăng khả năng phân biệt hai lớp này.

---

## 🗄️ Cấu Trúc Dữ Liệu Firebase Firestore

Mỗi lần phân tích được lưu thành một document trong collection `predictions`:

```json
{
  "id": "a1b2c3d4",
  "filename": "anh_qua_luu.jpg",
  "saved_as": "uuid_anh_qua_luu.jpg",
  "prediction": "Anthracnose",
  "confidence": 97.84,
  "severity": "high",
  "timestamp": "2024-08-13 15:17:00",
  "source": "image",
  "frame_count": null,
  "duration": null
}
```

> Với bản ghi từ video, `frame_count` và `duration` sẽ có giá trị thực.

---

## 📦 Danh Sách Thư Viện (requirements.txt)

```
flask==3.0.3
tensorflow==2.17.0
numpy
Pillow
werkzeug
flask-cors
opencv-python
firebase-admin
requests
```

Cài đặt tất cả bằng một lệnh:
```bash
pip install -r requirements.txt
```

> 💡 **Lưu ý:** TensorFlow 2.17 khá nặng (~500MB). Nếu dùng Colab thì đã được cài sẵn. Nếu chạy local, hãy đảm bảo Python 3.10 và dùng virtual environment để tránh xung đột.

---

## 🔒 Bảo Mật

- `firebase_key.json` — **không** đưa lên GitHub, thêm vào `.gitignore`
- `TELEGRAM_TOKEN` — nên chuyển sang biến môi trường (`os.getenv`) thay vì hardcode
- File upload được đổi tên bằng UUID để tránh path traversal attack
- Giới hạn kích thước: ảnh ≤16MB, video ≤100MB

---

## 🤝 Đóng Góp và Phát Triển

Dự án này được xây dựng như một nền tảng mở rộng được. Nếu bạn muốn phát triển tiếp:

1. Fork repository này
2. Tạo branch mới: `git checkout -b feature/ten-tinh-nang`
3. Commit thay đổi: `git commit -m "Thêm tính năng X"`
4. Push và tạo Pull Request

### Ý tưởng phát triển tiếp theo

- [ ] Thêm nhận diện bệnh cho các loại cây khác (cam, ổi, cà chua...)
- [ ] Phát triển app mobile (Flutter/React Native + TensorFlow Lite)
- [ ] Tích hợp camera IP theo dõi vườn realtime
- [ ] Kết hợp cảm biến IoT (nhiệt độ, độ ẩm) để dự báo nguy cơ bệnh
- [ ] Triển khai lên Cloud (Google Cloud Run / AWS Lambda)
- [ ] Thêm chức năng xuất báo cáo PDF

---

## 👨‍💻 Thông Tin Tác Giả

| Thông tin | Chi tiết |
|---|---|
| Sinh viên | Trần Hồng Quân |
| Lớp | CNTT 16-02 |
| Trường | Đại học Đại Nam |
| Môn học | TPTM & NNTM |
| Năm học | 2026 |

---

## 📄 Giấy Phép

Dự án này được phát triển phục vụ mục đích học thuật. Vui lòng ghi nguồn nếu sử dụng lại.

---

*Nếu bạn thấy README này hữu ích, hãy ⭐ star repository để ủng hộ nhé!*
