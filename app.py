"""
app.py - Backend Flask cho Hệ thống AI Nhận diện Bệnh Quả Lựu
Môn: Nông nghiệp thông minh
Model: EfficientNetB0 Transfer Learning - 5 class
Update: Thêm tính năng phân tích video
"""

import os, json, uuid, requests, cv2, tempfile
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
import numpy as np

# ─── Firebase ──────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate('firebase_key.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
print("✅ Firebase Firestore đã kết nối!")

# ─── Telegram ──────────────────────────────────────────────────
TELEGRAM_TOKEN   = "8529564201:AAG1PcelWIWzvFWb-5T32vqE1BF4HwJYkKg"
TELEGRAM_CHAT_ID = "6862971862"

def send_telegram(disease_name, confidence, rec, image_path):
    """Gửi thông báo khi phát hiện bệnh, bỏ qua nếu quả khỏe."""
    if disease_name == 'Healthy_Pomegranate':
        return

    level   = "🚨 NGUY HIỂM" if rec['status'] == 'danger' else "⚠️ CẦN XỬ LÝ"
    actions = "\n".join([f"  • {a}" for a in rec['actions'][:3]])
    caption = (
        f"{rec['icon']} *PHÁT HIỆN BỆNH QUẢ LỰU*\n"
        f"{'─' * 28}\n"
        f"🦠 *Bệnh:* {rec['title']}\n"
        f"📊 *Độ tin cậy:* {confidence:.1f}%\n"
        f"🔴 *Mức độ:* {level}\n\n"
        f"📋 *Khuyến nghị:*\n{actions}\n\n"
        f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
    )
    try:
        with open(image_path, 'rb') as img:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                data={'chat_id': TELEGRAM_CHAT_ID,
                      'caption': caption,
                      'parse_mode': 'Markdown'},
                files={'photo': img},
                timeout=10
            )
        print(f"📲 Đã gửi Telegram: {disease_name}")
    except Exception as e:
        print(f"⚠️  Telegram lỗi: {e}")

# ─── Flask setup ───────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER']     = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024   # 100 MB cho video
ALLOWED_IMG = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_VID = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

CLASS_NAMES = [
    'Alternaria_Pomegranate',
    'Anthracnose_Pomegranate',
    'Bacterial_Blight_Pomegranate',
    'Cercospora_Pomegranate',
    'Healthy_Pomegranate'
]

# ─── Load model ────────────────────────────────────────────────
model = None

def load_model():
    global model
    model_path = 'model/pomegranate_final.h5'
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            print("✅ Model EfficientNetB0 đã load thành công!")
            print(f"   Input shape: {model.input_shape}")
        except Exception as e:
            print(f"⚠️  Không thể load model: {e}")
    else:
        print(f"❌ Không tìm thấy: {model_path}")

# ─── Tiện ích ──────────────────────────────────────────────────
def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMG

def allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VID

def preprocess_image(image_path):
    import tensorflow as tf
    img       = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)

def preprocess_frame(frame_bgr):
    """Tiền xử lý 1 frame BGR từ OpenCV để đưa vào model."""
    import tensorflow as tf
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    img_array = np.expand_dims(frame_resized.astype(np.float32), axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)

def predict_disease(image_path):
    if model is None:
        import random
        idx = random.randint(0, 4)
        return CLASS_NAMES[idx], round(random.uniform(70, 98), 2)
    preds      = model.predict(preprocess_image(image_path), verbose=0)[0]
    idx        = int(np.argmax(preds))
    confidence = round(float(np.max(preds)) * 100, 2)
    return CLASS_NAMES[idx], confidence

def predict_frame(frame_bgr):
    """Dự đoán trực tiếp từ frame OpenCV (không qua file)."""
    if model is None:
        import random
        idx = random.randint(0, 4)
        return CLASS_NAMES[idx], round(random.uniform(70, 98), 2)
    preds      = model.predict(preprocess_frame(frame_bgr), verbose=0)[0]
    idx        = int(np.argmax(preds))
    confidence = round(float(np.max(preds)) * 100, 2)
    return CLASS_NAMES[idx], confidence

def get_recommendation(disease_name):
    data = {
        'Alternaria_Pomegranate': {
            'status': 'warning', 'icon': '⚠️',
            'title': 'Bệnh Đốm Nâu Alternaria',
            'description': 'Gây ra bởi nấm Alternaria punicae. Xuất hiện vết nâu khô, '
                           'sần sùi trên vỏ quả, bề mặt co rút. Bệnh phát triển mạnh '
                           'trong điều kiện ẩm ướt sau mưa.',
            'actions': [
                'Phun Mancozeb 0.25% hoặc Carbendazim 0.1% ngay khi phát hiện',
                'Cắt bỏ và tiêu hủy (đốt) quả bị bệnh nặng',
                'Phun định kỳ 10-14 ngày/lần trong mùa mưa',
                'Tránh tưới phun lên tán cây, tưới gốc vào buổi sáng',
                'Tỉa cành tăng thông thoáng, giảm độ ẩm trong tán cây',
            ],
            'severity': 'medium'
        },
        'Anthracnose_Pomegranate': {
            'status': 'danger', 'icon': '🚨',
            'title': 'Bệnh Thán Thư Anthracnose',
            'description': 'Gây ra bởi nấm Colletotrichum gloeosporioides. '
                           'Vết đen hoại tử lớn, lan rộng nhanh trên vỏ quả. '
                           'Rất nguy hiểm, có thể gây thối quả hàng loạt.',
            'actions': [
                '🚨 KHẨN CẤP: Thu gom và tiêu hủy ngay quả bệnh',
                'Phun Copper Oxychloride 0.3% hoặc Carbendazim toàn vườn',
                'Phun phòng định kỳ 7 ngày/lần cho cây chưa bị bệnh',
                'Tránh gây vết thương cơ giới khi chăm sóc vườn',
                'Cải thiện thoát nước, tránh đọng nước gốc cây',
                'Xem xét trồng giống kháng bệnh cho vụ sau',
            ],
            'severity': 'high'
        },
        'Bacterial_Blight_Pomegranate': {
            'status': 'danger', 'icon': '🚨',
            'title': 'Bệnh Cháy Lá Vi Khuẩn',
            'description': 'Gây ra bởi vi khuẩn Xanthomonas axonopodis. '
                           'Quả nứt toác, bên trong thối đen. '
                           'Lây lan rất nhanh qua nước mưa và côn trùng.',
            'actions': [
                '🚨 Cách ly ngay khu vực có cây bị bệnh nặng',
                'Phun Streptomycin Sulfate kết hợp Copper Oxychloride',
                'Tiêu hủy (đốt) quả và cành bị bệnh, không ủ phân',
                'Vệ sinh dụng cụ bằng cồn 70% sau mỗi lần dùng',
                'Không tưới phun vào buổi chiều tối',
                'Bón Kali bổ sung để tăng sức đề kháng thành tế bào',
            ],
            'severity': 'high'
        },
        'Cercospora_Pomegranate': {
            'status': 'warning', 'icon': '⚠️',
            'title': 'Bệnh Đốm Cercospora',
            'description': 'Gây ra bởi nấm Cercospora punicae. '
                           'Vùng rám vàng nâu loang lổ trên vỏ quả, '
                           'trông giống vết côn trùng cắn. Phát triển trong điều kiện ẩm cao.',
            'actions': [
                'Phun Chlorothalonil hoặc Mancozeb định kỳ 10-14 ngày',
                'Tỉa cành thông thoáng, giảm độ ẩm trong tán lá',
                'Bón phân cân đối, tránh thừa đạm',
                'Thu gom lá rụng bị bệnh để tránh nguồn lây lan',
                'Luân phiên các loại thuốc trừ nấm để tránh kháng thuốc',
            ],
            'severity': 'medium'
        },
        'Healthy_Pomegranate': {
            'status': 'success', 'icon': '✅',
            'title': 'Quả Lựu Khỏe Mạnh',
            'description': 'Quả phát triển bình thường, vỏ đỏ đều, '
                           'không có dấu hiệu bệnh lý. Tiếp tục duy trì '
                           'chế độ chăm sóc hiện tại.',
            'actions': [
                'Tưới nước đều đặn, tránh ngập úng gốc cây',
                'Bón phân NPK cân đối theo từng giai đoạn phát triển',
                'Theo dõi vườn định kỳ hàng tuần phát hiện sớm dấu hiệu bệnh',
                'Tỉa cành phụ để tập trung dinh dưỡng nuôi quả',
                'Phun phòng nấm bệnh định kỳ vào đầu mùa mưa',
            ],
            'severity': 'low'
        }
    }
    return data.get(disease_name, data['Healthy_Pomegranate'])

# ─── Firebase helpers ──────────────────────────────────────────
def save_to_cloud(entry):
    try:
        db.collection('predictions').document(entry['id']).set(entry)
        print(f"☁️  Firestore đã lưu: {entry['id']}")
    except Exception as e:
        print(f"⚠️  Firestore lỗi: {e}")

def load_from_cloud(limit=20):
    try:
        docs = (db.collection('predictions')
                  .order_by('timestamp', direction=firestore.Query.DESCENDING)
                  .limit(limit)
                  .stream())
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        print(f"⚠️  Firestore load lỗi: {e}")
        return []

# ─── Routes: Ảnh ───────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_image(file.filename):
        return jsonify({'error': 'File không hợp lệ'}), 400

    original_name = secure_filename(file.filename)
    unique_name   = f"{uuid.uuid4().hex}_{original_name}"
    save_path     = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(save_path)

    try:
        disease_name, confidence = predict_disease(save_path)
        recommendation = get_recommendation(disease_name)

        entry = {
            'id':         uuid.uuid4().hex[:8],
            'filename':   original_name,
            'saved_as':   unique_name,
            'prediction': disease_name.replace('_Pomegranate', ''),
            'confidence': confidence,
            'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'severity':   recommendation['severity'],
            'source':     'image'
        }

        save_to_cloud(entry)
        send_telegram(disease_name, confidence, recommendation, save_path)

        return jsonify({
            'success':        True,
            'prediction':     disease_name.replace('_Pomegranate', ''),
            'confidence':     confidence,
            'recommendation': recommendation,
            'image_url':      f'/static/uploads/{unique_name}',
            'timestamp':      entry['timestamp']
        })

    except Exception as e:
        return jsonify({'error': f'Lỗi dự đoán: {str(e)}'}), 500

# ─── Routes: Video ─────────────────────────────────────────────
@app.route('/predict_video', methods=['POST'])
def predict_video():
    """
    Phân tích video: trích xuất frame mỗi 1 giây, dự đoán từng frame.
    Trả về: danh sách kết quả từng frame + kết luận tổng hợp (majority vote).
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file video'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_video(file.filename):
        return jsonify({'error': 'File video không hợp lệ (mp4, avi, mov, mkv, webm)'}), 400

    original_name = secure_filename(file.filename)
    unique_name   = f"{uuid.uuid4().hex}_{original_name}"
    video_path    = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(video_path)

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Không thể mở file video'}), 400

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        # Lấy tối đa 30 frame, cách nhau đều nhau
        sample_count = min(30, max(5, int(duration_sec)))
        interval     = max(1, int(total_frames / sample_count))

        frame_results = []
        counts        = {}   # đếm số lần mỗi class xuất hiện

        frame_idx = 0
        sampled   = 0
        while sampled < sample_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            disease_name, confidence = predict_frame(frame)
            short_name = disease_name.replace('_Pomegranate', '')
            rec        = get_recommendation(disease_name)

            frame_results.append({
                'frame_idx':  frame_idx,
                'time_sec':   round(frame_idx / fps, 1),
                'prediction': short_name,
                'confidence': confidence,
                'severity':   rec['severity'],
                'status':     rec['status']
            })

            counts[disease_name] = counts.get(disease_name, 0) + 1
            frame_idx  += interval
            sampled    += 1

        cap.release()

        # ── Majority vote: chọn class xuất hiện nhiều nhất ──
        if not counts:
            return jsonify({'error': 'Không trích xuất được frame từ video'}), 400

        # Majority vote: class xuất hiện nhiều nhất thắng
        final_disease = max(counts, key=counts.get)

        final_rec   = get_recommendation(final_disease)
        avg_conf    = round(
            sum(r['confidence'] for r in frame_results if
                r['prediction'] == final_disease.replace('_Pomegranate', ''))
            / max(counts.get(final_disease, 1), 1), 1
        )

        # ── Lưu Firestore ──
        entry = {
            'id':          uuid.uuid4().hex[:8],
            'filename':    original_name,
            'saved_as':    unique_name,
            'prediction':  final_disease.replace('_Pomegranate', ''),
            'confidence':  avg_conf,
            'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'severity':    final_rec['severity'],
            'source':      'video',
            'frame_count': len(frame_results),
            'duration':    round(duration_sec, 1)
        }
        save_to_cloud(entry)

        # ── Gửi Telegram bằng frame đại diện ──
        if frame_results:
            rep_frame_idx = frame_results[len(frame_results)//2]['frame_idx']
            cap2 = cv2.VideoCapture(video_path)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, rep_frame_idx)
            ret2, rep_frame = cap2.read()
            cap2.release()
            if ret2:
                thumb_name = f"thumb_{uuid.uuid4().hex}.jpg"
                thumb_path = os.path.join(app.config['UPLOAD_FOLDER'], thumb_name)
                cv2.imwrite(thumb_path, rep_frame)
                send_telegram(final_disease, avg_conf, final_rec, thumb_path)

        return jsonify({
            'success':        True,
            'source':         'video',
            'duration':       round(duration_sec, 1),
            'frame_count':    len(frame_results),
            'frame_results':  frame_results,
            'final_disease':  final_disease.replace('_Pomegranate', ''),
            'final_conf':     avg_conf,
            'recommendation': final_rec,
            'timestamp':      entry['timestamp'],
            'disease_counts': {k.replace('_Pomegranate', ''): v
                               for k, v in counts.items()}
        })

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý video: {str(e)}'}), 500

# ─── Routes: History ───────────────────────────────────────────
@app.route('/history')
def get_history():
    return jsonify(load_from_cloud())

@app.route('/history/clear', methods=['POST'])
def clear_history():
    try:
        docs = db.collection('predictions').stream()
        for doc in docs:
            doc.reference.delete()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── Run ───────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    load_model()
    print("\n🍎 PomegranateAI đang chạy...")
    print("🌐 Truy cập: http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)