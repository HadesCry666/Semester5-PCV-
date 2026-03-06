# =====================================================
# 0. IMPORT LIBRARY
# =====================================================
import cv2
import numpy as np
import time
import os
import requests
import joblib
import threading
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.models import load_model
import serial 
from flask import Flask, Response
import threading

app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()


# =====================================================
# 1. PATH MODEL, SCALER, LABEL ENCODER
#    (JALANKAN DI PC/LAPTOP, BUKAN COLAB)
# =====================================================

# GANTI path ini sesuai lokasi file di komputer kamu
model_path  = r"C:\laragon\www\pcv\modelfix\model.h5"
scaler_path = r"C:\laragon\www\pcv\modelfix\scaler_tomat_backpro.pkl"
le_path     = r"C:\laragon\www\pcv\modelfix\labelencoder_tomat_backpro.pkl"

print("Memuat model & scaler ...")
model  = load_model(model_path)
scaler = joblib.load(scaler_path)
le     = joblib.load(le_path)
print("Model, scaler, dan label encoder berhasil di-load.\n")


# =====================================================
# 2. SEGMENTASI + CROP + RESIZE 300x300
# =====================================================
def segment_and_crop_bgr(img_bgr, resize_dim=(300, 300)):
    """
    return: img_asli, mask_full, crop_resized
    - mask_full : siluet tomat (putih = tomat, hitam = background)
    - crop_resized : gambar 300x300, background sudah hitam polos, pantulan tetap
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ==== RANGE WARNA TOMAT (tune sesuai dataset-mu) ====
    # Merah
    low1 = np.array([0,   80, 40])
    up1  = np.array([10, 255, 255])
    # Merah ujung
    low2 = np.array([170, 80, 40])
    up2  = np.array([180, 255, 255])
    # Oranye kemerahan (sangat jenuh)
    low3 = np.array([10, 140, 40])
    up3  = np.array([25, 255, 255])

    mask1 = cv2.inRange(hsv, low1, up1)
    mask2 = cv2.inRange(hsv, low2, up2)
    mask3 = cv2.inRange(hsv, low3, up3)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    # Bersihkan noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Cari kontur
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape

    if len(cnts) == 0:
        crop = img_bgr.copy()
        mask_full = np.zeros((H, W), dtype=np.uint8)
    else:
        img_area = H * W
        min_area = 0.01 * img_area  # min 1% area gambar

        big_cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]

        if len(big_cnts) == 0:
            crop = img_bgr.copy()
            mask_full = np.zeros((H, W), dtype=np.uint8)
        else:
            # 1) gabung semua kontur besar jadi mask_full
            mask_full = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(mask_full, big_cnts, -1, 255, thickness=-1)

            # 1b) rapikan dengan convex hull
            cnts_full, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            hull_mask = np.zeros_like(mask_full)
            for c in cnts_full:
                hull = cv2.convexHull(c)
                cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
            mask_full = hull_mask

            # 2) bounding box dari mask_full
            ys, xs = np.where(mask_full > 0)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()

            w = x1 - x0
            h = y1 - y0
            pad = int(0.05 * max(w, h))  # 5% padding

            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(W, x1 + pad)
            y1 = min(H, y1 + pad)

            # 3) hapus background di luar tomat
            img_clean = img_bgr.copy()
            img_clean[mask_full == 0] = (0, 0, 0)
            crop = img_clean[y0:y1, x0:x1]

    crop_resized = cv2.resize(crop, resize_dim)
    return img_bgr, mask_full, crop_resized


# =====================================================
# 3. MASK DARI HASIL CROP
# =====================================================
def get_mask_from_crop(crop_bgr):
    mask = np.any(crop_bgr > 0, axis=2).astype(np.uint8) * 255
    return mask


# =====================================================
# 4. FITUR WARNA (RGB + HSV) DARI CROP
# =====================================================
def extract_color_features_from_crop(crop_bgr):
    mask = get_mask_from_crop(crop_bgr)
    idx = mask > 0
    if np.count_nonzero(idx) == 0:
        idx = np.ones(mask.shape, dtype=bool)

    # RGB
    B, G, R = cv2.split(crop_bgr)
    mean_R = np.mean(R[idx])
    mean_G = np.mean(G[idx])
    mean_B = np.mean(B[idx])

    # HSV
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    mean_H = np.mean(H[idx])
    mean_S = np.mean(S[idx])
    mean_V = np.mean(V[idx])

    return {
        "Mean_R": mean_R,
        "Mean_G": mean_G,
        "Mean_B": mean_B,
        "Mean_H": mean_H,
        "Mean_S": mean_S,
        "Mean_V": mean_V
    }


# =====================================================
# 5. FITUR GLCM DARI GRAY-SCALE CROP (LAPLACIAN)
# =====================================================
def extract_glcm_from_crop(crop_bgr):
    mask = get_mask_from_crop(crop_bgr)
    if np.count_nonzero(mask) == 0:
        mask[:] = 255

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    gray_roi = gray[y0:y1+1, x0:x1+1]

    gray_roi = cv2.normalize(gray_roi, None, 0, 255,
                             cv2.NORM_MINMAX).astype('uint8')

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_roi = clahe.apply(gray_roi)

    # LAPLACIAN
    edges = cv2.Laplacian(gray_roi, cv2.CV_64F)
    edges = np.uint8(np.absolute(edges))

    # quantize 0..255 -> 0..7
    gray_q = (edges / 32).astype(np.uint8)

    glcm = graycomatrix(
        gray_q,
        distances=[1, 2, 4],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=8,
        symmetric=True,
        normed=True
    )

    contrast    = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy      = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    return {
        "GLCM_Contrast": contrast,
        "GLCM_Correlation": correlation,
        "GLCM_Energy": energy,
        "GLCM_Homogeneity": homogeneity
    }


# =====================================================
# 6. DAFTAR FITUR (HARUS SAMA DENGAN SAAT TRAINING)
# =====================================================
feature_cols = [
    "Mean_R",
    "Mean_G",
    "Mean_B",
    "Mean_H",
    "Mean_S",
    "Mean_V",
    "GLCM_Contrast",
    "GLCM_Correlation",
    "GLCM_Energy",
    "GLCM_Homogeneity"
]


# =====================================================
# 7. PREDIKSI DARI CROP (BGR)
# =====================================================
def predict_from_crop(crop_bgr):
    color_feat = extract_color_features_from_crop(crop_bgr)
    glcm_feat  = extract_glcm_from_crop(crop_bgr)
    feats = {**color_feat, **glcm_feat}

    X_raw = np.array([[feats[col] for col in feature_cols]], dtype=np.float32)
    X_scaled = scaler.transform(X_raw)

    prob = model.predict(X_scaled)[0]
    idx  = np.argmax(prob)
    label_pred = le.inverse_transform([idx])[0]

    confidence_percent = float(prob[idx] * 100.0)
    prob_percent_vec   = prob * 100.0

    return label_pred, confidence_percent, prob_percent_vec


# =====================================================
# 8. KOMUNIKASI DENGAN ESP32 VIA HTTP
# =====================================================

# GANTI IP ini dengan yang muncul di Serial Monitor:
# "IP ESP32: 192.168.x.x"
ESP32_BASE_URL = "http://10.200.27.114/"   # tanpa "/" di belakang


def send_grade_to_esp32_from_label(grade_label: str):
    """
    Terima label model ('Grade A', 'Grade B', 'Grade C' atau 'A'/'B'/'C'),
    ambil huruf A/B/C, lalu kirim ke ESP32 lewat HTTP: /sortir?grade=A/B/C
    """
    # Rapikan & kapital
    text = grade_label.strip().upper()   # contoh: "GRADE C"

    grade_char = None

    # Kalau format "GRADE A", "GRADE B", "GRADE C"
    if text.startswith("GRADE "):
        grade_char = text.split()[-1]    # ambil kata terakhir: "A"/"B"/"C"
    # Kalau label cuma "A"/"B"/"C"
    elif text in ["A", "B", "C"]:
        grade_char = text

    if grade_char not in ["A", "B", "C"]:
        print("Tidak bisa mapping grade dari label:", grade_label)
        return

    try:
        url = ESP32_BASE_URL + "/sortir"
        params = {"grade": grade_char}
        print("Kirim perintah sortir ke ESP32:", url, params)
        resp = requests.get(url, params=params, timeout=5)
        print("Status sortir ESP32:", resp.status_code)
        print("Respon ESP32      :", resp.text[:200])
    except Exception as e:
        print("Gagal kirim grade ke ESP32:", e)


def get_berat_stabil_from_esp32():
    try:
        # PENTING: pakai /data
        r = requests.get(ESP32_BASE_URL + "/data", timeout=5)
        if r.status_code != 200:
            print("Gagal ambil data dari ESP32, status:", r.status_code)
            return -1.0

        data = r.json()
        # ambil berat_stabil dari JSON
        berat_stabil = float(data.get("berat_stabil", 0.0))
        return berat_stabil

    except Exception as e:
        print("Error HTTP ke ESP32:", e)
        return -1.0


# =====================================================
# 9. SETTING API LARAVEL
# =====================================================

LARAVEL_URL = "http://127.0.0.1:8000/api/sorting"


def send_to_laravel(grade_label, confidence, berat, crop_img):
    # ---------- encode crop ke JPEG ----------
    success, buffer = cv2.imencode(".jpg", crop_img)
    if not success:
        print("Gagal encode gambar ke JPEG")
        return

    img_bytes = buffer.tobytes()

    # nama file unik, pakai timestamp
    filename = time.strftime("tomat_%Y%m%d_%H%M%S.jpg")

    # field biasa (multipart)
    data = {
        "grade": grade_label,
        "akurasi": f"{confidence:.2f}",
        "bobot": f"{berat:.2f}"
    }

    # field file (multipart)
    files = {
        "image": (filename, img_bytes, "image/jpeg")
    }

    try:
        resp = requests.post(
            LARAVEL_URL,
            data=data,
            files=files,
            timeout=5
        )
        print("Status Laravel:", resp.status_code)
        print("Body   Laravel:", resp.text[:400])
    except Exception as e:
        print("Gagal kirim ke Laravel:", e)

# =====================================================
# 10. TEST 1 GAMBAR DARI DATASET (CEK SERVO + LARAVEL)
# =====================================================
def is_tomato_in_frame(frame_bgr, area_threshold_ratio=0.05):
    """
    Deteksi apakah ada tomat di dalam frame
    berdasarkan warna merah/oranye di area ROI.
    return: True jika luas mask >= area_threshold_ratio dari gambar.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Range warna tomat (sama seperti segment_and_crop_bgr)
    low1 = np.array([0,   80, 40])
    up1  = np.array([10, 255, 255])

    low2 = np.array([170, 80, 40])
    up2  = np.array([180, 255, 255])

    low3 = np.array([10, 140, 40])
    up3  = np.array([25, 255, 255])

    mask1 = cv2.inRange(hsv, low1, up1)
    mask2 = cv2.inRange(hsv, low2, up2)
    mask3 = cv2.inRange(hsv, low3, up3)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    # Bersihkan noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Hitung luas area tomat
    nonzero = np.count_nonzero(mask)
    H, W = mask.shape
    ratio = nonzero / float(H * W)

    # Debug:
    # print(f"Luas mask tomat: {ratio * 100:.2f}%")

    # Balikkan True jika luas > threshold
    return ratio >= area_threshold_ratio

def gen_frames():
    global latest_frame

    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue

            frame = latest_frame.copy()

        # ===== DRAW ROI BIAR SAMA DENGAN SORTING =====
        h, w = frame.shape[:2]
        roi_x0 = int(w * 0.34)
        roi_x1 = int(w * 0.80)
        roi_y0 = int(h * 0.35)
        roi_y1 = int(h * 0.75)

        cv2.rectangle(frame,
                      (roi_x0, roi_y0),
                      (roi_x1, roi_y1),
                      (0, 255, 0), 2)

        cv2.putText(frame,
                    "Live Kamera Tomat (Mirror Main.py)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

        time.sleep(0.03)  # 30 FPS stabil


@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
        # JALANKAN FLASK MIRROR DI THREAD TERPISAH
    flask_thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    )
    flask_thread.daemon = True
    flask_thread.start()

    print("Flask mirror aktif di http://127.0.0.1:5000/video_feed")

    CAMERA_INDEX = 1 # Iriun

    print("Membuka kamera ...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)


    # Biar halus, turunkan resolusi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Gagal membuka kamera. Coba ganti CAMERA_INDEX atau cek USB.")
        exit(1)

    print("=== MODE LIVE SORTIR TOMAT (PAKAI BOBOT, DELAY 5 DETIK) ===")
    print("Tekan 'q' untuk keluar.\n")

    ready_for_new_tomato = True   # supaya 1 tomat = 1x proses

    # ---- Anti duplikasi + stabilitas ----
    MIN_FRAMES_TOMATO = 5   # tomat harus terlihat 5 frame berturut-turut
    MIN_FRAMES_EMPTY  = 2   # tomat harus hilang 2 frame berturut-turut

    tomato_frames = 0
    empty_frames  = 0

    last_process_time = 0
    MIN_COOLDOWN = 0.5          # jeda minimal antar proses (cadangan)

    # ---- Delay setelah DETEKSI tomat (bukan setelah kosong) ----
    DELAY_SETELAH_DETEKSI = 5.0   # 5 detik
    pending_tomato = False
    pending_start_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal baca frame dari kamera.")
            continue   #  sekarang ini VALID karena di dalam while

        #  SIMPAN FRAME KE GLOBAL UNTUK STREAM FLASK
        with frame_lock:
            latest_frame = frame.copy()   #  HANYA BARIS INI di dalam with

        # ============================
        # SEMUA PROSES DI BAWAH INI SUDAH NORMAL
        # ============================
        h, w, _ = frame.shape
        roi_x0 = int(w * 0.34)
        roi_x1 = int(w * 0.80)
        roi_y0 = int(h * 0.35)
        roi_y1 = int(h * 0.75)

        # gambar kotak ROI
        cv2.rectangle(frame, (roi_x0, roi_y0), (roi_x1, roi_y1), (0, 255, 0), 2)

        cv2.putText(
            frame,
            "Live Sortir Tomat (anti duplikasi) - Tekan 'q' untuk keluar",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        # cv2.imshow("Preview Kamera", frame)

        # === DETEKSI TOMAT DARI ROI ===
        roi = frame[roi_y0:roi_y1, roi_x0:roi_x1].copy()
        tomato_present = is_tomato_in_frame(roi, area_threshold_ratio=0.05)


        # Update counter frame
        if tomato_present:
            tomato_frames += 1
            empty_frames = 0
        else:
            empty_frames += 1
            tomato_frames = 0

        now = time.time()

        # ---- Reset siap tomat baru kalau sudah kosong beberapa frame ----
        if empty_frames >= MIN_FRAMES_EMPTY and not ready_for_new_tomato:
            print("Tomat sudah tidak terlihat di kamera beberapa frame, siap tomat berikutnya.")
            ready_for_new_tomato = True
            # kalau kosong, batalin pending deteksi
            pending_tomato = False

        # ---- Logika delay 5 detik setelah tomat terdeteksi stabil ----
        if ready_for_new_tomato and tomato_frames >= MIN_FRAMES_TOMATO:
            if not pending_tomato:
                pending_tomato = True
                pending_start_time = now
                print("Tomat terdeteksi stabil. Mulai hitung delay 5 detik sebelum capture...")
        else:
            # kalau tomat hilang sebelum 5 detik, batal
            pending_tomato = False

        # ---- 2) Proses tomat hanya jika:
        # - ada pending_tomato
        # - delay 5 detik sudah lewat
        # - sistem siap tomat baru
        # - cooldown waktu terpenuhi
        if (
            pending_tomato and
            ready_for_new_tomato and
            (now - pending_start_time >= DELAY_SETELAH_DETEKSI) and
            (now - last_process_time >= MIN_COOLDOWN)
        ):
            print("\n=== TOMAT BARU TERDETEKSI STABIL + DELAY 5s ===")
            print("Ambil frame dari kamera untuk diproses ...")

            img = frame.copy()

            # 1) Segmentasi + crop 300x300
            _, mask_full, crop300 = segment_and_crop_bgr(
                img,
                resize_dim=(300, 300)
            )

            if np.count_nonzero(mask_full) == 0:
                print("Mask kosong (tomat tidak terdeteksi di gambar), "
                      "prediksi tetap dilanjutkan pakai crop.")

            # 2) Prediksi grade
            grade_label, conf_percent, prob_vec_percent = predict_from_crop(crop300)
            print("Prediksi Grade:", grade_label)
            print(f"Confidence   : {conf_percent:.2f}%")

            # 3) Baca berat sesaat dari ESP32 (sekali saja)
            berat = get_berat_stabil_from_esp32()
            if berat < 0:
                print("Berat tidak valid (ESP32 tidak terbaca), pakai 0.0 gram.")
                berat = 0.0

            print(f"Berat stabil (gram): {berat:.2f}")

            # 4) Kirim perintah sortir ke ESP32
            print(f"Kirim perintah sortir ke ESP32 berdasarkan label: {grade_label}")
            send_grade_to_esp32_from_label(grade_label)

            # 5) Kirim data ke Laravel
            print("Kirim data ke Laravel ...")
            send_to_laravel(grade_label, conf_percent, berat, crop300)

            print("=== PROSES TOMAT SELESAI ===\n")
            print("Tunggu tomat diangkat (keluar ROI) untuk memproses tomat berikutnya ...")

            ready_for_new_tomato = False
            last_process_time = now
            pending_tomato = False  # selesai proses, pending dibersihkan

        # Tombol keluar
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Keluar dari program.")
            break

    cap.release()
    cv2.destroyAllWindows()