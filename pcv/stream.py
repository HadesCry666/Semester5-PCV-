from flask import Flask, Response
import cv2
import time

app = Flask(__name__)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
time.sleep(2)

def gen_frames():
    while True:
        success, frame = cap.read()

        if not success or frame is None:
            print("⚠️ Frame kosong...")
            time.sleep(0.1)
            continue

        # ✅ HANYA TAMPILKAN TEKS SAJA (TANPA KOTAK HIJAU)
        cv2.putText(frame,
                    "Live Kamera Tomat",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0), 2)

        # Encode ke JPG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    print("✅ Stream aktif di http://127.0.0.1:5000/video_feed")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
