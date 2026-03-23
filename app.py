from flask import Flask, render_template, Response, jsonify
from drowsiness_detector import DrowsinessDetector
import cv2

app = Flask(__name__)
detector = DrowsinessDetector()

def gen_frames():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = camera.read()
        if not success:
            break
        frame, _ = detector.detect_drowsiness(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify(detector.get_status())


@app.route("/status")
def status():
    s = detector.get_status()
    return jsonify({
        "status":       s["overall"],
        "headPosition": s["head_direction"],
        "framesDrowsy": s["counter"],
        "totalAlerts":  s["total_alerts"],
        "yawningFrames": s["yawning"] and s["total_alerts"] or 0
    })


if __name__ == '__main__':
    print("🚀 Open your browser and go to http://127.0.0.1:5000/")
    app.run(debug=True)
