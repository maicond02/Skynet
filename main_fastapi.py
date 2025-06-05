# main_fastapi.py
import os, cv2, time, logging
from datetime import datetime
from collections import deque
from threading import Thread, Lock
from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.detection import (
    run_all_models,
    process_alerts,
    process_review_alert,
    save_video_clip,
    BUFFER_SECONDS,
    SeverityTracker,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_VIDEO_SOURCE = "videos/luta.mp4"
MAX_LOG_ENTRIES = 10

class AppState:
    def __init__(self):
        self.detection_status = {
            "level": "NONE",
            "max_confidence": 0.0,
            "detections": 0,
            "last_update": "",
            "alert": "",
            "logs": deque(maxlen=MAX_LOG_ENTRIES),
        }
        self.incident_history = []
        self.last_telegram_alert_time = 0
        self.last_emergency_call_time = 0
        self.detection_lock = Lock()
        self.incident_lock = Lock()
        self.settings = {
            "video_save_path": "output",
            "telegram_alert_interval": 10,
            "emergency_call_interval": 30,
        }
        os.makedirs(self.settings["video_save_path"], exist_ok=True)

app_state = AppState()
severity_tracker = SeverityTracker(5)

app = FastAPI()

# CORS p/ React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_alert_message(sev: str, conf: float, dets: int):
    now = datetime.now()
    base = (
        "🚨 Violent Activity Detected!\n"
        f"Date: {now:%Y-%m-%d}\n"
        f"Time: {now:%I:%M %p}\n"
        f"Severity: {sev}\n"
        f"Confidence: {conf:.2f}\n"
        f"Detections: {dets}"
    )
    log = f"{now:%H:%M:%S} - {sev} alert (Confidence: {conf:.2f}, Detections: {dets})"
    return base, log, now

def update_detection_status(sev: str, conf: float, dets: int, alert=""):
    with app_state.detection_lock:
        app_state.detection_status.update({
            "level": sev,
            "max_confidence": round(conf, 2),
            "detections": dets,
            "last_update": time.strftime("%H:%M:%S"),
            "alert": alert,
        })

def add_incident(sev: str, conf: float, dets: int, msg: str, dt: datetime):
    with app_state.incident_lock:
        app_state.incident_history.append({
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M:%S"),
            "severity": sev,
            "confidence": round(conf, 2),
            "detections": dets,
            "message": msg,
        })

def process_alert(severity, frame_buffer, max_conf, det_count, results, now, make_call=False):
    name = f"violent_clip_{int(now)}.mp4"
    path = os.path.join(app_state.settings["video_save_path"], name)
    saved_path = save_video_clip(frame_buffer, path, int(cap.get(cv2.CAP_PROP_FPS)) or 30)

    base_msg, log_entry, dt = generate_alert_message(severity, max_conf, det_count)

    with app_state.detection_lock:
        app_state.detection_status["logs"].append(log_entry)

    extra = {"model2": results.get("model2", []), "model3": results.get("model3", [])}

    if severity == "HIGH":
        fn = process_alerts
        alert_txt = "High alert triggered: Telegram alert sent" + (
            ", emergency call initiated." if make_call else "."
        )
    else:
        fn = process_review_alert
        alert_txt = "Mild alert triggered: Telegram review alert sent."

    Thread(target=fn, args=(saved_path, base_msg, extra, make_call)).start()
    add_incident(severity, max_conf, det_count, base_msg, dt)

    with app_state.detection_lock:
        app_state.detection_status["alert"] = alert_txt

    severity_tracker.dets.clear()

def detection_frame_generator():
    global cap
    cap = cv2.VideoCapture(DEFAULT_VIDEO_SOURCE)
    if not cap.isOpened():
        logging.error("Error: Cannot access video source.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_buffer = deque(maxlen=fps * (BUFFER_SECONDS * 2))

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            logging.error("Error: Cannot read frame.")
            break

        frame_buffer.append(frame.copy())
        now = time.time()

        results = run_all_models(frame)
        for det in results.get("model1", []):
            severity_tracker.add(det["confidence"])

        sev_info = severity_tracker.severity()
        update_detection_status(sev_info["level"], sev_info["max_confidence"], sev_info["count"])

        for det in results.get("model1", []):
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        overlay = (
            f"Severity: {sev_info['level']} | "
            f"Confidence: {sev_info['max_confidence']:.2f} | "
            f"Detections: {sev_info['count']} | "
            f"Last Update: {time.strftime('%H:%M:%S')}"
        )
        #cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        level = sev_info["level"]
        tel_ok = (now - app_state.last_telegram_alert_time) >= app_state.settings["telegram_alert_interval"]
        call_ok = level == "HIGH" and (now - app_state.last_emergency_call_time) >= app_state.settings["emergency_call_interval"]

        if level in ("HIGH", "MILD") and (tel_ok or call_ok):
            if tel_ok:
                app_state.last_telegram_alert_time = now
            if call_ok:
                app_state.last_emergency_call_time = now

            process_alert(level, frame_buffer, sev_info["max_confidence"], sev_info["count"], results, now, call_ok)
        else:
            with app_state.detection_lock:
                app_state.detection_status["alert"] = ""

        ret, buf = cv2.imencode(".jpg", frame)
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    cap.release()

# ---------------------- API ENDPOINTS -----------------------
@app.get("/status_view")
async def status_view():
    with app_state.detection_lock:
        return app_state.detection_status.copy()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(detection_frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/incidents")
def get_incidents():
    with app_state.incident_lock:
        return {"incidents": app_state.incident_history}

@app.get("/settings")
def get_settings():
    return app_state.settings

@app.post("/update_settings")
async def update_settings(
    telegram_alert_interval: int = Form(...),
    emergency_call_interval: int = Form(...),
    video_save_path: str = Form(...),
):
    if telegram_alert_interval < 1 or emergency_call_interval < 1:
        return JSONResponse({"error": "Interval must be >= 1"}, status_code=400)

    app_state.settings.update({
        "telegram_alert_interval": telegram_alert_interval,
        "emergency_call_interval": emergency_call_interval,
        "video_save_path": video_save_path,
    })
    os.makedirs(video_save_path, exist_ok=True)
    return {"success": True, "settings": app_state.settings}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_fastapi:app", host="0.0.0.0", port=8001, reload=True)