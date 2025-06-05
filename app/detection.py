# app/detection.py – versão turbo 🏎️
import os, time, logging, cv2, numpy as np, torch
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

from app.millis_call import make_emergency_call
from app.telegram_alert import send_telegram_video
from app.config import MODEL1_PATH, MODEL2_PATH, MODEL3_PATH

# ------------------------------------------------------------------
# Configuração global
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF   = DEVICE == "cuda"

FRAME_FPS   = 30     # ajuste conforme fonte
BUFFER_SEC  = 5
BUFFER_SECONDS = BUFFER_SEC        # compat
BUFFER_LEN  = FRAME_FPS * BUFFER_SEC

# Turbo‑parâmetros
INFER_SIZE  = 640                  # lado maior após resize
BATCH_SIZE  = 1                    # frames por inferência

HYPER = dict(
    mild_threshold          = 0.80,
    detection_count_thresh  = 20,
    mild_consecutive_thresh = 5,
)

# ------------------------------------------------------------------
# Carregamento dos modelos
# ------------------------------------------------------------------
def _load_model(path: str) -> YOLO:
    m = YOLO(path).to(DEVICE)
    if HALF:
        m.half()
    m.fuse()
    m.eval()
    logging.info("Modelo %s pronto (%s)", os.path.basename(path), DEVICE)
    return m

MODEL1, MODEL2, MODEL3 = map(_load_model, (MODEL1_PATH, MODEL2_PATH, MODEL3_PATH))

# ------------------------------------------------------------------
# Resize rápido mantendo proporção
# ------------------------------------------------------------------
def _resize_keep_ratio(img: np.ndarray, max_side: int = INFER_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:  # só encolhe
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return img

# ------------------------------------------------------------------
# Inferência em lote (só MODEL1)
# ------------------------------------------------------------------
_frame_batch: list[np.ndarray] = []
_pending_results: deque = deque()

@torch.inference_mode()
def _infer_model1(frames_bgr: list[np.ndarray]) -> list[list[dict]]:
    """Roda MODEL1 em lote, devolvendo detecções por frame."""
    if not frames_bgr:
        return []

    imgs = [_resize_keep_ratio(f)[..., ::-1] for f in frames_bgr]  # BGR→RGB
    preds = MODEL1.predict(imgs, device=DEVICE, imgsz=INFER_SIZE, half=HALF, verbose=False)

    out = []
    for r in preds:
        if not len(r.boxes):
            out.append([])
            continue
        b = r.boxes
        boxes = b.xyxy.cpu().int().tolist()
        confs = b.conf.cpu().tolist()
        clss  = b.cls.cpu().int().tolist()
        out.append(
            [
                {
                    "confidence": c,
                    "box": tuple(xyxy),
                    "class": int(cls),
                }
                for xyxy, c, cls in zip(boxes, confs, clss)
                if cls == 1  # só violência
            ]
        )
    return out

# ------------------------------------------------------------------
# API principal
# ------------------------------------------------------------------
def run_all_models(frame: np.ndarray) -> dict:
    """
    Acumula frames e executa MODEL1 em lote.
    Retorna detecções do frame atual.
    MODEL2 e MODEL3 deixam de ser chamados aqui para poupar GPU.
    """
    global _frame_batch, _pending_results

    _frame_batch.append(frame.copy())
    if len(_frame_batch) == BATCH_SIZE:
        _pending_results.extend(_infer_model1(_frame_batch))
        _frame_batch.clear()

    if not _pending_results:
        return {"model1": [], "model2": [], "model3": []}

    return {
        "model1": _pending_results.popleft(),
        "model2": [],
        "model3": [],
    }

# Funções avulsas caso queira rodar MODEL2/3 sob demanda
@torch.inference_mode()
def run_model2(frame: np.ndarray):
    return _run_generic(MODEL2, frame)

@torch.inference_mode()
def run_model3(frame: np.ndarray):
    return _run_generic(MODEL3, frame)

def _run_generic(model: YOLO, frame: np.ndarray):
    r = model(
        _resize_keep_ratio(frame)[..., ::-1],
        imgsz=INFER_SIZE,
        half=HALF,
        device=DEVICE,
        verbose=False,
    )[0]
    if not len(r.boxes):
        return []
    b = r.boxes
    return [
        {
            "confidence": float(c),
            "box": tuple(map(int, xyxy)),
            "class": int(cls),
        }
        for xyxy, c, cls in zip(
            b.xyxy.cpu().tolist(), b.conf.cpu().tolist(), b.cls.cpu().tolist()
        )
    ]

# ------------------------------------------------------------------
# Rastreador de severidade
# ------------------------------------------------------------------
class SeverityTracker:
    def __init__(self, window_sec: int):
        self.win         = window_sec
        self.dets        = deque(maxlen=HYPER["detection_count_thresh"] * 2)
        self.mild_streak = 0

    def add(self, conf: float):
        t = time.time()
        self.dets.append((t, conf))
        self._cleanup(t)

    def _cleanup(self, now: float):
        while self.dets and now - self.dets[0][0] > self.win:
            self.dets.popleft()

    def severity(self) -> dict:
        now = time.time()
        self._cleanup(now)
        count    = len(self.dets)
        max_conf = max((c for _, c in self.dets), default=0.0)

        level = "NONE"
        if count >= HYPER["detection_count_thresh"]:
            level = "HIGH" if max_conf >= HYPER["mild_threshold"] else "MILD"

        if level == "MILD":
            self.mild_streak += 1
            if self.mild_streak >= HYPER["mild_consecutive_thresh"]:
                level, self.mild_streak = "HIGH", 0
        elif level == "HIGH":
            self.mild_streak = 0
        else:
            self.mild_streak = 0

        return {"level": level, "count": count, "max_confidence": max_conf}

# ------------------------------------------------------------------
# Funções de alerta
# ------------------------------------------------------------------
def _enrich_message(base: str, extra: dict) -> str:
    parts = [base]
    for key, title in (("model2", "Lethal Objects"), ("model3", "Violence")):
        if extra.get(key):
            lines = [
                f" - conf={d['confidence']:.2f}, cls={d['class']}, box={d['box']}"
                for d in extra[key]
            ]
            parts.append(f"\n{title}:\n" + "\n".join(lines))
    return "".join(parts)

def process_alerts(saved_path: str, msg: str, extra: dict, do_call: bool):
    enriched = _enrich_message(msg, extra)
    meta = send_telegram_video(saved_path, enriched)
    if do_call and meta:
        torch.cuda.current_stream().synchronize()
        ThreadPoolExecutor(1).submit(make_emergency_call, meta)

def process_review_alert(saved_path: str, msg: str, extra: dict):
    send_telegram_video(saved_path, _enrich_message(msg, extra))

# ------------------------------------------------------------------
# Salvamento de vídeo em lote
# ------------------------------------------------------------------
def save_video_clip(buffer: deque, out_path: str, fps: int = FRAME_FPS):
    if not buffer:
        logging.warning("Buffer vazio, nada a salvar.")
        return None
    frames = np.stack(buffer)
    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    ok = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    logging.info("Vídeo %s", "salvo" if ok else "falhou")
    return out_path if ok else None