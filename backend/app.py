from __future__ import annotations

import io
import json
import traceback
import uuid
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# ===============================
# KONFIGURASI PATH
# ===============================
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
UPLOAD_FOLDER: Final[Path] = BASE_DIR / "uploads"
MODEL_DIR: Final[Path] = BASE_DIR / "models"

ALLOWED_EXTENSIONS: Final[set[str]] = {"png", "jpg", "jpeg", "webp"}
MODEL_PATH: Final[Path] = MODEL_DIR / "emotion_model_best.keras"
CLASS_NAMES_PATH: Final[Path] = MODEL_DIR / "class_names.json"

UPLOAD_FOLDER.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ===============================
# INISIALISASI FLASK
# ===============================
app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
CORS(app)

# ===============================
# LOAD MODEL & LABEL
# ===============================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(f"File class_names.json tidak ditemukan: {CLASS_NAMES_PATH}")

model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    raw_class_names = json.load(f)


def normalize_class_names(data: object) -> list[str]:
    if isinstance(data, list):
        return [str(x) for x in data]

    if isinstance(data, dict):
        keys = list(data.keys())

        if all(str(k).isdigit() for k in keys):
            ordered = sorted(data.items(), key=lambda item: int(item[0]))
            return [str(v) for _, v in ordered]

        if all(isinstance(v, int) for v in data.values()):
            ordered = sorted(data.items(), key=lambda item: item[1])
            return [str(k) for k, _ in ordered]

    raise ValueError("Format class_names.json tidak valid.")


class_names: list[str] = normalize_class_names(raw_class_names)

EMOSI_ID = {
    "angry": "Marah",
    "disgust": "Jijik",
    "fear": "Takut",
    "happy": "Senang",
    "neutral": "Netral",
    "sad": "Sedih",
    "surprise": "Terkejut",
}

# Haar cascade bawaan OpenCV
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Model berhasil dimuat")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Label:", class_names)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_and_crop_face(image_bytes: bytes) -> np.ndarray:
    """
    Mengembalikan wajah hasil crop dalam bentuk grayscale.
    Jika wajah tidak ditemukan, pakai seluruh gambar grayscale sebagai fallback.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    # RGB -> BGR untuk OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )

    if len(faces) == 0:
        print("Wajah tidak terdeteksi, memakai seluruh gambar.")
        return gray

    # ambil wajah terbesar
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # tambah margin sedikit
    margin = int(0.15 * max(w, h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(gray.shape[1], x + w + margin)
    y2 = min(gray.shape[0], y + h + margin)

    face_crop = gray[y1:y2, x1:x2]
    print(f"Wajah terdeteksi di x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")
    return face_crop


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    face = detect_and_crop_face(image_bytes)
    face = cv2.resize(face, (48, 48))
    face = face.astype(np.float32) / 255.0

    # shape: (1, 48, 48, 1)
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "server aktif"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Gambar tidak ditemukan."}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Silakan pilih gambar."}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Format file tidak didukung."}), 400

        ext = Path(file.filename).suffix.lower()
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)

        print("Gambar diterima:", file.filename)

        with open(filepath, "rb") as f:
            image_bytes = f.read()

        processed = preprocess_image(image_bytes)
        print("Shape input ke model:", processed.shape)

        prediction = model.predict(processed, verbose=0)[0]

        if len(prediction) != len(class_names):
            return jsonify({
                "error": "Jumlah output model tidak sesuai dengan jumlah label."
            }), 500

        index = int(np.argmax(prediction))
        emosi_en = class_names[index]
        emosi = EMOSI_ID.get(emosi_en, emosi_en)
        confidence = float(prediction[index] * 100)

        detail = {
            EMOSI_ID.get(class_names[i], class_names[i]): round(float(score) * 100, 2)
            for i, score in enumerate(prediction)
        }

        return jsonify({
            "emotion": emosi,
            "emosi": emosi,
            "confidence": round(confidence, 2),
            "scores": detail,
            "detail": detail,
            "file": filename
        }), 200

    except UnidentifiedImageError:
        return jsonify({"error": "File bukan gambar yang valid."}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Ukuran file terlalu besar. Maksimal 5 MB."}), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint tidak ditemukan."}), 404


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Kesalahan internal server."}), 500


if __name__ == "__main__":
    app.run(debug=True)