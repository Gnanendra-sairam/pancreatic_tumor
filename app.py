import os
import time

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from keras.models import load_model
from utils.preprocess import load_and_preprocess_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# SavedModel directory (matches training)
MODEL_PATH = os.path.join(BASE_DIR, "model", "pancreas_vgg_model")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

print("Loading model from:", MODEL_PATH)
print("Path exists:", os.path.exists(MODEL_PATH))

# Load model once at startup
model = load_model(MODEL_PATH)

# 0 -> Normal, 1 -> Tumor
CLASS_LABELS = {
    0: "Normal",
    1: "Tumor",
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    if file:
        # Make filename safe and unique
        filename = secure_filename(file.filename)
        filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Preprocess and predict
        img_array = load_and_preprocess_image(filepath)
        raw = model.predict(img_array)[0][0]
        print("Raw model output:", raw)  # debug in terminal

        prob = float(raw)
        label = 1 if prob > 0.5 else 0
        prediction_text = CLASS_LABELS[label]

        image_url = url_for("static", filename=f"uploads/{filename}")

        return render_template(
            "result.html",
            prediction=prediction_text,
            probability=prob,
            image_url=image_url,
        )

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
