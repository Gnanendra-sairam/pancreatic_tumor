import os
import time
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
from utils.preprocess import (
    load_and_preprocess_image_vgg,
    load_and_preprocess_image_resnet,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

VGG_MODEL_PATH = os.path.join(BASE_DIR, "model", "pancreas_vgg_model")
RESNET_MODEL_PATH = os.path.join(BASE_DIR, "model", "pancreas_resnet_model")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

print("Loading VGG16 model from:", VGG_MODEL_PATH)
vgg_model = load_model(VGG_MODEL_PATH)

print("Loading ResNet50 model from:", RESNET_MODEL_PATH)
resnet_model = load_model(RESNET_MODEL_PATH)

# 0 -> Normal, 1 -> Tumor
CLASS_LABELS = {
    0: "Normal",
    1: "Tumor"
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
        filename = secure_filename(file.filename)
        filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Preprocess for both models
        img_vgg = load_and_preprocess_image_vgg(filepath)
        img_resnet = load_and_preprocess_image_resnet(filepath)

        # Predictions
        prob_vgg = float(vgg_model.predict(img_vgg)[0][0])
        prob_resnet = float(resnet_model.predict(img_resnet)[0][0])

        label_vgg = 1 if prob_vgg > 0.5 else 0
        label_resnet = 1 if prob_resnet > 0.5 else 0

        pred_text_vgg = CLASS_LABELS[label_vgg]
        pred_text_resnet = CLASS_LABELS[label_resnet]

        image_url = url_for("static", filename=f"uploads/{filename}")

        return render_template(
            "result.html",
            image_url=image_url,
            prob_vgg=prob_vgg,
            prob_resnet=prob_resnet,
            pred_vgg=pred_text_vgg,
            pred_resnet=pred_text_resnet
        )

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
