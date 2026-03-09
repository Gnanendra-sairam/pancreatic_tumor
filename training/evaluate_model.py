import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input   # ⬅ changed this line
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Match training/app: SavedModel directory
MODEL_PATH = os.path.join(BASE_DIR, "model", "pancreas_vgg_model")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Data generator for test set
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH)

# Predictions
y_prob = model.predict(test_gen)
y_pred = (y_prob > 0.5).astype("int32").ravel()
y_true = test_gen.classes

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

class_labels = list(test_gen.class_indices.keys())
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))
