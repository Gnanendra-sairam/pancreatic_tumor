import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_DIR = os.path.join(DATA_DIR, "test")

VGG_MODEL_PATH = os.path.join(BASE_DIR, "model", "pancreas_vgg_model")
RESNET_MODEL_PATH = os.path.join(BASE_DIR, "model", "pancreas_resnet_model")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# =========================
# VGG Test Generator
# =========================
vgg_test_datagen = ImageDataGenerator(
    preprocessing_function=vgg_preprocess
)

vgg_test_gen = vgg_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

# =========================
# ResNet Test Generator
# =========================
resnet_test_datagen = ImageDataGenerator(
    preprocessing_function=resnet_preprocess
)

resnet_test_gen = resnet_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

print("Loading VGG16 model from:", VGG_MODEL_PATH)
vgg_model = load_model(VGG_MODEL_PATH)

print("Loading ResNet50 model from:", RESNET_MODEL_PATH)
resnet_model = load_model(RESNET_MODEL_PATH)

# =========================
# Evaluate VGG
# =========================
print("\n========== VGG16 EVALUATION ==========")
vgg_probs = vgg_model.predict(vgg_test_gen)
vgg_pred = (vgg_probs > 0.5).astype("int32").ravel()
y_true = vgg_test_gen.classes
class_labels = list(vgg_test_gen.class_indices.keys())

print("Confusion Matrix (VGG16):")
print(confusion_matrix(y_true, vgg_pred))

print("\nClassification Report (VGG16):")
print(classification_report(y_true, vgg_pred, target_names=class_labels))

# =========================
# Evaluate ResNet
# =========================
print("\n========== RESNET50 EVALUATION ==========")
resnet_probs = resnet_model.predict(resnet_test_gen)
resnet_pred = (resnet_probs > 0.5).astype("int32").ravel()

print("Confusion Matrix (ResNet50):")
print(confusion_matrix(y_true, resnet_pred))

print("\nClassification Report (ResNet50):")
print(classification_report(y_true, resnet_pred, target_names=class_labels))
