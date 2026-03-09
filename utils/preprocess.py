import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

IMG_SIZE = (224, 224)

def load_and_preprocess_image_vgg(image_path):
    """
    Load image and preprocess it for VGG16.
    """
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = vgg_preprocess(img_array)
    return img_array

def load_and_preprocess_image_resnet(image_path):
    """
    Load image and preprocess it for ResNet50.
    """
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = resnet_preprocess(img_array)
    return img_array

# For backward compatibility (if anything still imports old name)
def load_and_preprocess_image(image_path):
    return load_and_preprocess_image_vgg(image_path)
