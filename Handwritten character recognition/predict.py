import numpy as np
import tensorflow as tf
from PIL import Image

# load model
model = tf.keras.models.load_model("model/model.h5")

def predict_character(image_path):

    # load image
    img = Image.open(image_path).convert("L")

    # resize to MNIST format
    img = img.resize((28, 28))

    # convert to numpy
    img = np.array(img).astype("float32")

    # normalize ONLY (IMPORTANT FIX)
    img = img / 255.0

    # reshape for CNN
    img = img.reshape(1, 28, 28, 1)

    # predict
    prediction = model.predict(img)

    result = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    return result, confidence
