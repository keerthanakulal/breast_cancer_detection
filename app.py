from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, Flatten, Dense, Dropout

app = Flask(__name__)

IMG_SIZE = 300

def build_vgg16_model():
    base_model = VGG16(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    model = Sequential([
        InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        Conv2D(3, (3, 3), padding='same'),
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_vgg16_model()
model.load_weights('final_vgg16_weights.weights.h5')
print("Model loaded successfully!")

CATEGORIES = ["benign", "malignant", "normal"]

DESCRIPTIONS = {
    "benign": "A non-cancerous tumor was detected. Benign tumors do not invade nearby tissue or spread. Please consult your doctor for further evaluation.",
    "malignant": "A potentially cancerous tumor was detected. Malignant tumors can invade nearby tissue. Immediate medical consultation is strongly recommended.",
    "normal": "No tumor was detected. The tissue appears normal. Regular check-ups are still recommended."
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/detect")
def detect():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)

    prediction = model.predict(img_input)
    predicted_class = CATEGORIES[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    probabilities = {
        cat: round(float(prob) * 100, 2)
        for cat, prob in zip(CATEGORIES, prediction[0])
    }

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "description": DESCRIPTIONS[predicted_class]
    })

if __name__ == "__main__":
    app.run(debug=True)