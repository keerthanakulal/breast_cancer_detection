# 🩺 OncoScan — AI Breast Cancer Detection

A deep learning web application that classifies breast ultrasound images as **Benign**, **Malignant**, or **Normal** using VGG16 transfer learning, deployed with Flask.

---

## 📸 Preview

| Home Page | Detection Page | About Page |
|-----------|---------------|------------|
| Hero section with model overview | Drag & drop image upload + results | Architecture, dataset, tech stack |

---

## 🧠 Models Compared

Three models were trained and evaluated on breast ultrasound images:

| Model | Architecture | Accuracy |
|-------|-------------|----------|
| MLP | 6 Dense layers (256→8→3) | ~70% |
| **VGG16** ⭐ | Conv2D → VGG16 → Dense(256) → Softmax | **Best** |
| ResNet50 | Conv2D → ResNet50 → Dense(256) → Softmax | ~85% |

> VGG16 was selected as the final deployed model.

---

## 📁 Project Structure

```
OncoScan/
├── app.py                          # Flask backend
├── final_vgg16_weights.weights.h5  # Trained model weights
├── compare_model.ipynb             # Training & evaluation notebook
├── templates/
│   ├── home.html                   # Landing page
│   ├── index.html                  # Detection page
│   └── about.html                  # About & architecture page
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

- **Source:** Breast Ultrasound Images Dataset
- **Classes:** Benign, Malignant, Normal
- **Images per class:** 1,000
- **Total:** 3,000 images
- **Split:** 80% train / 20% test
- **Input size:** 300 × 300 px (grayscale)

---

## 🏗️ Model Architecture (VGG16)

```
Input (300×300×1 grayscale)
    ↓
Conv2D(3, 3×3, padding='same')        ← grayscale → 3-channel
    ↓
VGG16 Base (frozen, ImageNet weights)
    ↓
Flatten()
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(3, activation='softmax')        ← benign / malignant / normal
```

**Compiled with:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/oncoscan.git
cd oncoscan
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add model weights

Place the trained weights file in the project root:
```
final_vgg16_weights.weights.h5
```

> ⚠️ The weights file is not included in the repository due to file size. Train the model using `compare_model.ipynb` or download from the release.

### 4. Run the Flask app

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000/**

---

## 🌐 Web App Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Home | Project overview, how it works, model comparison |
| `/detect` | Detection | Upload ultrasound image, get AI prediction |
| `/about` | About | Dataset, architecture, tech stack, limitations |
| `/predict` | API (POST) | JSON endpoint for image classification |

---

## 🔌 API Usage

**POST** `/predict`

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@your_image.jpg"
```

**Response:**

```json
{
  "predicted_class": "benign",
  "confidence": 91.34,
  "probabilities": {
    "benign": 91.34,
    "malignant": 5.12,
    "normal": 3.54
  },
  "description": "A non-cancerous tumor was detected. Benign tumors do not invade nearby tissue or spread. Please consult your doctor for further evaluation."
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow / Keras |
| Pretrained Model | VGG16 (ImageNet) |
| Image Processing | OpenCV |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Training Environment | Google Colab |
| Model Storage | Google Drive |

---

## 📦 Requirements

```
tensorflow
flask
numpy
opencv-python
```

> See `requirements.txt` for exact versions.

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is **not** a certified medical device and should never replace professional medical diagnosis. Always consult a qualified medical professional for any health concerns.

---

## 📓 Training

To retrain the model, open `compare_model.ipynb` in Google Colab:

1. Mount Google Drive
2. Point `folder_name` to your dataset directory
3. Run all cells — models are saved after each training
4. If Colab disconnects, use the **Recovery Cell** to reload models without retraining
5. Download `final_vgg16_weights.weights.h5` and place in project root

---

## 👩‍💻 Author

**Keerthana** — Final Year Project  
Breast Cancer Classification using Deep Learning

---

## 📄 License

This project is for academic purposes. Feel free to use and adapt with attribution.
