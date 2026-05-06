# 🌿 Plant Disease Detection

A simple AI-powered web app that detects plant leaf diseases from images using deep learning.

---

## 🚀 Features

* Upload a leaf image and get instant predictions
* Identifies plant type and disease
* Displays prediction confidence
* Clean and interactive UI built with Streamlit

---

## 🧠 Model

* Base model: **MobileNetV2 (Transfer Learning)**
* Input size: **224 × 224**
* Two-phase training:

  * Feature extraction (frozen base model)
  * Fine-tuning (top layers unfrozen)
* Data augmentation used to improve generalization

---

## 📁 Project Structure

```
├── app.py                 # Streamlit web app
├── train_model.py         # Model training script
├── model/
│   └── plant_disease_model.keras
├── dataset_split/
│   ├── train/
│   └── validation/
├── class_names.json       # Label mapping
```

---

## ⚙️ Installation

```bash
git clone https://github.com/srijan-trm/LeafDetectionModel
cd LeafDetectionModel

pip install -r requirements.txt
```

---

## 🏋️‍♂️ Train the Model

```bash
python train_model.py
```

## ▶️ Run the App

```bash
streamlit run app.py
```

---


---

## 📸 Usage

1. Open the web app
2. Upload a leaf image (JPG/PNG)
3. View prediction results with confidence score

---

## ⚠️ Notes

* Works best with clear, well-lit leaf images
* Low confidence predictions may be inaccurate

---

## 📌 Future Improvements

* Add more plant species and diseases
* Improve model accuracy
* Deploy online (Streamlit Cloud / AWS / GCP)
* Add treatment suggestions

---

## 📜 License

This project is open-source and available under the MIT License.
