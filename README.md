# 🩺 Smart Disease Prediction System

A machine learning-based web application that predicts potential diseases based on user-provided symptoms. Built using Python, Scikit-learn, and Streamlit, the application provides an interactive interface for disease classification using a trained machine learning model.

## 🌐 Live Demo

**Try the application:** [Smart Disease Prediction](https://smart-disease-prediction-hhmounqgfvt5cpvgwncrae.streamlit.app/)

## ✨ Features

- **Disease Prediction:** Predicts potential diseases based on input symptoms.
- **Machine Learning:** Uses a trained model for disease classification.
- **Interactive Interface:** Simple and user-friendly Streamlit web application.
- **Data-Driven Approach:** Uses training and testing datasets for model development and evaluation.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Web Framework:** Streamlit

## 📁 Project Structure

```text
smart-disease-prediction/
├── app.py
├── requirements.txt
├── model_bundle.pkl
├── models/
│   └── model_bundle.pkl
├── Training.csv
├── Testing.csv
└── Smart_Disease_Prediction.ipynb
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Riyaa2025/smart-disease-prediction.git
cd smart-disease-prediction
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Dataset

The project includes:
- `Training.csv` — training dataset.
- `Testing.csv` — testing dataset.

## 🎯 Objective

The objective of this project is to explore the application of machine learning in symptom-based disease classification and build an accessible, interactive prediction interface.

## ⚠️ Disclaimer

This application is intended for educational and informational purposes only. Its predictions are not medical diagnoses and should not replace consultation with a qualified healthcare professional.

## 👩‍💻 Author

**Riya Setia**

- GitHub: [@Riyaa2025](https://github.com/Riyaa2025)
- LinkedIn: [Riya Setia](https://www.linkedin.com/in/riya-setia-806025343/)
