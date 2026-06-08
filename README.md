# 🏠 House Price Prediction App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shivansh-mishra-bbd-house-price-prediction-group-project-indian.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange?logo=scikit-learn)](https://scikit-learn.org/)

An advanced, interactive web application that predicts house prices based on various property features. This project utilizes a highly accurate Machine Learning pipeline to provide precise, real-time real estate valuations.

## 🚀 Live Demo
**[Click Here to Use the Live App](https://shivansh-mishra-bbd-house-price-prediction-group-project-indian.streamlit.app/)**

---

## ✨ Key Features
- **Instant Predictions:** Get real-time house price estimates instantly via an interactive UI.
- **Advanced ML Pipeline:** Powered by a robust Scikit-Learn pipeline (`RandomForestRegressor`, `StandardScaler`, `OneHotEncoder`).
- **Premium Glassmorphic UI:** A beautifully responsive, animated front-end with floating cards, glowing buttons, and a dynamic mesh background.
- **Interactive Data Visualizations:** Deep dive into the market with dynamic **Altair** charts (Histograms, Heatmaps, Boxplots) in the new Analytics tab.
- **Real-Time Price Comparison:** Instantly compare your predicted home value against the local city average visually.

## 🛠️ Technology Stack
- **Frontend / Backend:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (Random Forest)

---

## 👨‍💻 Development Team
This is an academic Machine Learning project developed by students at **BBD University** (B.Tech CSE - Cloud Computing & ML):
- **Shivansh Mishra**
- **Ravi Gupta**
- **Shiwanshu Singh**
- **Harshvardhan Sisodiya**
- **Dhuru Madhuwal**
- **Vishal Patel**

---

## 💻 Local Installation

To run this project locally on your machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/Shivansh-mishraji/house-price-prediction-app.git
cd house-price-prediction-app
```

### 2. Install Dependencies
Make sure you have Python installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Streamlit server:
```bash
streamlit run app.py
```

### 4. (Optional) Retrain the Model
If you wish to re-train the machine learning model from scratch with the provided dataset:
```bash
python train.py
```
This will generate a fresh `model.pkl` file using the robust Random Forest pipeline.

---
*Built with ❤️ for predicting your dream home's value with precision.*
