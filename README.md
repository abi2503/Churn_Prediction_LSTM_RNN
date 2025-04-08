
# LSTM-RNN Customer Churn Prediction

## 🚀 Project Overview
This project builds an end-to-end deep learning pipeline to predict customer churn using **LSTM Recurrent Neural Networks (RNNs)** on time-series behavioral data. It includes:

- 🧠 Customer_Data(15,000 customers, 12 months each)
- 🔍 Realistic simulation of banking features (balance, activity, product usage)
- 🧼 Preprocessing with normalization and train/test split
- 🤖 LSTM-RNN model using TensorFlow/Keras
- 📈 Evaluation using accuracy, F1-score, precision, recall
- 🌐 Optional Streamlit deployment for interactive churn prediction
- 📦 Complete GitHub structure and documentation

## 📁 Dataset
The dataset contains customer records over 12 months with:
- Customer demographics (credit score, age, gender)
- Monthly behavior (salary, balance, activity, product usage)
- Binary churn label (1 = churned, 0 = retained)

## 📦 Project Structure
```
churn-lstm-rnn/
├── data/
│   └── Churn_Data.csv
├── notebooks/
│   └── 01_preprocessing.ipynb
│   └── 02_train_lstm_rnn.ipynb
├── src/
│   └── preprocess.py
│   └── model.py
├── models/
│   └── lstm_rnn_model.h5
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## ⚙️ Tech Stack
- Python, Pandas, NumPy
- TensorFlow/Keras
- Scikit-learn
- Matplotlib & Seaborn
- Streamlit (for optional deployment)

## ✨ Author
Abhishek Suresh  
M.S. in Information Systems, NYU  
Passionate about AI, LLMs, and real-world ML applications

---
> 📌 *This project is built as part of a portfolio to demonstrate deep learning skills and real-world deployment workflows.*
