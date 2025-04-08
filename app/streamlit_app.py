import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained LSTM model
model = load_model('/Users/abhisheksuresh/Documents/Deep_Learning/Churn_Prediction_Model_LSTM_RNN/data/lstm_churn_model.keras')

# Function to preprocess input data
def preprocess_data(df):
    features = ['credit_score', 'age', 'gender', 'salary', 'balance', 'is_active', 'products_used']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    grouped = df.groupby('customer_id')
    X = np.stack([group[features].values for _, group in grouped])
    
    return X

# Title of the app
st.title("Customer Churn Prediction")

# Upload CSV file
st.sidebar.header("Upload Customer Data")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a dataframe
    df = pd.read_csv(uploaded_file)
    
    # Preprocess the data
    X = preprocess_data(df)

    # Make predictions
    predictions = model.predict(X)
    churn_probabilities = predictions.flatten()
    churn_predictions = (churn_probabilities > 0.5).astype(int)

    # Display the results
    st.write(f"Predicted churn probabilities: {churn_probabilities[:10]}")
    st.write(f"Predicted churn classes (0 = not churn, 1 = churn): {churn_predictions[:10]}")

    # Show the input data and the predictions
    st.subheader("Uploaded Customer Data (First 10 Rows)")
    st.write(df.head(10))

    # Display churn probability distribution
    st.subheader("Churn Probability Distribution")
    st.write(churn_probabilities)
    
    # Show confusion matrix or other metrics if desired
    # You can add additional evaluations here if necessary
else:
    st.write("Please upload a CSV file to get started.")
