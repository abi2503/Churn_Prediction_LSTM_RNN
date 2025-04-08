import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('/Users/abhisheksuresh/Documents/Deep_Learning/Churn_Prediction_Model_LSTM_RNN/data/lstm_churn_model.keras')

# Load the preprocessed test data
X_test = np.load('/Users/abhisheksuresh/Documents/Deep_Learning/Churn_Prediction_Model_LSTM_RNN/data/X_test.npy')
y_test = np.load('/Users/abhisheksuresh/Documents/Deep_Learning/Churn_Prediction_Model_LSTM_RNN/data/y_test.npy')

# Make a prediction for a random customer from the test set
random_idx = np.random.randint(0, X_test.shape[0])
sample_input = X_test[random_idx:random_idx + 1]
pred_prob = model.predict(sample_input)[0][0]
pred_class = int(pred_prob > 0.5)

print(f"Predicted Probability of Churn: {pred_prob:.4f}")
print(f"Predicted Class: {pred_class} | Actual: {y_test[random_idx]}")
