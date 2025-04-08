import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_data(file_path):
    # Step 1: Load the data
    df = pd.read_csv(file_path)
    
    # Step 2: Preprocess the features
    features = ['credit_score', 'age', 'gender', 'salary', 'balance', 'is_active', 'products_used']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # Step 3: Prepare X and y
    grouped = df.groupby('customer_id')
    X = np.stack([group[features].values for _, group in grouped])  # Shape: (num_customers, 12, 7)
    y = df.groupby('customer_id')['churned'].first().values  # Labels: 0 or 1

    # Step 4: Save X and y as .npy files
    np.save('X_train.npy', X)  # Saving the features
    np.save('y_train.npy', y)  # Saving the labels

    return X, y

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train):
    model = build_model(X_train.shape[1:])
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
                        verbose=1)
    model.save('lstm_churn_model.keras')  # Save the trained model
    return model, history
