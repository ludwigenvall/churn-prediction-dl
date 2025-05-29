# scripts/lstm_model.py

#Importing library
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Function to combine the 3 sequence columns into a 3D array for the LSTM model
# (n_samples, 30 timesteps, 3 features)
def prepare_lstm_data(df):
    sequences = np.stack([
        df['logins_seq'].to_list(),
        df['support_seq'].to_list(),
        df['data_seq'].to_list()
    ], axis=-1)
    
    # Encodes "Churn" as 0=No, 1=Yes
    le = LabelEncoder()
    labels = le.fit_transform(df['Churn'])

    # Returns a split into training/test datasets
    return train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Function to build baseline model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
 
    return model

# Function to compile model
def compile_lstm_model(model):
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   return model