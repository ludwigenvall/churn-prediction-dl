# scripts/lstm_model.py

#Importing library
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Function to combine the 3 sequence columns into a tensor for the LSTM model
def prepare_lstm_data(df):
    # n_samples = number of customers, 30 timesteps, 3 features
    n_samples = len(df)
    n_timesteps = len(df.iloc[0]['logins_seq'])

    # Preallocate array
    sequences = np.zeros((n_samples, n_timesteps, 3))

    # Fill features
    sequences[:, :, 0] = np.array(df['logins_seq'].to_list())
    sequences[:, :, 1] = np.array(df['support_seq'].to_list())
    sequences[:, :, 2] = np.array(df['data_seq'].to_list())

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