# scripts/lstm_model.py

#Importing library
import numpy as np
import tensorflow as tf
import ast
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to combine the 3 sequence columns into a tensor
def prepare_lstm_data(df):
    # Convert stringified lists with/without brackets to proper Python lists
    df['logins_seq'] = df['logins_seq'].apply(lambda x: list(map(int, ast.literal_eval(x))) if isinstance(x, str) else x)
    df['support_seq'] = df['support_seq'].apply(lambda x: list(map(int, ast.literal_eval(x))) if isinstance(x, str) else x)
    df['data_seq'] = df['data_seq'].apply(lambda x: list(map(float, ast.literal_eval(x))) if isinstance(x, str) else x)

    #Stack into (n_samples, 30, 3)
    sequences = np.stack([
        df['logins_seq'].to_list(),
        df['support_seq'].to_list(),
        df['data_seq'].to_list()
    ], axis=-1)
    # Encodes "Churn" as 0=No, 1=Yes
    le = LabelEncoder()
    labels = le.fit_transform(df['Churn'])

    # Returns a split into training/test datasets
    return train_test_split(sequences, labels, test_size=0.3, random_state=42)




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
