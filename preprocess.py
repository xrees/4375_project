import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path='data/rblx.csv', sequence_length=50):
    # Load the dataset
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data.sort_values(by='Date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Convert numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        # Ensure column is treated as a string before using string methods
        if data[col].dtype == 'object':
            data[col] = data[col].str.replace(',', '').str.replace('"', '')
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows with missing values
    data.dropna(subset=numeric_cols, inplace=True)

    # Derived features
    data['Momentum'] = data['Close'] - data['Close'].shift(10)
    data['Moving_Avg'] = data['Close'].rolling(window=10).mean()
    data.fillna(0, inplace=True)

    # Features and targets
    features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Momentum', 'Moving_Avg']].values
    targets = data['Close'].values.reshape(-1, 1)

    # Scaling
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)
    scaled_targets = target_scaler.fit_transform(targets)

    # Create sequences
    sequences, sequence_targets = [], []
    for i in range(len(scaled_features) - sequence_length):
        sequences.append(scaled_features[i:i + sequence_length])
        sequence_targets.append(scaled_targets[i + sequence_length])
    
    sequences = np.array(sequences)
    sequence_targets = np.array(sequence_targets)

    # Split data
    split_idx = int(0.8 * len(sequences))
    X_train, y_train = sequences[:split_idx], sequence_targets[:split_idx]
    X_test, y_test = sequences[split_idx:], sequence_targets[split_idx:]

    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, _, _ = preprocess_data()
    print("Data preprocessing completed successfully.")