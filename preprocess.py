import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path='data/nvda.csv', sequence_length=50):
    data = pd.read_csv(file_path)

    # Ensure that the necessary columns are numeric
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    data.dropna(subset=numeric_cols, inplace=True)

    # Select features and target
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    targets = data['Close'].values.reshape(-1, 1)

    # Scale features and target separately
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)

    target_scaler = MinMaxScaler()
    scaled_targets = target_scaler.fit_transform(targets)

    sequences = []
    sequence_targets = []
    for i in range(len(scaled_features) - sequence_length):
        sequences.append(scaled_features[i:i+sequence_length])
        sequence_targets.append(scaled_targets[i+sequence_length])

    sequences = np.array(sequences)
    sequence_targets = np.array(sequence_targets)

    split = int(0.8 * len(sequences))
    X_train = sequences[:split]
    y_train = sequence_targets[:split]
    X_test = sequences[split:]
    y_test = sequence_targets[split:]

    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, feature_scaler, target_scaler = preprocess_data()
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)