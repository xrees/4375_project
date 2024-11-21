
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path='data/nvda.csv', sequence_length=50):
    data = pd.read_csv(file_path)

    # Ensure that the necessary columns are numeric
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    data.dropna(subset=numeric_cols, inplace=True)

    # Add technical indicators
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA100'] = data['Close'].rolling(window=100).mean()
    data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['Momentum'] = data['Close'] - data['Close'].shift(1)
    data.dropna(inplace=True)

    # Select features and target
    features = data[['Open', 'High', 'Low', 'Close', 'Volume',
                     'MA10', 'MA50', 'MA100', 'EMA10', 'EMA50', 'Momentum']].values
    targets = data['Close'].values.reshape(-1, 1)

    # Scale features and target separately
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(features)

    target_scaler = StandardScaler()
    scaled_targets = target_scaler.fit_transform(targets)

    sequences = []
    sequence_targets = []
    for i in range(len(scaled_features) - sequence_length):
        sequences.append(scaled_features[i:i+sequence_length])
        sequence_targets.append(scaled_targets[i+sequence_length])

    sequences = np.array(sequences)
    sequence_targets = np.array(sequence_targets)

    # Split data into training, validation, and test sets (70%, 15%, 15%)
    total_samples = len(sequences)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)

    X_train = sequences[:train_size]
    y_train = sequence_targets[:train_size]
    X_val = sequences[train_size:train_size+val_size]
    y_val = sequence_targets[train_size:train_size+val_size]
    X_test = sequences[train_size+val_size:]
    y_test = sequence_targets[train_size+val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler = preprocess_data()
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/y_val.npy', y_val)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)