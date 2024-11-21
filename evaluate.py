# evaluate.py

import numpy as np
from lstm import LSTM
from preprocess import preprocess_data

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def load_model_parameters(model):
    for i, layer in enumerate(model.layers):
        layer.Wi = np.load(f'model/layer{i}_Wi.npy')
        layer.Ui = np.load(f'model/layer{i}_Ui.npy')
        layer.bi = np.load(f'model/layer{i}_bi.npy')
        layer.Wf = np.load(f'model/layer{i}_Wf.npy')
        layer.Uf = np.load(f'model/layer{i}_Uf.npy')
        layer.bf = np.load(f'model/layer{i}_bf.npy')
        layer.Wo = np.load(f'model/layer{i}_Wo.npy')
        layer.Uo = np.load(f'model/layer{i}_Uo.npy')
        layer.bo = np.load(f'model/layer{i}_bo.npy')
        layer.Wc = np.load(f'model/layer{i}_Wc.npy')
        layer.Uc = np.load(f'model/layer{i}_Uc.npy')
        layer.bc = np.load(f'model/layer{i}_bc.npy')

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler = preprocess_data()

    input_dim = X_train.shape[2]
    hidden_dims = [32, 16]  # Same as in training
    output_dim = y_train.shape[1]
    reg_lambda = 0.01

    model = LSTM(input_dim, hidden_dims, output_dim, reg_lambda, dropout_rate=0.0)  # Disable dropout
    load_model_parameters(model)

    # Compute test predictions
    predictions = []
    for x_seq in X_test:
        y_pred = model.forward(x_seq, training=False)
        predictions.append(y_pred.item())

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = target_scaler.inverse_transform(predictions)
    y_test_original = target_scaler.inverse_transform(y_test)

    # Calculate RMSE
    test_rmse = compute_rmse(y_test_original, predictions)

    # Output parameters and test results
    print("Parameters Chosen:")
    print("Neural Net:")
    print(f"Number of layers = {len(hidden_dims)}")
    print(f"Neurons = {tuple(hidden_dims)}")
    print("Error Function = RMSE")
    print(f"Regularization Parameter = {reg_lambda}")
    print(f"Train/Validation/Test Split = 70:15:15")
    print(f"Size of dataset = {len(X_train) + len(X_val) + len(X_test)}")
    print("\nResults:")
    print(f"Test RMSE = {test_rmse}")