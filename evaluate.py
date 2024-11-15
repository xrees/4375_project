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
    X_train, y_train, X_test, y_test, scaler = preprocess_data()
    
    input_dim = 1
    hidden_dims = [8, 8, 4, 2]
    output_dim = 1
    reg_lambda = 0.6
    train_test_split = '80:20'
    dataset_size = len(X_train) + len(X_test)
    
    model = LSTM(input_dim, hidden_dims, output_dim, reg_lambda)
    load_model_parameters(model)
    
    # Compute test predictions
    predictions = []
    for x_seq in X_test:
        y_pred = model.forward(x_seq)
        predictions.append(y_pred.item())
    
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate RMSE
    test_rmse = compute_rmse(y_test_original, predictions)
    
    # Output parameters and test results
    print("Parameters Chosen:")
    print("Neural Net:")
    print(f"Number of layers = {len(hidden_dims)}")
    print(f"Neurons = {tuple(hidden_dims)}")
    print("Error Function = RMSE")
    print(f"Regularization Parameter = {reg_lambda}")
    print(f"Train/Test Split = {train_test_split}")
    print(f"Size of dataset = {dataset_size}")
    print("\nResults:")
    print(f"Test RMSE = {test_rmse[0]}")
