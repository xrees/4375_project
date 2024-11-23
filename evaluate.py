import numpy as np
from lstm import LSTM
from preprocess import preprocess_data
import os


def load_model_parameters(model, num_layers):
    """
    Load the saved model parameters for each LSTM layer.
    Handles LSTMCell attributes directly.
    """
    for i in range(num_layers):
        try:
            layer = model.layers[i]  # Access the LSTMCell object
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
        except FileNotFoundError as e:
            print(f"Error: Missing file for layer {i}. Ensure training saved all parameters.")
            raise e


def compute_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_accuracy(y_true, y_pred, threshold=10.0):
    """
    Compute accuracy based on a threshold:
    Counts predictions within +/- threshold as correct.
    """
    correct = np.abs(y_true - y_pred) < threshold
    accuracy = np.mean(correct) * 100 
    return accuracy


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, _, target_scaler = preprocess_data()

    # starting LSTM model w/ training architecture
    input_dim = X_train.shape[2]
    hidden_dims = [64, 32, 16] 
    output_dim = 1
    model = LSTM(input_dim, hidden_dims, output_dim, reg_lambda=0.001)

    # model params
    print("Loading model parameters...")
    load_model_parameters(model, num_layers=len(hidden_dims))
    print("Model parameters loaded successfully.")

    # predictions
    print("Evaluating the model...")
    y_pred_test = [model.forward(x_seq).item() for x_seq in X_test]
    y_pred_test = np.array(y_pred_test).reshape(-1, 1)
    y_pred_test = target_scaler.inverse_transform(y_pred_test)
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    # evaluation
    test_rmse = compute_rmse(y_test_original, y_pred_test)
    test_accuracy = compute_accuracy(y_test_original, y_pred_test)
    print("\nEvaluation Results:")
    print(f"Test RMSE: {test_rmse}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    print("\nSample Predictions:")
    for i in range(10):
        print(f"Actual: {y_test_original[i][0]:.2f}, Predicted: {y_pred_test[i][0]:.2f}")
