import numpy as np
from lstm import LSTM
from preprocess import preprocess_data

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def compute_accuracy(y_true, y_pred, threshold=10.0):
    # Compute accuracy based on a threshold: counts predictions within +/- threshold as correct
    correct = np.abs(y_true - y_pred) < threshold
    accuracy = np.mean(correct) * 100  # Accuracy in percentage
    return accuracy

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, _, target_scaler = preprocess_data()  # Ignore feature_scaler
    
    input_dim = X_train.shape[2]
    hidden_dims = [64, 32, 16]  # Neurons in each LSTM layer
    output_dim = 1
    epochs = 50
    learning_rate = 0.001
    reg_lambda = 0.001  # Regularization parameter
    train_test_split = '80:20'
    dataset_size = len(X_train) + len(X_test)
    
    model = LSTM(input_dim, hidden_dims, output_dim, reg_lambda)
    
    # Training loop with parameter updates
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for x_seq, y_true in zip(X_train, y_train):
            y_pred = model.forward(x_seq)
            loss = np.square(y_pred - y_true.reshape(-1, 1)) / 2
            total_loss += loss
            
            # Compute gradients and update parameters
            dy = y_pred - y_true.reshape(-1, 1)
            model.backward(dy)  # Backpropagation
            model.update_parameters(learning_rate)  # Update weights
        
        avg_loss = total_loss / len(X_train)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss[0][0]}')

    # Calculate training RMSE and accuracy
    y_pred_train = []
    for x_seq in X_train:
        y_pred = model.forward(x_seq)
        y_pred_train.append(y_pred.item())
    y_pred_train = np.array(y_pred_train).reshape(-1, 1)
    y_pred_train = target_scaler.inverse_transform(y_pred_train)
    y_train_original = target_scaler.inverse_transform(y_train.reshape(-1, 1))
    
    training_rmse = compute_rmse(y_train_original, y_pred_train)
    training_accuracy = compute_accuracy(y_train_original, y_pred_train)
    
    # Calculate test RMSE and accuracy
    y_pred_test = []
    for x_seq in X_test:
        y_pred = model.forward(x_seq)
        y_pred_test.append(y_pred.item())
    y_pred_test = np.array(y_pred_test).reshape(-1, 1)
    y_pred_test = target_scaler.inverse_transform(y_pred_test)
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    test_rmse = compute_rmse(y_test_original, y_pred_test)
    test_accuracy = compute_accuracy(y_test_original, y_pred_test)
    # Examine a few test predictions
    for i in range(5):
        print(f"Sample {i+1}: True Value = {y_test_original[i][0]:.2f}, Predicted Value = {y_pred_test[i][0]:.2f}")


    # Output parameters and training results
    print("Parameters Chosen:")
    print("Neural Net:")
    print(f"Number of layers = {len(hidden_dims)}")
    print(f"Neurons = {tuple(hidden_dims)}")
    print("Error Function = RMSE")
    print(f"Regularization Parameter = {reg_lambda}")
    print(f"Train/Test Split = {train_test_split}")
    print(f"Size of dataset = {dataset_size}")
    print("\nResults:")
    print(f"Training Accuracy = {training_accuracy:.2f}%")
    print(f"Test Accuracy = {test_accuracy:.2f}%")
    print(f"Training RMSE = {training_rmse}")
    print(f"Test RMSE = {test_rmse}")

    # Save model parameters after training
    for i, layer in enumerate(model.layers):
        np.save(f'model/layer{i}_Wi.npy', layer.Wi)
        np.save(f'model/layer{i}_Ui.npy', layer.Ui)
        np.save(f'model/layer{i}_bi.npy', layer.bi)
        np.save(f'model/layer{i}_Wf.npy', layer.Wf)
        np.save(f'model/layer{i}_Uf.npy', layer.Uf)
        np.save(f'model/layer{i}_bf.npy', layer.bf)
        np.save(f'model/layer{i}_Wo.npy', layer.Wo)
        np.save(f'model/layer{i}_Uo.npy', layer.Uo)
        np.save(f'model/layer{i}_bo.npy', layer.bo)
        np.save(f'model/layer{i}_Wc.npy', layer.Wc)
        np.save(f'model/layer{i}_Uc.npy', layer.Uc)
        np.save(f'model/layer{i}_bc.npy', layer.bc)