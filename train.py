# train.py

import numpy as np
import copy  # Import copy for deep copying the model
from lstm import LSTM
from preprocess import preprocess_data

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def compute_accuracy(y_true, y_pred, threshold=0.02):  # Using 2% threshold
    correct = np.abs(y_true - y_pred) / y_true < threshold
    accuracy = np.mean(correct) * 100  # Accuracy in percentage
    return accuracy

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test, _, target_scaler = preprocess_data()

    input_dim = X_train.shape[2]
    hidden_dims = [64, 32]  # Adjusted for potential performance improvement
    output_dim = 1
    epochs = 100
    learning_rate = 0.005
    reg_lambda = 0.001  # Reduced regularization
    dropout_rate = 0.1  # Reduced dropout rate
    patience = 10  # Increased patience for early stopping

    model = LSTM(input_dim, hidden_dims, output_dim, reg_lambda, dropout_rate=dropout_rate)

    best_val_loss = float('inf')
    patience_counter = 0

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch+1}/{epochs}")
        permutation = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        total_loss = 0

        for x_seq, y_true in zip(X_train_shuffled, y_train_shuffled):
            # Forward pass
            y_pred = model.forward(x_seq, training=True)
            loss = np.square(y_pred - y_true.reshape(-1, 1)) / 2
            total_loss += loss.item()

            # Backward pass and parameter update
            dy = y_pred - y_true.reshape(-1, 1)
            model.backward(dy)
            model.update_parameters(learning_rate)  # Ensure learning_rate is passed here

        avg_loss = total_loss / len(X_train)
        training_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss}')

        # Validation
        val_loss = 0
        for x_seq, y_true in zip(X_val, y_val):
            y_pred = model.forward(x_seq, training=False)
            loss = np.square(y_pred - y_true.reshape(-1, 1)) / 2
            val_loss += loss.item()
        avg_val_loss = val_loss / len(X_val)
        validation_losses.append(avg_val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model using copy.deepcopy
            best_model = copy.deepcopy(model)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Use the best model for evaluation
    model = best_model

    # Evaluate on training data
    y_pred_train = []
    for x_seq in X_train:
        y_pred = model.forward(x_seq, training=False)
        y_pred_train.append(y_pred.item())
    y_pred_train = np.array(y_pred_train).reshape(-1, 1)
    y_pred_train = target_scaler.inverse_transform(y_pred_train)
    y_train_original = target_scaler.inverse_transform(y_train)

    training_rmse = compute_rmse(y_train_original, y_pred_train)
    training_accuracy = compute_accuracy(y_train_original, y_pred_train)

    # Evaluate on test data
    y_pred_test = []
    for x_seq in X_test:
        y_pred = model.forward(x_seq, training=False)
        y_pred_test.append(y_pred.item())
    y_pred_test = np.array(y_pred_test).reshape(-1, 1)
    y_pred_test = target_scaler.inverse_transform(y_pred_test)
    y_test_original = target_scaler.inverse_transform(y_test)

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
    print("Error Function = Mean Squared Error")
    print(f"Regularization Parameter = {reg_lambda}")
    print(f"Train/Validation/Test Split = 70:15:15")
    print(f"Size of dataset = {len(X_train) + len(X_val) + len(X_test)}")
    print("\nResults:")
    print(f"Training Accuracy = {training_accuracy:.2f}%")
    print(f"Test Accuracy = {test_accuracy:.2f}%")
    print(f"Training RMSE = {training_rmse}")
    print(f"Test RMSE = {test_rmse}")

    # Save model parameters after training
    model.save_parameters('model/')