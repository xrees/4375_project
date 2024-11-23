import numpy as np
from lstm import LSTM
from preprocess import preprocess_data

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def compute_accuracy(y_true, y_pred, threshold=10.0):
    correct = np.abs(y_true - y_pred) < threshold
    accuracy = np.mean(correct) * 100  
    return accuracy

class ReduceLROnPlateau:
    def __init__(self, monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait = 0
        self.current_lr = None

    def update(self, current_loss, optimizer_lr):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(optimizer_lr * self.factor, self.min_lr)
                self.wait = 0
                print(f"Reducing learning rate from {optimizer_lr} to {new_lr}")
                return new_lr
        return optimizer_lr

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, _, target_scaler = preprocess_data()

    input_dim = X_train.shape[2]
    hidden_dims = [64, 32, 16]
    output_dim = 1
    epochs = 50
    learning_rate = 0.001
    reg_lambda = 0.001
    lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)

    model = LSTM(input_dim, hidden_dims, output_dim, reg_lambda)

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for x_seq, y_true in zip(X_train, y_train):
            y_pred = model.forward(x_seq)
            loss = np.square(y_pred - y_true.reshape(-1, 1)) / 2
            total_loss += loss

            dy = y_pred - y_true.reshape(-1, 1)
            model.backward(dy)
            model.update_parameters(learning_rate)

        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss[0][0]}")

        learning_rate = lr_scheduler.update(avg_loss[0][0], learning_rate)

    # training rmse
    y_pred_train = [model.forward(x_seq).item() for x_seq in X_train]
    y_pred_train = np.array(y_pred_train).reshape(-1, 1)
    y_pred_train = target_scaler.inverse_transform(y_pred_train)
    y_train_original = target_scaler.inverse_transform(y_train.reshape(-1, 1))

    training_rmse = compute_rmse(y_train_original, y_pred_train)
    training_accuracy = compute_accuracy(y_train_original, y_pred_train)

    # testing rmse
    y_pred_test = [model.forward(x_seq).item() for x_seq in X_test]
    y_pred_test = np.array(y_pred_test).reshape(-1, 1)
    y_pred_test = target_scaler.inverse_transform(y_pred_test)
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    test_rmse = compute_rmse(y_test_original, y_pred_test)
    test_accuracy = compute_accuracy(y_test_original, y_pred_test)

    print("\nResults:")
    print(f"Training Accuracy = {training_accuracy:.2f}%")
    print(f"Test Accuracy = {test_accuracy:.2f}%")
    print(f"Training RMSE = {training_rmse}")
    print(f"Test RMSE = {test_rmse}")

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
