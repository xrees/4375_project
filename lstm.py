import numpy as np
import copy

class LSTMCell:
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.0):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Weight matrices for input (input_dim features)
        self.Wi = np.random.randn(hidden_dim, input_dim) * np.sqrt(1 / input_dim)
        self.Wf = np.random.randn(hidden_dim, input_dim) * np.sqrt(1 / input_dim)
        self.Wo = np.random.randn(hidden_dim, input_dim) * np.sqrt(1 / input_dim)
        self.Wc = np.random.randn(hidden_dim, input_dim) * np.sqrt(1 / input_dim)

        # Weight matrices for hidden state
        self.Ui = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1 / hidden_dim)
        self.Uf = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1 / hidden_dim)
        self.Uo = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1 / hidden_dim)
        self.Uc = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1 / hidden_dim)

        # Bias terms
        self.bi = np.zeros((hidden_dim, 1))
        self.bf = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))

        # Storage for gates during forward pass for backpropagation
        self.o_s = []
        self.f_s = []
        self.i_s = []
        self.g_s = []
        self.h_s = []  # Store hidden states

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x_t, h_prev, c_prev, training=True):
        # Input gate
        i_t = self.sigmoid(self.Wi @ x_t + self.Ui @ h_prev + self.bi)
        # Forget gate
        f_t = self.sigmoid(self.Wf @ x_t + self.Uf @ h_prev + self.bf)
        # Output gate
        o_t = self.sigmoid(self.Wo @ x_t + self.Uo @ h_prev + self.bo)
        # Cell gate
        g_t = self.tanh(self.Wc @ x_t + self.Uc @ h_prev + self.bc)
        # New cell state
        c_t = f_t * c_prev + i_t * g_t
        # New hidden state
        h_t = o_t * self.tanh(c_t)

        # Apply dropout if in training mode
        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*h_t.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            h_t *= self.dropout_mask
        else:
            self.dropout_mask = np.ones_like(h_t)

        # Store the gates for backpropagation
        self.o_s.append(o_t)
        self.f_s.append(f_t)
        self.i_s.append(i_t)
        self.g_s.append(g_t)
        self.h_s.append(h_t)

        return h_t, c_t


class LSTM:
    def __init__(self, input_dim, hidden_dims, output_dim, reg_lambda=0.01, dropout_rate=0.0):
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.reg_lambda = reg_lambda  # Regularization parameter

        # Initialize LSTM cells for each layer
        self.layers = []
        dims = [input_dim] + hidden_dims
        for i in range(self.num_layers):
            self.layers.append(LSTMCell(dims[i], dims[i+1], dropout_rate=dropout_rate))

        # Output layer weights
        self.V = np.random.randn(output_dim, hidden_dims[-1]) * np.sqrt(1 / hidden_dims[-1])
        self.c = np.zeros((output_dim, 1))

        self.x_s = []  # Store the input sequence for backpropagation

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, training=True):
        # Initialize hidden and cell states for each layer
        h = [np.zeros((dim, 1)) for dim in self.hidden_dims]
        c = [np.zeros((dim, 1)) for dim in self.hidden_dims]

        self.cache = []  # To store intermediate states for backpropagation
        self.x_s = []    # Reset x_s for each new sequence

        # Reset gate storage for each layer
        for layer in self.layers:
            layer.o_s = []
            layer.f_s = []
            layer.i_s = []
            layer.g_s = []
            layer.h_s = []

        # Iterate through each time step in the input sequence
        for t in range(x.shape[0]):
            x_t = x[t].reshape(-1, 1)
            self.x_s.append(x_t)  # Store each input at time step t

            x_input = x_t
            for l, layer in enumerate(self.layers):
                h_l, c_l = layer.forward(x_input, h[l], c[l], training=training)
                x_input = h_l  # Pass output to the next layer
                h[l] = h_l
                c[l] = c_l

            # Store the hidden and cell states for backpropagation
            self.cache.append((h.copy(), c.copy()))

        # Compute the final output
        y_pred = self.V @ h[-1] + self.c
        return y_pred

    def backward(self, dy):
        # Initialize gradients for output layer
        dV = np.zeros_like(self.V)
        dc = np.zeros_like(self.c)

        # Initialize gradients for each LSTM layer
        d_layers = [
            {
                "Wi": np.zeros_like(layer.Wi),
                "Ui": np.zeros_like(layer.Ui),
                "bi": np.zeros_like(layer.bi),
                "Wf": np.zeros_like(layer.Wf),
                "Uf": np.zeros_like(layer.Uf),
                "bf": np.zeros_like(layer.bf),
                "Wo": np.zeros_like(layer.Wo),
                "Uo": np.zeros_like(layer.Uo),
                "bo": np.zeros_like(layer.bo),
                "Wc": np.zeros_like(layer.Wc),
                "Uc": np.zeros_like(layer.Uc),
                "bc": np.zeros_like(layer.bc)
            }
            for layer in self.layers
        ]

        # Initialize hidden state gradients for backpropagation through time
        dh_next = [np.zeros((dim, 1)) for dim in self.hidden_dims]
        dc_next = [np.zeros((dim, 1)) for dim in self.hidden_dims]

        # Backpropagation through time
        for t in reversed(range(len(self.cache))):
            h_states, c_states = self.cache[t]

            # Initialize dh for each layer
            dh = [np.zeros((dim, 1)) for dim in self.hidden_dims]

            # Backprop through output layer only at the last time step
            if t == len(self.cache) - 1:
                dy_t = dy
                # Gradient from output to last LSTM layer
                dh[-1] = self.V.T @ dy_t
                dV += dy_t @ h_states[-1].T
                dc += dy_t
            else:
                dy_t = np.zeros_like(dy)

            for l in reversed(range(self.num_layers)):
                layer = self.layers[l]
                h_prev = h_states[l-1] if l > 0 else self.x_s[t]
                h_curr = h_states[l]
                c_curr = c_states[l]

                dh_total = dh_next[l] + dh[l]

                # Apply dropout mask during backpropagation
                dh_total *= layer.dropout_mask

                dc_total = dc_next[l]

                # Gates derivatives
                do = dh_total * self.tanh(c_curr)
                do_raw = do * layer.o_s[t] * (1 - layer.o_s[t])

                dc_t = (dh_total * layer.o_s[t] * (1 - self.tanh(c_curr) ** 2)) + dc_total
                c_prev = self.cache[t-1][1][l] if t > 0 else np.zeros_like(c_curr)
                df = dc_t * c_prev
                df_raw = df * layer.f_s[t] * (1 - layer.f_s[t])

                di = dc_t * layer.g_s[t]
                di_raw = di * layer.i_s[t] * (1 - layer.i_s[t])

                dg = dc_t * layer.i_s[t]
                dg_raw = dg * (1 - layer.g_s[t] ** 2)

                # Accumulate gradients for this layer's weights and biases
                d_layers[l]["Wf"] += df_raw @ h_prev.T
                d_layers[l]["Uf"] += df_raw @ layer.h_s[t-1].T if t > 0 else np.zeros_like(layer.Uf)
                d_layers[l]["bf"] += df_raw

                d_layers[l]["Wi"] += di_raw @ h_prev.T
                d_layers[l]["Ui"] += di_raw @ layer.h_s[t-1].T if t > 0 else np.zeros_like(layer.Ui)
                d_layers[l]["bi"] += di_raw

                d_layers[l]["Wo"] += do_raw @ h_prev.T
                d_layers[l]["Uo"] += do_raw @ layer.h_s[t-1].T if t > 0 else np.zeros_like(layer.Uo)
                d_layers[l]["bo"] += do_raw

                d_layers[l]["Wc"] += dg_raw @ h_prev.T
                d_layers[l]["Uc"] += dg_raw @ layer.h_s[t-1].T if t > 0 else np.zeros_like(layer.Uc)
                d_layers[l]["bc"] += dg_raw

                # Correct computation of dh_prev using U matrices
                dh_prev = (
                    layer.Ui.T @ di_raw +
                    layer.Uf.T @ df_raw +
                    layer.Uo.T @ do_raw +
                    layer.Uc.T @ dg_raw
                )
                dh_next[l] = dh_prev
                dc_next[l] = dc_t * layer.f_s[t]

        # Store the gradients for parameter update
        self.d_layers = d_layers
        self.dV = dV
        self.dc = dc

    def update_parameters(self, learning_rate):
        # Update output layer weights with regularization
        self.V -= learning_rate * (self.dV + self.reg_lambda * self.V)
        self.c -= learning_rate * self.dc

        # Update weights and biases for each LSTM layer with regularization
        for i, layer in enumerate(self.layers):
            layer.Wi -= learning_rate * (self.d_layers[i]["Wi"] + self.reg_lambda * layer.Wi)
            layer.Ui -= learning_rate * (self.d_layers[i]["Ui"] + self.reg_lambda * layer.Ui)
            layer.bi -= learning_rate * self.d_layers[i]["bi"]

            layer.Wf -= learning_rate * (self.d_layers[i]["Wf"] + self.reg_lambda * layer.Wf)
            layer.Uf -= learning_rate * (self.d_layers[i]["Uf"] + self.reg_lambda * layer.Uf)
            layer.bf -= learning_rate * self.d_layers[i]["bf"]

            layer.Wo -= learning_rate * (self.d_layers[i]["Wo"] + self.reg_lambda * layer.Wo)
            layer.Uo -= learning_rate * (self.d_layers[i]["Uo"] + self.reg_lambda * layer.Uo)
            layer.bo -= learning_rate * self.d_layers[i]["bo"]

            layer.Wc -= learning_rate * (self.d_layers[i]["Wc"] + self.reg_lambda * layer.Wc)
            layer.Uc -= learning_rate * (self.d_layers[i]["Uc"] + self.reg_lambda * layer.Uc)
            layer.bc -= learning_rate * self.d_layers[i]["bc"]

    def reset_gradients(self):
        # Reset gradients before processing a new batch
        self.d_layers = [
            {
                "Wi": np.zeros_like(layer.Wi),
                "Ui": np.zeros_like(layer.Ui),
                "bi": np.zeros_like(layer.bi),
                "Wf": np.zeros_like(layer.Wf),
                "Uf": np.zeros_like(layer.Uf),
                "bf": np.zeros_like(layer.bf),
                "Wo": np.zeros_like(layer.Wo),
                "Uo": np.zeros_like(layer.Uo),
                "bo": np.zeros_like(layer.bo),
                "Wc": np.zeros_like(layer.Wc),
                "Uc": np.zeros_like(layer.Uc),
                "bc": np.zeros_like(layer.bc)
            }
            for layer in self.layers
        ]
        self.dV = np.zeros_like(self.V)
        self.dc = np.zeros_like(self.c)

    def copy(self):
        return copy.deepcopy(self)